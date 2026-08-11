from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from daxalgo_numeric_transformer.chmt_q import ChmtQBatch, ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.model import (
    BookQueryPool,
    ChmtQAttention,
    ChmtQTransformer,
    MaskedPatchPool,
    NumericTimeEncoding,
    RelativeTimeBias,
    RMSNorm,
    SwiGLU,
)


def _availability_batch(config: ChmtQConfig) -> ChmtQBatch:
    batch_size = 2
    bars_mask = torch.zeros(batch_size, config.bar_steps, dtype=torch.bool)
    l1_mask = torch.zeros(batch_size, config.micro_steps, dtype=torch.bool)
    level_mask = torch.zeros(
        batch_size,
        config.micro_steps,
        config.l2_sides,
        config.l2_levels,
        dtype=torch.bool,
    )
    bars_mask[0, -1] = True
    level_mask[0, -1, 0, 0] = True
    l1_mask[1, -1] = True
    return ChmtQBatch(
        bars=torch.zeros(batch_size, config.bar_steps, config.bar_features),
        bars_mask=bars_mask,
        l1=torch.zeros(batch_size, config.micro_steps, config.l1_features),
        l1_mask=l1_mask,
        l2=torch.zeros(
            batch_size,
            config.micro_steps,
            config.l2_sides,
            config.l2_levels,
            config.l2_features,
        ),
        l2_level_mask=level_mask,
        l2_snapshot_mask=level_mask.any(dim=(-1, -2)),
        quality=torch.zeros(batch_size, config.quality_features),
    ).validate(config)


def test_elapsed_time_bias_and_temperature_match_joint_attention_equation() -> None:
    epsilon = 1.0e-6
    attention = ChmtQAttention(4, 2, qk_epsilon=epsilon).eval()
    relative_bias = RelativeTimeBias(2, time_scale=2.0)
    temperatures = torch.tensor([0.5, 2.0])
    decays = torch.tensor([0.25, 1.5])
    with torch.no_grad():
        for projection in (
            attention.q_projection,
            attention.k_projection,
            attention.v_projection,
            attention.output_projection,
        ):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
        attention.log_temperature.copy_(temperatures.log())
        relative_bias.gamma.copy_(torch.log(torch.expm1(decays)))

    query = torch.tensor([[[1.0, 0.0, 1.0, 1.0]]])
    memory = torch.tensor(
        [[[1.0, 0.0, 1.0, -1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, -1.0, 1.0]]]
    )
    query_ages = torch.tensor([0.0])
    key_ages = torch.tensor([4.0, 2.0, 0.0])

    actual_bias = relative_bias(query_ages, key_ages)
    _, actual_weights = attention(
        query,
        memory,
        additive_bias=actual_bias,
        need_weights=True,
    )

    elapsed = (key_ages.unsqueeze(0) - query_ages.unsqueeze(1)).clamp_min(0.0)
    distances = torch.log1p(elapsed / 2.0)
    expected_bias = -decays[None, :, None, None] * distances[None, None, :, :]
    query_heads = query.view(1, 1, 2, 2).transpose(1, 2)
    key_heads = memory.view(1, 3, 2, 2).transpose(1, 2)
    normalized_query = query_heads / torch.sqrt(
        query_heads.square().mean(dim=-1, keepdim=True) + epsilon
    )
    normalized_keys = key_heads / torch.sqrt(
        key_heads.square().mean(dim=-1, keepdim=True) + epsilon
    )
    content_scores = normalized_query @ normalized_keys.transpose(-2, -1)
    content_scores *= temperatures[None, :, None, None] / math.sqrt(2.0)
    expected_weights = torch.softmax(content_scores + expected_bias, dim=-1)

    torch.testing.assert_close(actual_bias, expected_bias)
    torch.testing.assert_close(actual_weights, expected_weights)
    assert actual_bias[0, 0, 0, 0] < actual_bias[0, 0, 0, 1] < 0.0
    assert actual_bias[..., -1].count_nonzero() == 0


def test_book_query_pool_matches_masked_hierarchical_reference() -> None:
    epsilon = 1.0e-6
    pool = BookQueryPool(3, rms_epsilon=epsilon).eval()
    with torch.no_grad():
        pool.query.copy_(torch.tensor([0.5, -1.0, 2.0]))
        pool.level_norm.gain.copy_(torch.tensor([1.0, 1.5, 0.75]))

    levels = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [float("nan"), float("nan"), float("nan")], [3.0, 1.0, 2.0]],
            [[7.0, -5.0, 1.0], [2.0, 4.0, 8.0], [-3.0, 6.0, 9.0]],
        ]
    )
    level_mask = torch.tensor([[True, False, True], [False, False, False]])

    actual, actual_weights = pool(levels, level_mask, need_weights=True)

    safe_levels = torch.where(level_mask.unsqueeze(-1), levels, torch.zeros_like(levels))
    inverse_rms = torch.rsqrt(safe_levels.square().mean(dim=-1, keepdim=True) + epsilon)
    normalized = safe_levels * inverse_rms * pool.level_norm.gain
    scores = (normalized @ pool.query) / math.sqrt(3.0)
    expected_weights = torch.zeros_like(scores)
    expected_weights[0, [0, 2]] = torch.softmax(scores[0, [0, 2]], dim=0)
    expected = (expected_weights.unsqueeze(-1) * safe_levels).sum(dim=1)

    torch.testing.assert_close(actual_weights, expected_weights)
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all()
    assert actual[1].count_nonzero() == 0


def test_fusion_gates_apply_hard_availability_before_normalization_and_bias() -> None:
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = _availability_batch(config)
    gate_probabilities = torch.tensor([0.2, 0.4, 0.8])
    with torch.no_grad():
        for gate, probability in zip(model.gate_mlps, gate_probabilities, strict=True):
            first = gate[0]
            final = gate[2]
            assert isinstance(first, nn.Linear)
            assert isinstance(final, nn.Linear)
            first.weight.zero_()
            first.bias.zero_()
            final.weight.zero_()
            final.bias.fill_(torch.logit(probability))
        encoded = model.encode_modalities(batch)

    availability = batch.modality_availability()
    expected_gates = gate_probabilities[None, :] * availability
    expected_normalized = expected_gates / (
        expected_gates.sum(dim=1, keepdim=True) + config.gate_epsilon
    )
    expected_token_gates = torch.cat(
        (
            expected_normalized[:, 0:1].expand(-1, config.bar_steps),
            expected_normalized[:, 1:2].expand(-1, config.micro_patch_steps),
            expected_normalized[:, 2:3].expand(-1, config.micro_patch_steps),
        ),
        dim=1,
    )
    expected_bias = torch.log(expected_token_gates + config.gate_epsilon)[:, None, None, :]

    torch.testing.assert_close(encoded.gates, expected_gates)
    torch.testing.assert_close(encoded.normalized_gates, expected_normalized)
    torch.testing.assert_close(encoded.memory_bias, expected_bias)
    assert encoded.gates[~availability].count_nonzero() == 0
    assert encoded.memory_mask[0, config.bar_steps : config.bar_steps + 60].count_nonzero() == 0
    assert encoded.memory_mask[1, : config.bar_steps].count_nonzero() == 0
    assert encoded.memory_mask[1, config.bar_steps + 60 :].count_nonzero() == 0


def test_model_initialization_matches_documented_numeric_invariants() -> None:
    torch.manual_seed(73)
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config)

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert module.bias is not None
            assert module.bias.count_nonzero() == 0
            bound = math.sqrt(6.0 / (module.in_features + module.out_features))
            assert float(module.weight.abs().max()) <= bound + 1.0e-7
            assert module.weight.count_nonzero() == module.weight.numel()

    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.testing.assert_close(module.gain, torch.ones_like(module.gain))
        elif isinstance(module, ChmtQAttention):
            assert module.log_temperature.count_nonzero() == 0
        elif isinstance(module, RelativeTimeBias):
            assert module.gamma.count_nonzero() == 0

    layer_scales = [
        parameter
        for name, parameter in model.named_parameters()
        if name.endswith("layer_scale")
    ]
    assert layer_scales
    for parameter in layer_scales:
        torch.testing.assert_close(
            parameter,
            torch.full_like(parameter, config.layer_scale_initial),
        )

    embeddings = torch.cat(
        tuple(
            parameter.detach().flatten()
            for parameter in (
                model.modality_embedding,
                model.side_embedding,
                model.rank_embedding,
                model.horizon_queries,
                model.book_pool.query,
                model.l1_patch_pool.within_patch_embedding,
                model.l2_patch_pool.within_patch_embedding,
            )
        )
    )
    assert 0.015 < float(embeddings.std(unbiased=False)) < 0.025
    assert float(embeddings.abs().max()) < 0.1
    output_expected_std = math.sqrt(
        2.0 / (model.output_head.in_features + model.output_head.out_features)
    )
    assert float(model.output_head.weight.std(unbiased=False)) == pytest.approx(
        output_expected_std,
        rel=0.15,
    )


def test_numeric_time_encoding_matches_explicit_physical_feature_reference() -> None:
    encoding = NumericTimeEncoding(2, (2.0, 4.0), hidden_dim=3, context_max=8.0)
    with torch.no_grad():
        encoding.input_projection.weight.copy_(
            torch.tensor(
                [
                    [0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
                    [-0.3, 0.2, 0.1, -0.4, 0.6, 0.5],
                    [0.7, -0.1, 0.2, 0.3, 0.4, -0.2],
                ]
            )
        )
        encoding.input_projection.bias.copy_(torch.tensor([0.05, -0.1, 0.2]))
        encoding.output_projection.weight.copy_(
            torch.tensor([[0.5, -0.25, 0.75], [-0.4, 0.6, 0.2]])
        )
        encoding.output_projection.bias.copy_(torch.tensor([0.1, -0.2]))

    ages = torch.tensor([0.0, 2.0, 8.0])
    phases = ages[:, None] / torch.tensor([2.0, 4.0])
    features = torch.cat(
        (
            (ages / 8.0)[:, None],
            (torch.log1p(ages) / math.log(9.0))[:, None],
            torch.sin(phases),
            torch.cos(phases),
        ),
        dim=-1,
    )
    hidden = features @ encoding.input_projection.weight.T + encoding.input_projection.bias
    hidden = hidden * torch.sigmoid(hidden)
    expected = hidden @ encoding.output_projection.weight.T + encoding.output_projection.bias

    torch.testing.assert_close(encoding(ages), expected)


def test_masked_patch_pool_matches_explicit_reference_and_zeroes_empty_patch() -> None:
    pool = MaskedPatchPool(2, hidden_dim=2, patch_size=4).eval()
    with torch.no_grad():
        pool.within_patch_embedding.copy_(
            torch.tensor([[0.1, 0.2], [0.3, -0.1], [-0.2, 0.4], [0.5, 0.6]])
        )
        pool.score_projection.weight.copy_(torch.tensor([[0.5, -0.25], [0.75, 0.4]]))
        pool.score_projection.bias.copy_(torch.tensor([0.1, -0.2]))
        pool.score_output.weight.copy_(torch.tensor([[0.8, -0.3]]))
        pool.score_output.bias.fill_(0.05)

    frames = torch.tensor(
        [
            [[1.0, 2.0], [float("nan"), float("nan")], [3.0, 1.0], [2.0, 4.0]],
            [[7.0, 3.0], [2.0, 9.0], [5.0, 4.0], [1.0, 8.0]],
        ]
    )
    mask = torch.tensor([[True, False, True, True], [False, False, False, False]])

    actual, patch_mask = pool(frames, mask)

    offsets = frames + pool.within_patch_embedding[None, :, :]
    safe = torch.where(mask.unsqueeze(-1), offsets, torch.zeros_like(offsets))
    hidden = safe @ pool.score_projection.weight.T + pool.score_projection.bias
    scores = torch.tanh(hidden) @ pool.score_output.weight.T + pool.score_output.bias
    weights = torch.zeros_like(scores.squeeze(-1))
    weights[0, [0, 2, 3]] = torch.softmax(scores[0, [0, 2, 3], 0], dim=0)
    expected = (weights.unsqueeze(-1) * safe).sum(dim=1, keepdim=True)

    torch.testing.assert_close(actual, expected)
    assert patch_mask.tolist() == [[True], [False]]
    assert actual[1].count_nonzero() == 0


def test_swiglu_matches_explicit_silu_gate_reference() -> None:
    module = SwiGLU(3, 2)
    with torch.no_grad():
        module.gate_projection.weight.copy_(torch.tensor([[0.2, -0.4, 0.6], [0.5, 0.3, -0.1]]))
        module.gate_projection.bias.copy_(torch.tensor([0.1, -0.2]))
        module.value_projection.weight.copy_(torch.tensor([[0.7, 0.2, -0.5], [-0.3, 0.8, 0.4]]))
        module.value_projection.bias.copy_(torch.tensor([0.05, 0.15]))
        module.output_projection.weight.copy_(torch.tensor([[0.4, -0.6], [0.2, 0.5], [-0.7, 0.3]]))
        module.output_projection.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    values = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 0.25, 2.0]])
    gate = values @ module.gate_projection.weight.T + module.gate_projection.bias
    value = values @ module.value_projection.weight.T + module.value_projection.bias
    hidden = (gate * torch.sigmoid(gate)) * value
    expected = hidden @ module.output_projection.weight.T + module.output_projection.bias

    torch.testing.assert_close(module(values), expected)


def test_horizon_query_construction_matches_numeric_reference() -> None:
    model = ChmtQTransformer(ChmtQConfig.tiny()).eval()
    context = torch.linspace(-0.5, 0.5, model.config.d_model).repeat(2, 1)
    horizons = torch.arange(1, 6, dtype=torch.float32)
    features = torch.stack(
        (
            horizons / 5.0,
            torch.log1p(horizons),
            horizons * model.config.forecast_interval_seconds,
        ),
        dim=-1,
    )
    first = model.horizon_mlp[0]
    final = model.horizon_mlp[2]
    assert isinstance(first, nn.Linear)
    assert isinstance(final, nn.Linear)
    hidden = features @ first.weight.T + first.bias
    encoded = (hidden * torch.sigmoid(hidden)) @ final.weight.T + final.bias
    expected = (
        model.horizon_queries[None, :, :]
        + encoded[None, :, :]
        + (context @ model.context_projection.weight.T + model.context_projection.bias)[:, None, :]
    )

    torch.testing.assert_close(model._make_horizon_queries(context), expected)
