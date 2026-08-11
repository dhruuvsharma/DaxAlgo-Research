from __future__ import annotations

from dataclasses import replace

import torch

from daxalgo_numeric_transformer.chmt_q import ChmtQBatch, ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.model import (
    ChmtQAttention,
    ChmtQDecoderBlock,
    ChmtQEncoderBlock,
    ChmtQTransformer,
    MaskedPatchPool,
    RMSNorm,
    SwiGLU,
)
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch


def _batch(config: ChmtQConfig, *, batch_size: int = 1) -> ChmtQBatch:
    generator = torch.Generator().manual_seed(23)
    bars_mask = torch.ones(batch_size, 64, dtype=torch.bool)
    l1_mask = torch.ones(batch_size, 240, dtype=torch.bool)
    level_mask = torch.zeros(batch_size, 240, 2, 10, dtype=torch.bool)
    level_mask[:, :, :, :3] = True
    return ChmtQBatch(
        bars=torch.randn(batch_size, 64, 16, generator=generator),
        bars_mask=bars_mask,
        l1=torch.randn(batch_size, 240, 16, generator=generator),
        l1_mask=l1_mask,
        l2=torch.randn(batch_size, 240, 2, 10, 12, generator=generator),
        l2_level_mask=level_mask,
        l2_snapshot_mask=level_mask.any(dim=(-1, -2)),
        quality=torch.rand(batch_size, 16, generator=generator),
    ).validate(config)


def test_reference_model_matches_corrected_literal_parameter_ledger() -> None:
    config = ChmtQConfig.reference()
    model = ChmtQTransformer(config)

    model.assert_reference_parameter_count()
    assert model.parameter_count == 2_089_554
    assert model.output_head.weight.numel() + model.output_head.bias.numel() == 2_193
    temperatures = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("log_temperature")
    )
    temporal_gammas = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.endswith("relative_time_bias.gamma")
    )
    assert temperatures == 56
    assert temporal_gammas == 36
    assert not any(isinstance(module, torch.nn.LayerNorm) for module in model.modules())
    assert sum(isinstance(module, RMSNorm) for module in model.modules()) > 0
    assert not hasattr(model, "l1_frame_time_encoding")
    assert not hasattr(model, "level_time_encoding")


def test_rmsnorm_swiglu_and_layerscale_match_selected_block_contract() -> None:
    norm = RMSNorm(3, epsilon=1.0e-6)
    values = torch.tensor([[3.0, 4.0, 0.0]])
    actual = norm(values)
    expected = values / torch.sqrt(values.square().mean(dim=-1, keepdim=True) + 1.0e-6)
    torch.testing.assert_close(actual, expected)

    block = ChmtQEncoderBlock(
        8,
        2,
        16,
        0.0,
        rms_epsilon=1.0e-6,
        qk_epsilon=1.0e-6,
        layer_scale_initial=0.1,
        causal=False,
    )
    assert isinstance(block.ffn, SwiGLU)
    torch.testing.assert_close(block.attention_layer_scale, torch.full((8,), 0.1))
    torch.testing.assert_close(block.ffn_layer_scale, torch.full((8,), 0.1))


def test_forward_raw_training_and_distribution_contracts() -> None:
    torch.manual_seed(29)
    config = ChmtQConfig.tiny(decoder_layers=2)
    model = ChmtQTransformer(config).eval()
    batch = _batch(config)

    with torch.no_grad():
        raw = model(batch)
        training = model.forward_training(batch)
        distribution = model.forward_distribution(batch)
        encoded = model.encode_modalities(batch)

    assert raw.shape == (1, 5, 17)
    torch.testing.assert_close(raw, training.raw.final, rtol=0, atol=0)
    torch.testing.assert_close(raw, model.forward_raw(**batch.as_model_kwargs()), rtol=0, atol=0)
    torch.testing.assert_close(distribution.quantiles, training.final.quantiles, rtol=0, atol=0)
    assert training.raw.auxiliary.shape == (1, 5, 17)
    assert not torch.equal(training.raw.auxiliary, training.raw.final)
    assert torch.all(distribution.q10 <= distribution.median)
    assert torch.all(distribution.median <= distribution.q90)
    assert encoded.memory.shape == (1, 184, config.d_model)
    assert encoded.memory_mask.shape == (1, 184)
    assert tuple(model.bar_ages[[0, -1]].tolist()) == (3780.0, 0.0)
    assert tuple(model.patch_ages[[0, -1]].tolist()) == (118.0, 0.0)


def test_export_facing_raw_path_omits_the_auxiliary_head_call() -> None:
    config = ChmtQConfig.tiny(decoder_layers=2)
    model = ChmtQTransformer(config).eval()
    batch = _batch(config)
    calls: list[torch.Tensor] = []
    handle = model.output_head.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output)
    )
    try:
        with torch.no_grad():
            model.forward_raw(batch)
        assert len(calls) == 1
        calls.clear()
        with torch.no_grad():
            model.forward_training(batch)
        assert len(calls) == 2
    finally:
        handle.remove()


def test_missing_l1_l2_are_hard_masked_and_cannot_change_output() -> None:
    torch.manual_seed(31)
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = _batch(config)
    level_mask = torch.zeros_like(batch.l2_level_mask)
    empty = replace(
        batch,
        l1=torch.zeros_like(batch.l1),
        l1_mask=torch.zeros_like(batch.l1_mask),
        l2=torch.zeros_like(batch.l2),
        l2_level_mask=level_mask,
        l2_snapshot_mask=torch.zeros_like(batch.l2_snapshot_mask),
    )
    garbage = replace(
        empty,
        l1=torch.full_like(batch.l1, 1.0e8),
        l2=torch.full_like(batch.l2, -1.0e8),
    )

    with torch.no_grad():
        first = model(empty)
        second = model(garbage)
        encoded = model.encode_modalities(garbage)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.isfinite(first).all()
    assert encoded.gates[:, 1:].count_nonzero() == 0
    assert encoded.memory_mask[:, 64:].count_nonzero() == 0


def test_one_sided_depth_is_valid_and_finite() -> None:
    torch.manual_seed(37)
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = _batch(config)
    one_side = torch.zeros_like(batch.l2_level_mask)
    one_side[:, :, 0, :3] = True
    batch = replace(
        batch,
        l2_level_mask=one_side,
        l2_snapshot_mask=one_side.any(dim=(-1, -2)),
    )

    with torch.no_grad():
        result = model(batch)

    assert result.shape == (1, 5, 17)
    assert torch.isfinite(result).all()


def test_patch_pool_never_reads_a_later_patch_and_uses_static_offsets() -> None:
    torch.manual_seed(41)
    pool = MaskedPatchPool(8, hidden_dim=4, patch_size=4).eval()
    frames = torch.randn(1, 8, 8, requires_grad=True)
    valid = torch.ones(1, 8, dtype=torch.bool)

    pooled, patch_mask = pool(frames, valid)
    pooled[:, 0].sum().backward()

    assert pooled.shape == (1, 2, 8)
    assert patch_mask.tolist() == [[True, True]]
    assert pool.within_patch_embedding.shape == (4, 8)
    assert frames.grad is not None and frames.grad[:, :4].abs().sum() > 0
    assert frames.grad[:, 4:].count_nonzero() == 0


def test_model_collects_and_resets_named_attention_logit_telemetry() -> None:
    config = ChmtQConfig.tiny(decoder_layers=2)
    model = ChmtQTransformer(config).eval()
    batch = _batch(config)
    expected = {
        name: module.num_heads
        for name, module in model.named_modules()
        if isinstance(module, ChmtQAttention)
    }

    assert model.collect_attention_logit_telemetry() == {}
    model.enable_attention_logit_telemetry(runaway_threshold=80.0)
    with torch.no_grad():
        model(batch)
    collected = model.collect_attention_logit_telemetry(reset=True)

    assert set(collected) == set(expected)
    for name, telemetry in collected.items():
        assert telemetry.observations == 1
        assert len(telemetry.max_abs_by_head) == expected[name]
        assert max(telemetry.max_abs_by_head) < telemetry.runaway_threshold
    assert model.collect_attention_logit_telemetry() == {}

    model.disable_attention_logit_telemetry(reset=True)
    with torch.no_grad():
        model(batch)
    assert model.collect_attention_logit_telemetry() == {}


def test_decoder_horizon_attention_is_noncausal_and_masks_memory_garbage() -> None:
    torch.manual_seed(47)
    block = ChmtQDecoderBlock(
        8,
        2,
        16,
        0.0,
        rms_epsilon=1.0e-6,
        qk_epsilon=1.0e-6,
        layer_scale_initial=0.1,
    ).eval()
    queries = torch.randn(1, 5, 8, requires_grad=True)
    memory = torch.randn(1, 4, 8)
    memory_mask = torch.tensor([[True, False, True, False]])
    garbage = memory.clone()
    garbage[:, 1::2] = 1.0e8

    first = block(queries, memory, memory_mask)
    second = block(queries.detach(), garbage, memory_mask)
    first[:, 0].sum().backward()

    torch.testing.assert_close(first.detach(), second, rtol=0, atol=0)
    assert queries.grad is not None
    assert queries.grad[:, 1:].abs().sum() > 0


def test_partially_masked_l2_levels_cannot_affect_encoded_result() -> None:
    torch.manual_seed(53)
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = _batch(config)
    garbage_l2 = batch.l2.clone()
    invalid = ~batch.l2_level_mask.unsqueeze(-1).expand_as(garbage_l2)
    garbage_l2[invalid] = 1.0e8
    changed = replace(batch, l2=garbage_l2)

    with torch.no_grad():
        expected = model(batch)
        actual = model(changed)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_width_fp32_forward_backward_is_finite_and_complete() -> None:
    torch.manual_seed(59)
    config = ChmtQConfig.reference()
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=61)

    training = model.forward_training(batch)
    objective = training.raw.final.square().mean() + training.raw.auxiliary.square().mean()
    objective.backward()

    assert training.raw.final.dtype == torch.float32
    assert training.raw.final.shape == (1, 5, 17)
    assert torch.isfinite(training.raw.final).all()
    assert torch.isfinite(objective)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
