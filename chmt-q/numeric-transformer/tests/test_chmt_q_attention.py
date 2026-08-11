from __future__ import annotations

import math

import pytest
import torch

from daxalgo_numeric_transformer.chmt_q.model import (
    ChmtQAttention,
    ChmtQEncoderBlock,
    RelativeTimeBias,
    safe_masked_softmax,
)


def test_safe_masked_softmax_uses_only_eligible_scores_and_zeroes_empty_rows() -> None:
    scores = torch.tensor([[0.25, float("nan"), -0.75, 1.0e20], [3.0, -2.0, 4.0, 1.0]])
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])

    actual = safe_masked_softmax(scores, mask)

    torch.testing.assert_close(actual[0, [0, 2]], torch.softmax(torch.tensor([0.25, -0.75]), dim=0))
    assert actual[0, [1, 3]].count_nonzero() == 0
    assert actual[1].count_nonzero() == 0
    assert torch.isfinite(actual).all()


def test_qk_normalized_attention_matches_explicit_reference() -> None:
    torch.manual_seed(7)
    attention = ChmtQAttention(4, 2, qk_epsilon=1.0e-6).eval()
    query = torch.randn(1, 2, 4)
    memory = torch.randn(1, 3, 4)
    key_mask = torch.tensor([[True, False, True]])
    with torch.no_grad():
        attention.log_temperature.copy_(torch.tensor([math.log(0.5), math.log(2.0)]))

    _, weights = attention(query, memory, key_mask=key_mask, need_weights=True)

    q = attention.q_projection(query).view(1, 2, 2, 2).transpose(1, 2)
    safe_memory = torch.where(key_mask.unsqueeze(-1), memory, torch.zeros_like(memory))
    k = attention.k_projection(safe_memory).view(1, 3, 2, 2).transpose(1, 2)
    q = q / torch.sqrt(q.square().mean(dim=-1, keepdim=True) + 1.0e-6)
    k = k / torch.sqrt(k.square().mean(dim=-1, keepdim=True) + 1.0e-6)
    temperature = torch.tensor([0.5, 2.0])[None, :, None, None]
    scores = (q @ k.transpose(-2, -1)) * temperature / math.sqrt(2.0)
    expected = safe_masked_softmax(scores, key_mask[:, None, None, :])

    torch.testing.assert_close(weights, expected)
    assert weights[..., 1].count_nonzero() == 0


def test_invalid_keys_queries_and_all_masked_rows_are_exact_zero() -> None:
    torch.manual_seed(11)
    attention = ChmtQAttention(8, 2).eval()
    values = torch.randn(2, 3, 8)
    key_mask = torch.tensor([[True, False, True], [False, False, False]])
    query_mask = torch.tensor([[True, False, True], [True, True, True]])
    changed = values.clone()
    changed[0, 1] = 1.0e8

    first, first_weights = attention(
        values,
        key_mask=key_mask,
        query_mask=query_mask,
        need_weights=True,
    )
    second = attention(changed, key_mask=key_mask, query_mask=query_mask)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first[0, 1].count_nonzero() == 0
    assert first_weights[0, :, 1].count_nonzero() == 0
    assert first[1].count_nonzero() == 0
    assert first_weights[1].count_nonzero() == 0


def test_temporal_block_has_no_gradient_to_a_future_token() -> None:
    torch.manual_seed(13)
    block = ChmtQEncoderBlock(
        8,
        2,
        16,
        0.0,
        rms_epsilon=1.0e-6,
        qk_epsilon=1.0e-6,
        layer_scale_initial=0.1,
        causal=True,
        relative_time_scale=2.0,
    ).eval()
    values = torch.randn(1, 6, 8, requires_grad=True)
    valid = torch.ones(1, 6, dtype=torch.bool)
    ages = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0, 0.0])

    encoded = block(values, valid, ages=ages)
    encoded[:, 2].sum().backward()

    assert values.grad is not None
    assert values.grad[:, :3].abs().sum() > 0
    assert values.grad[:, 3:].count_nonzero() == 0


def test_temperature_and_elapsed_decay_are_independent_learned_parameters() -> None:
    attention = ChmtQAttention(8, 2)
    time_bias = RelativeTimeBias(2, 2.0)

    assert attention.log_temperature.shape == (2,)
    assert time_bias.gamma.shape == (2,)
    assert attention.log_temperature.requires_grad
    assert time_bias.gamma.requires_grad
    assert attention.log_temperature.data_ptr() != time_bias.gamma.data_ptr()


def test_opt_in_logit_telemetry_records_heads_and_fails_fast() -> None:
    attention = ChmtQAttention(4, 2).eval()
    values = torch.tensor([[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]])

    attention(values)
    assert attention.collect_logit_telemetry() is None

    attention.enable_logit_telemetry(runaway_threshold=100.0)
    attention(values)
    telemetry = attention.collect_logit_telemetry()
    assert telemetry is not None
    assert telemetry.observations == 1
    assert telemetry.runaway_threshold == 100.0
    assert len(telemetry.max_abs_by_head) == 2
    assert all(math.isfinite(value) and value >= 0.0 for value in telemetry.max_abs_by_head)

    attention.enable_logit_telemetry(runaway_threshold=1.0e-6)
    with torch.no_grad():
        attention.log_temperature.fill_(4.0)
    with pytest.raises(FloatingPointError, match="exceeded"):
        attention(values)

    attention.enable_logit_telemetry(runaway_threshold=100.0)
    additive_bias = torch.zeros(1, 2, 2, 2)
    additive_bias[0, 0, 0, 0] = torch.nan
    with pytest.raises(FloatingPointError, match="non-finite eligible"):
        attention(values, additive_bias=additive_bias)
