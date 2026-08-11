from __future__ import annotations

import torch

from daxalgo_numeric_transformer.masks import masked_softmax
from daxalgo_numeric_transformer.model.attention import PrimitiveMultiheadAttention


def test_masked_softmax_all_valid_matches_torch() -> None:
    logits = torch.tensor([[[0.5, -0.25, 1.25], [2.0, 0.0, -1.0]]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    actual = masked_softmax(logits, valid, dim=-1)

    torch.testing.assert_close(actual, torch.softmax(logits, dim=-1))


def test_masked_softmax_partial_mask_normalizes_only_valid_keys() -> None:
    logits = torch.tensor([[0.0, 1.0, 2000.0, -1.0]])
    valid = torch.tensor([[True, True, False, False]])

    actual = masked_softmax(logits, valid, dim=-1)

    assert actual[0, 2:].count_nonzero() == 0
    torch.testing.assert_close(actual.sum(dim=-1), torch.ones(1))
    torch.testing.assert_close(actual[0, :2], torch.softmax(logits[0, :2], dim=-1))


def test_masked_softmax_all_masked_is_exact_zero_and_finite() -> None:
    logits = torch.tensor([[float("nan"), 1000.0, -1000.0]])
    valid = torch.zeros_like(logits, dtype=torch.bool)

    actual = masked_softmax(logits, valid, dim=-1)

    assert torch.isfinite(actual).all()
    assert actual.count_nonzero() == 0


def test_invalid_key_logits_cannot_change_valid_distribution() -> None:
    valid = torch.tensor([[True, False, True, False]])
    first = torch.tensor([[0.25, -1.0e9, -0.75, 1.0e9]])
    second = torch.tensor([[0.25, 12345.0, -0.75, -54321.0]])

    first_result = masked_softmax(first, valid, dim=-1)
    second_result = masked_softmax(second, valid, dim=-1)

    torch.testing.assert_close(first_result, second_result, rtol=0, atol=0)


def test_attention_invalid_key_values_cannot_change_output() -> None:
    torch.manual_seed(7)
    attention = PrimitiveMultiheadAttention(d_model=8, num_heads=2).eval()
    query = torch.randn(1, 2, 8)
    memory = torch.randn(1, 3, 8)
    key_mask = torch.tensor([[True, False, True]])
    changed_memory = memory.clone()
    changed_memory[:, 1] = 1.0e6

    first = attention(query, memory, key_mask=key_mask)
    second = attention(query, changed_memory, key_mask=key_mask)

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_attention_invalid_queries_and_all_masked_rows_are_exact_zero() -> None:
    torch.manual_seed(11)
    attention = PrimitiveMultiheadAttention(d_model=8, num_heads=2).eval()
    tokens = torch.randn(2, 3, 8)
    key_mask = torch.tensor([[True, True, True], [False, False, False]])
    query_mask = torch.tensor([[True, False, True], [True, True, True]])

    output, weights = attention(
        tokens,
        key_mask=key_mask,
        query_mask=query_mask,
        need_weights=True,
    )

    assert torch.isfinite(output).all()
    assert torch.isfinite(weights).all()
    assert output[0, 1].count_nonzero() == 0
    assert weights[0, :, 1].count_nonzero() == 0
    assert output[1].count_nonzero() == 0
    assert weights[1].count_nonzero() == 0

