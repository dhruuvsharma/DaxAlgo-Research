from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from daxalgo_numeric_transformer.config import ModelConfig
from daxalgo_numeric_transformer.model.numeric_transformer import NumericMarketTransformer
from daxalgo_numeric_transformer.synthetic import make_synthetic_batch

MODEL_INPUT_NAMES = (
    "bars",
    "bars_mask",
    "l1",
    "l1_mask",
    "l2",
    "l2_level_mask",
    "l2_snapshot_mask",
    "quality",
)


def test_synthetic_batch_matches_exact_tensor_contract(
    tiny_config: ModelConfig, market_batch
) -> None:
    batch = market_batch
    batch.validate(tiny_config)

    assert batch.bars.shape == (2, 64, 16)
    assert batch.bars_mask.shape == (2, 64)
    assert batch.l1.shape == (2, 240, 16)
    assert batch.l1_mask.shape == (2, 240)
    assert batch.l2.shape == (2, 240, 2, 10, 12)
    assert batch.l2_level_mask.shape == (2, 240, 2, 10)
    assert batch.l2_snapshot_mask.shape == (2, 240)
    assert batch.quality.shape == (2, 16)
    assert batch.targets.shape == (2, 5, 5)
    assert batch.target_mask.shape == (2, 5, 5)

    for name in ("bars", "l1", "l2", "quality", "targets"):
        value = getattr(batch, name)
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
    for name in ("bars_mask", "l1_mask", "l2_level_mask", "l2_snapshot_mask", "target_mask"):
        assert getattr(batch, name).dtype == torch.bool


def test_batch_validator_rejects_wrong_shape(tiny_config: ModelConfig, market_batch) -> None:
    with pytest.raises((TypeError, ValueError), match="bars"):
        invalid = replace(market_batch, bars=market_batch.bars[:, :-1])
        invalid.validate(tiny_config)


def test_forward_accepts_batch_and_named_contract(tiny_config: ModelConfig) -> None:
    torch.manual_seed(3)
    model = NumericMarketTransformer(tiny_config).eval()
    batch = make_synthetic_batch(tiny_config, batch_size=1, seed=5, device="cpu")

    with torch.no_grad():
        from_batch = model(batch)
        from_kwargs = model(**batch.as_model_kwargs())

    assert from_batch.shape == (1, 5, 5)
    assert from_batch.dtype == torch.float32
    assert torch.isfinite(from_batch).all()
    assert torch.equal(from_batch, from_kwargs)
    assert torch.all(from_batch[..., 4] >= -1.0)
    assert torch.all(from_batch[..., 4] <= 1.0)
    assert tuple(batch.as_model_kwargs()) == MODEL_INPUT_NAMES

