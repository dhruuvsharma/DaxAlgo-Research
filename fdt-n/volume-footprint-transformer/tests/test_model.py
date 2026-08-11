from __future__ import annotations

import torch

from daxalgo_footprint_transformer.config import (
    BAR_FEATURE_NAMES,
    ROW_FEATURE_NAMES,
    TARGET_NAMES,
    EncodingConfig,
    ModelConfig,
)
from daxalgo_footprint_transformer.loss import quantile_loss
from daxalgo_footprint_transformer.model import FootprintDistributionTransformer


def _inputs(config: EncodingConfig, batch: int = 2, rows: int | None = None):
    width = rows or config.max_rows
    row = torch.randn(batch, config.lookback_bars, width, len(ROW_FEATURE_NAMES))
    row_mask = torch.ones(batch, config.lookback_bars, width, dtype=torch.bool)
    row_mask[:, :, -3:] = False
    bar = torch.randn(batch, config.lookback_bars, len(BAR_FEATURE_NAMES))
    bar_mask = torch.ones(batch, config.lookback_bars, dtype=torch.bool)
    return row, row_mask, bar, bar_mask


def test_model_emits_ordered_direct_horizon_quantiles() -> None:
    encoding = EncodingConfig(lookback_bars=8, horizon_bars=3, max_rows=10)
    model = FootprintDistributionTransformer(
        encoding,
        ModelConfig(d_model=32, row_queries=3, attention_heads=4, temporal_layers=2),
    )
    model.eval()
    with torch.inference_mode():
        output = model(*_inputs(encoding))
    assert output.shape == (2, 3, len(TARGET_NAMES), 3)
    assert torch.all(output[..., 0] <= output[..., 1])
    assert torch.all(output[..., 1] <= output[..., 2])
    assert torch.all(output[..., 1, :] <= output[..., 0, :])
    assert torch.all(output[..., 2, :] >= output[..., 0, :])
    assert torch.all(output[..., 3, :] >= output[..., 1, :])
    assert torch.all(output[..., 3, :] <= output[..., 2, :])
    assert torch.all(output[..., 4, :] >= output[..., 1, :])
    assert torch.all(output[..., 4, :] <= output[..., 2, :])
    assert torch.all(output[..., 6, :] >= -1)
    assert torch.all(output[..., 6, :] <= 1)
    assert model.parameter_count < 2_000_000


def test_tiny_preset_is_fixed_and_materially_smaller() -> None:
    config = ModelConfig.from_preset("fdt-n-tiny")
    assert config.to_dict() == {
        "d_model": 32,
        "row_queries": 2,
        "attention_heads": 4,
        "temporal_layers": 2,
        "decoder_layers": 1,
        "feedforward_multiplier": 2,
        "dropout": 0.10,
    }
    model = FootprintDistributionTransformer(EncodingConfig(), config)
    assert model.parameter_count == 59_797
    assert model.parameter_count < FootprintDistributionTransformer(
        EncodingConfig(), ModelConfig.from_preset("fdt-n")
    ).parameter_count / 5


def test_model_accepts_dynamic_exact_row_widths() -> None:
    encoding = EncodingConfig(lookback_bars=8, horizon_bars=3, max_rows=128)
    model = FootprintDistributionTransformer(
        encoding,
        ModelConfig(d_model=32, attention_heads=4, temporal_layers=1, decoder_layers=1),
    ).eval()
    with torch.inference_mode():
        narrow = model(*_inputs(encoding, batch=1, rows=5))
        wide = model(*_inputs(encoding, batch=1, rows=37))
    assert narrow.shape == wide.shape == (1, 3, len(TARGET_NAMES), 3)


def test_loss_backpropagates_finite_gradients() -> None:
    encoding = EncodingConfig(lookback_bars=8, horizon_bars=2, max_rows=8)
    model = FootprintDistributionTransformer(
        encoding,
        ModelConfig(d_model=32, attention_heads=4, temporal_layers=1, decoder_layers=1),
    )
    prediction = model(*_inputs(encoding, batch=1))
    target = torch.randn(1, encoding.horizon_bars, len(TARGET_NAMES))
    low = torch.minimum(target[..., 1], target[..., 0])
    high = torch.maximum(target[..., 2], target[..., 0])
    target[..., 1] = low
    target[..., 2] = high
    mask = torch.ones_like(target, dtype=torch.bool)
    loss = quantile_loss(prediction, target, mask)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
