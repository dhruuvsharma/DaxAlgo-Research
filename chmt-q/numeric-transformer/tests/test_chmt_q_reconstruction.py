from __future__ import annotations

from dataclasses import replace

import torch

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.output import ChmtQOutput
from daxalgo_numeric_transformer.chmt_q.reconstruction import reconstruct_forecast
from daxalgo_numeric_transformer.chmt_q.scaling import fit_robust_scaler
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch


def _fixture():
    config = ChmtQConfig.tiny()
    batch = make_chmt_q_synthetic_batch(config, batch_size=4, seed=227)
    scaler = fit_robust_scaler([batch], source_batch_set_hash="c" * 64, config=config)
    quantiles = torch.zeros(1, config.horizons, config.targets, config.quantiles)
    quantiles[..., 0] = -0.25
    quantiles[..., 2] = 0.25
    output = ChmtQOutput(
        quantiles,
        torch.tensor([[[2.0, -2.0]] * config.horizons]),
    ).validate(config)
    return config, scaler, output


def test_reconstruction_produces_ordered_physical_values_and_presence() -> None:
    config, scaler, output = _fixture()
    result = reconstruct_forecast(
        output,
        scaler,
        anchor_price=50_000.0,
        log_volume_reference=1.0,
        tick_size=0.5,
        config=config,
    )
    assert result.unavailable_reason is None
    assert result.forecast is not None
    forecast = result.forecast
    assert forecast.primitive_quantiles.shape == (1, 5, 5, 3)
    assert bool(
        (forecast.primitive_quantiles[..., 0] <= forecast.primitive_quantiles[..., 1]).all()
    )
    assert bool(
        (forecast.primitive_quantiles[..., 1] <= forecast.primitive_quantiles[..., 2]).all()
    )
    assert forecast.presence_decisions[0, 0].tolist() == [True, False]
    assert torch.isfinite(forecast.raw_delta_median).all()


def test_reconstruction_rejects_the_entire_forecast_on_invalid_bounds() -> None:
    config, scaler, output = _fixture()
    broken_scales = list(scaler.targets.scales)
    broken_scales[0] = 1.0e9
    broken_targets = replace(scaler.targets, scales=tuple(broken_scales))
    broken_scaler = replace(scaler, targets=broken_targets)
    result = reconstruct_forecast(
        output,
        broken_scaler,
        anchor_price=50_000.0,
        log_volume_reference=1.0,
        tick_size=0.5,
        maximum_absolute_offset_ticks=10.0,
        config=config,
    )
    assert result.forecast is None
    assert result.unavailable_reason is not None
