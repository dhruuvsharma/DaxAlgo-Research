from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from daxalgo_footprint_transformer.config import BAR_FEATURE_NAMES, ROW_FEATURE_NAMES, TARGET_NAMES
from daxalgo_footprint_transformer.data import (
    EncodedFootprintSeries,
    EncodingConfig,
    build_footprint_bars,
    iter_binance_trades,
    resample_footprint_bars,
)
from daxalgo_footprint_transformer.training import collate_windows


def _synthetic_bars(count: int = 90) -> EncodedFootprintSeries:
    config = EncodingConfig(lookback_bars=16, horizon_bars=4, max_rows=12, row_size=1.0)
    trades = []
    for minute in range(count):
        anchor = 100.0 + minute * 0.2
        timestamp = minute * 60_000
        trades.extend(
            (
                timestamp + 1_000,
                anchor - 1.0,
                10.0,
                True,
            )
            for _ in range(2)
        )
        trades.extend(
            (
                timestamp + 2_000,
                anchor,
                20.0,
                False,
            )
            for _ in range(3)
        )
        trades.append((timestamp + 3_000, anchor + 1.0, 8.0, False))
    bars = build_footprint_bars(trades, config)
    return EncodedFootprintSeries.from_bars(bars, config)


def test_bar_builder_preserves_aggressor_volume_and_time_order() -> None:
    config = EncodingConfig(lookback_bars=2, horizon_bars=1, max_rows=8, row_size=1.0)
    trades = [
        (1_000, 100.0, 3.0, False),
        (2_000, 100.0, 2.0, True),
        (61_000, 101.0, 4.0, False),
        (62_000, 100.0, 1.0, True),
        (121_000, 102.0, 5.0, False),
    ]
    bars = build_footprint_bars(trades, config)
    assert len(bars) == 3
    assert bars[0].buy_volume == 3.0
    assert bars[0].sell_volume == 2.0
    assert bars[1].cumulative_delta == 4.0
    assert bars[0].start_ms < bars[1].start_ms < bars[2].start_ms


def test_binance_quantity_scaling_matches_live_connector_rounding(tmp_path: Path) -> None:
    source = tmp_path / "BTCUSDT-trades-2024-01-01.csv"
    source.write_text(
        "1,100.0,0.0004,0.04,1000,false,true\n"
        "2,100.0,0.0006,0.06,2000,true,true\n"
        "3,100.0,0.0015,0.15,3000,false,true\n",
        encoding="utf-8",
    )

    trades = list(iter_binance_trades([source], quantity_scale=1_000.0))

    assert [trade[2] for trade in trades] == [1.0, 2.0]
    assert [trade[3] for trade in trades] == [True, False]


def test_resample_merges_complete_aligned_bars_without_changing_volume() -> None:
    config = EncodingConfig(lookback_bars=2, horizon_bars=1, max_rows=8, row_size=1.0)
    trades = [
        (minute * 60_000 + 1_000, 100.0 + minute % 2, 2.0, minute % 3 == 0)
        for minute in range(10)
    ]
    bars = build_footprint_bars(trades, config)
    resampled = resample_footprint_bars(bars, 300)

    assert [bar.start_ms for bar in resampled] == [0, 300_000]
    assert [bar.end_ms for bar in resampled] == [300_000, 600_000]
    assert sum(bar.total_volume for bar in resampled) == sum(bar.total_volume for bar in bars)
    assert resampled[0].cumulative_delta == bars[4].cumulative_delta
    assert resampled[1].cumulative_delta == bars[9].cumulative_delta


def test_resample_drops_a_partial_group_instead_of_hiding_a_gap() -> None:
    config = EncodingConfig(lookback_bars=2, horizon_bars=1, max_rows=8, row_size=1.0)
    trades = [
        (minute * 60_000 + 1_000, 100.0, 1.0, False)
        for minute in range(10)
        if minute != 2
    ]
    bars = build_footprint_bars(trades, config)
    resampled = resample_footprint_bars(bars, 300)

    assert [bar.start_ms for bar in resampled] == [300_000]


def test_encoder_retains_every_observed_price_row() -> None:
    series = _synthetic_bars()
    assert series.row_features.shape[0] == int(series.row_offsets[-1])
    assert np.all(np.diff(series.row_offsets) == 3)


def test_encoded_window_has_declared_shapes_and_no_target_in_inputs(tmp_path: Path) -> None:
    series = _synthetic_bars()
    index = int(series.valid_window_indices()[5])
    row, row_mask, bar, targets, target_mask = series.make_window(index)
    assert row.shape[0] == series.config.lookback_bars
    assert row.shape[1] <= series.config.max_rows
    assert row.shape[2] == len(ROW_FEATURE_NAMES)
    assert series.row_features.shape[0] == int(series.row_offsets[-1])
    assert row_mask.shape == row.shape[:2]
    assert bar.shape == (series.config.lookback_bars, len(BAR_FEATURE_NAMES))
    assert targets.shape == (series.config.horizon_bars, len(TARGET_NAMES))
    assert target_mask.shape == targets.shape
    assert np.isfinite(row).all()
    assert np.isfinite(bar).all()
    assert np.isfinite(targets).all()

    before = (row.copy(), row_mask.copy(), bar.copy())
    series.bar_values[index + 1 :, :] += 1000.0
    after = series.make_window(index)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])
    np.testing.assert_array_equal(before[2], after[2])

    path = tmp_path / "series.npz"
    series.save(path)
    restored = EncodedFootprintSeries.load(path)
    assert restored.config == series.config
    np.testing.assert_array_equal(restored.row_features, series.row_features)


def test_windows_crossing_a_missing_interval_are_excluded() -> None:
    series = _synthetic_bars(count=50)
    series.timestamps_ms[25:] += series.config.interval_seconds * 1000
    valid = series.valid_window_indices()
    assert valid.size > 0
    assert not any(
        index - series.config.lookback_bars + 1 <= 24 and index + series.config.horizon_bars >= 25
        for index in valid
    )


def test_encoder_rejects_a_bar_above_the_explicit_safety_bound() -> None:
    config = EncodingConfig(lookback_bars=2, horizon_bars=1, max_rows=4, row_size=1.0)
    trades = [
        (minute * 60_000 + offset, 100.0 + level, 1.0, level % 2 == 0)
        for minute in range(3)
        for offset, level in enumerate(range(-2, 3), start=1)
    ]
    bars = build_footprint_bars(trades, config)

    with pytest.raises(ValueError, match="explicit safety bound"):
        EncodedFootprintSeries.from_bars(bars, config)


def test_batch_collation_pads_without_dropping_rows() -> None:
    def item(width: int):
        return (
            torch.ones(4, width, len(ROW_FEATURE_NAMES)),
            torch.ones(4, width, dtype=torch.bool),
            torch.zeros(4, len(BAR_FEATURE_NAMES)),
            torch.zeros(2, len(TARGET_NAMES)),
            torch.ones(2, len(TARGET_NAMES), dtype=torch.bool),
        )

    rows, mask, _, _, _ = collate_windows([item(3), item(7)])
    assert rows.shape == (2, 4, 7, len(ROW_FEATURE_NAMES))
    assert mask[0].sum().item() == 12
    assert mask[1].sum().item() == 28
