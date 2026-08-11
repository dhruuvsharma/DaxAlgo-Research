from __future__ import annotations

import numpy as np

from daxalgo_footprint_transformer.config import EncodingConfig
from daxalgo_footprint_transformer.data import EncodedFootprintSeries, build_footprint_bars
from daxalgo_footprint_transformer.screening import (
    _futility_stop_length,
    execution_coverage_mask,
    screen_sample_gate,
    selection_score,
)


def _protocol() -> dict[str, object]:
    return {
        "development": {
            "split": {
                "train_fraction": 0.60,
                "validation_fraction": 0.15,
                "reserved_screen_fraction": 0.25,
            }
        },
        "sample_gate": {
            "minimum_train_blocks": 2,
            "minimum_validation_blocks": 2,
            "minimum_reserved_screen_blocks": 2,
            "minimum_continuous_window_fraction": 0.99,
        },
    }


def test_sample_gate_uses_physical_dependence_blocks() -> None:
    config = EncodingConfig(
        lookback_bars=8,
        horizon_bars=2,
        max_rows=8,
        row_size=1.0,
        interval_seconds=300,
    )
    trades = [
        (bar * 300_000 + 1_000, 100.0 + bar % 2, 1.0, False) for bar in range(1_000)
    ]
    series = EncodedFootprintSeries.from_bars(build_footprint_bars(trades, config), config)

    gate = screen_sample_gate(series, _protocol())

    assert gate["passed"] is True
    assert gate["dependence_bars"] == 48
    assert gate["continuous_window_fraction"] == 1.0


def test_selection_score_uses_only_final_horizon_poc_and_delta() -> None:
    result = {
        "encoding_config": {"horizon_bars": 2},
        "validation": {
            "by_horizon_target": {
                "2": {
                    "poc_offset": {"median_mae_normalized": 0.20},
                    "delta_fraction": {"median_mae_normalized": 0.30},
                }
            }
        },
        "baselines": {
            "direct_ridge": {
                "validation": {
                    "by_horizon_target": {
                        "2": {
                            "poc_offset": {"median_mae_normalized": 0.25},
                            "delta_fraction": {"median_mae_normalized": 0.50},
                        }
                    }
                }
            }
        },
    }

    assert selection_score(result) == 0.3


def test_execution_coverage_excludes_a_delayed_exit_at_source_end() -> None:
    mask = execution_coverage_mask(
        np.asarray((0, 60_000)),
        interval_ms=60_000,
        horizon_bars=1,
        execution_latency_ms=1_000,
        source_end_exclusive_ms=121_000,
    )

    np.testing.assert_array_equal(mask, (True, False))


def test_futility_gate_waits_for_ten_complete_days_and_minimum_evidence() -> None:
    decisions_per_day = 48
    decisions = 12 * decisions_per_day
    interval_ms = 300_000
    horizon = 6
    eligible = np.arange(decisions, dtype=np.int64) * interval_ms * horizon + 1_000
    stop, event = _futility_stop_length(
        eligible,
        np.full(decisions, 100.0),
        np.full(decisions, 99.0),
        np.ones(decisions, dtype=np.int8),
        interval_ms=interval_ms,
        horizon_bars=horizon,
        side_cost_bps=0.0,
        minimum_days=10,
        minimum_decisions=200,
        minimum_trades=30,
        net_return_floor=-0.10,
    )

    assert stop == 10 * decisions_per_day
    assert event is not None
    assert event["complete_utc_days"] == 10
    assert event["trades"] == stop
