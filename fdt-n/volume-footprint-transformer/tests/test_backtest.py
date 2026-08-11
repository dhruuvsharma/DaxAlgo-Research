from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from daxalgo_footprint_transformer.backtest import (
    FrozenPolicy,
    _write_investor_report,
    first_retained_trade_fills,
    policy_signals,
    select_decision_indices,
    simulate_blocks,
)
from daxalgo_footprint_transformer.config import TARGET_NAMES, EncodingConfig
from daxalgo_footprint_transformer.data import EncodedFootprintSeries, build_footprint_bars
from daxalgo_footprint_transformer.training import DirectRidgeBaseline


def _series(count: int = 120) -> EncodedFootprintSeries:
    config = EncodingConfig(lookback_bars=16, horizon_bars=8, max_rows=8, row_size=1.0)
    trades = []
    for minute in range(count):
        timestamp = minute * 60_000
        price = 100.0 + 0.1 * minute
        trades.extend(
            (
                (timestamp + 1_000, price - 1.0, 5.0, True),
                (timestamp + 2_000, price, 10.0, False),
                (timestamp + 3_000, price + 1.0, 5.0, False),
            )
        )
    return EncodedFootprintSeries.from_bars(build_footprint_bars(trades, config), config)


def test_decision_indices_are_causal_complete_and_utc_aligned() -> None:
    series = _series()
    policy = FrozenPolicy(horizon_bars=8, decision_interval_bars=8)
    selected = select_decision_indices(series, 0, 120 * 60_000, policy)
    decision_times = series.timestamps_ms[selected] + 60_000
    entry_times = decision_times + policy.execution_latency_ms
    exit_times = decision_times + 8 * 60_000 + policy.execution_latency_ms
    assert np.all((decision_times // 60_000) % 8 == 0)
    assert np.all(entry_times > decision_times)
    assert np.all(exit_times <= 120 * 60_000 + policy.execution_latency_ms)
    assert np.all(selected + policy.horizon_bars < len(series))


def test_frozen_policy_rejects_overlap_and_nonsensical_costs() -> None:
    with pytest.raises(ValueError, match="non-overlapping"):
        FrozenPolicy(horizon_bars=8, decision_interval_bars=4).validate()
    with pytest.raises(ValueError, match="10,000"):
        FrozenPolicy(primary_side_cost_bps=10_000).validate()


def test_policy_uses_only_frozen_horizon_median_poc_and_delta() -> None:
    series = _series()
    indices = select_decision_indices(series, 0, 120 * 60_000, FrozenPolicy())[:2]
    prediction = np.zeros((2, 8, len(TARGET_NAMES), 3), dtype=np.float64)
    poc_index = TARGET_NAMES.index("poc_offset")
    delta_index = TARGET_NAMES.index("delta_fraction")
    anchor_ticks = series.bar_values[indices, 0]
    prediction[:, 7, poc_index, 1] = np.asarray((30.0, 20.0)) / 10_000 * anchor_ticks / 64
    prediction[:, 7, delta_index, 1] = 0.1

    positions, edge_bps, delta = policy_signals(prediction, series, indices, FrozenPolicy())

    np.testing.assert_allclose(edge_bps, (30.0, 20.0))
    np.testing.assert_array_equal(positions, (1, 0))
    np.testing.assert_allclose(delta, (0.1, 0.1))


def test_execution_prices_use_first_retained_trade_after_latency(tmp_path: Path) -> None:
    source = tmp_path / "BTCUSDT-trades-2024-01-01.csv"
    source.write_text(
        "1,100.0,0.001,0.1,1000,false,true\n"
        "2,101.0,0.001,0.1,2000,true,true\n"
        "3,102.0,0.001,0.1,61000,false,true\n",
        encoding="utf-8",
    )
    prices, fill_times = first_retained_trade_fills(
        [source], np.asarray((1_500, 60_000)), quantity_scale=1_000.0
    )
    np.testing.assert_allclose(prices, (101.0, 102.0))
    np.testing.assert_array_equal(fill_times, (2_000, 61_000))


def test_block_simulation_charges_both_sides_and_keeps_flat_blocks() -> None:
    metrics, trades, curve, daily = simulate_blocks(
        np.asarray((0, 480_000)),
        np.asarray((480_000, 960_000)),
        np.asarray((100.0, 100.0)),
        np.asarray((101.0, 99.0)),
        np.asarray((1, 0)),
        initial_capital_usd=10_000.0,
        side_cost_bps=10.0,
    )
    expected_factor = 0.999**2 * 1.01
    assert metrics["trades"] == 1
    assert metrics["blocks"] == 2
    assert metrics["exposure_fraction"] == 0.5
    assert metrics["net_return"] == expected_factor - 1.0
    assert trades[0]["net_pnl_usd"] == 10_000.0 * (expected_factor - 1.0)
    np.testing.assert_allclose(curve, (1.0, expected_factor, expected_factor))
    assert daily.shape == (1,)


def test_daily_returns_are_bucketed_by_entry_day_at_midnight_exit() -> None:
    day = 86_400_000
    _, _, _, daily = simulate_blocks(
        np.asarray((0, day - 480_000, day, 2 * day - 480_000)),
        np.asarray((480_000, day, day + 480_000, 2 * day)),
        np.full(4, 100.0),
        np.full(4, 100.0),
        np.zeros(4),
        initial_capital_usd=10_000.0,
        side_cost_bps=12.0,
    )
    assert daily.shape == (2,)


def test_block_simulation_rejects_overlapping_positions() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        simulate_blocks(
            np.asarray((0, 400_000)),
            np.asarray((480_000, 880_000)),
            np.full(2, 100.0),
            np.full(2, 101.0),
            np.ones(2),
            initial_capital_usd=10_000.0,
            side_cost_bps=12.0,
        )


def test_direct_ridge_round_trip_preserves_schema(tmp_path: Path) -> None:
    baseline = DirectRidgeBaseline(
        feature_mean=np.zeros(3),
        feature_scale=np.ones(3),
        coefficients=np.zeros((8, len(TARGET_NAMES), 4)),
        alpha=1.0,
        training_examples=100,
    )
    path = tmp_path / "ridge.npz"
    baseline.save(path)
    restored = DirectRidgeBaseline.load(path)
    assert restored.alpha == 1.0
    assert restored.training_examples == 100
    np.testing.assert_array_equal(restored.coefficients, baseline.coefficients)


def test_investor_report_separates_forecasts_from_non_deployable_economics(
    tmp_path: Path,
) -> None:
    def metric(gross: float, net: float, trades: int, exposure: float) -> dict[str, object]:
        return {
            "gross_return": gross,
            "net_return": net,
            "net_pnl_usd": 10_000 * net,
            "trades": trades,
            "exposure_fraction": exposure,
            "block_boundary_max_drawdown": min(net, 0.0),
            "turnover_sides": 2 * trades,
            "trade_hit_rate": 0.25,
            "profit_factor": 0.4,
            "daily_sharpe_annualized": -2.0,
            "daily_observations": 30,
            "bootstrap": {"net_return_ci_95": [-0.2, -0.05]},
        }

    result = {
        "financial_metrics": {
            "fdt_n_tiny_ensemble_primary": metric(0.01, -0.1, 10, 0.1),
            "direct_ridge_primary": metric(0.02, -0.2, 20, 0.2),
            "buy_and_hold_primary": metric(0.1, 0.09, 1, 1.0),
            "fdt_n_tiny_ensemble_stress": metric(0.01, -0.15, 10, 0.1),
            "no_trade": {
                "gross_return": 0.0,
                "net_return": 0.0,
                "net_pnl_usd": 0.0,
                "trades": 0,
                "block_boundary_max_drawdown": 0.0,
            },
        },
        "forecast_metrics": {
            "fdt_n_tiny_ensemble": {
                "pooled": {
                    "median_mae_normalized": 0.22,
                    "poc_directional_hit_rate": 0.53,
                    "central_80_coverage": 0.84,
                }
            },
            "direct_ridge": {
                "pooled": {
                    "median_mae_normalized": 0.23,
                    "poc_directional_hit_rate": 0.55,
                }
            },
            "persistence": {
                "pooled": {
                    "median_mae_normalized": 0.24,
                    "poc_directional_hit_rate": 0.05,
                }
            },
        },
        "policy": {
            "initial_capital_usd": 10_000.0,
            "primary_side_cost_bps": 12.0,
            "stress_side_cost_bps": 17.0,
            "horizon_bars": 8,
            "decision_interval_bars": 8,
            "signal_edge_bps": 5.0,
            "execution_latency_ms": 1_000,
        },
        "evaluation": {
            "start_utc": "2024-01-15",
            "end_exclusive_utc": "2024-02-14",
            "decision_examples": 5_400,
            "first_entry": "2024-01-15T00:00:01+00:00",
            "last_exit": "2024-02-14T00:00:01+00:00",
        },
        "model": {"parameter_counts": [59_797] * 3, "seeds": [17, 29, 43]},
        "data_provenance": {
            "source_archives": 32,
            "preregistration_sha256": "a" * 64,
            "dataset_sha256": "b" * 64,
        },
        "development_calibration": {
            "selected_signal_edge_bps": 5.0,
            "selected_trades": 93,
            "selected_gross_return": 0.026,
            "selected_net_return": -0.179,
        },
    }
    path = tmp_path / "investor-report.md"
    _write_investor_report(path, result)
    text = path.read_text(encoding="utf-8")

    assert "No deployment or capital-allocation recommendation" in text
    assert "Cash / no trade" in text
    assert "10.00%" in text
    assert "-17.90%" in text
    assert "rather than a trade win rate" in text
    assert "does not have an external immutable timestamp" in text
