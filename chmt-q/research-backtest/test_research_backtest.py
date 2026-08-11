from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from math import isinf
from statistics import fmean

import pytest

from research_backtest import (
    CostScenario,
    TradeObservation,
    buy_and_hold_benchmark,
    cash_benchmark,
    correlation_metrics,
    day_block_bootstrap_ci,
    sign_classification_metrics,
    simulate,
)


def _observation(
    minute: int,
    direction: int,
    entry: float,
    exit_price: float,
    *,
    day_offset: int = 0,
) -> TradeObservation:
    timestamp = datetime(2026, 6, 19 + day_offset, tzinfo=timezone.utc) + timedelta(
        minutes=minute
    )
    return TradeObservation(
        timestamp=timestamp,
        exit_timestamp=timestamp + timedelta(minutes=5),
        day=timestamp.date(),
        direction=direction,
        entry_price=entry,
        exit_price=exit_price,
    )


def test_simulate_fixed_notional_long_short_and_long_only() -> None:
    observations = (
        _observation(0, 1, 100.0, 110.0),
        _observation(5, -1, 100.0, 90.0),
        _observation(10, 0, 100.0, 500.0),
        _observation(0, -1, 100.0, 105.0, day_offset=1),
    )
    costs = CostScenario("ten bps each way", one_way_bps=10.0)

    result = simulate(observations, costs, mode="long_short", initial_capital=100_000)

    assert [trade.net_return for trade in result.trades] == pytest.approx(
        [0.098, 0.098, -0.052]
    )
    assert result.metrics.gross_total_return == pytest.approx(0.15)
    assert result.metrics.net_total_return == pytest.approx(0.144)
    assert result.metrics.gross_compounded_return == pytest.approx(1.1 * 1.1 * 0.95 - 1)
    assert result.metrics.net_compounded_return == pytest.approx(1.098 * 1.098 * 0.948 - 1)
    assert result.metrics.pnl_per_initial_capital == pytest.approx(14_400)
    assert result.metrics.trade_count == 3
    assert result.metrics.wins == 2
    assert result.metrics.losses == 1
    assert result.metrics.breakeven == 0
    assert result.metrics.hit_rate == pytest.approx(2 / 3)
    assert result.metrics.average_trade_return == pytest.approx(0.048)
    assert result.metrics.median_trade_return == pytest.approx(0.098)
    assert result.metrics.profit_factor == pytest.approx(0.196 / 0.052)
    assert result.metrics.turnover == 6.0
    assert result.metrics.exposure_minutes == 15.0
    assert result.metrics.max_drawdown == pytest.approx(0.052 / 1.196)
    assert result.metrics.annualized_daily_sharpe is not None
    assert result.metrics.annualized_daily_sortino is not None
    assert result.metrics.annualized_calmar is not None
    assert result.metrics.var_95 == pytest.approx(-0.0396)
    assert result.metrics.cvar_95 == pytest.approx(-0.052)
    assert [item.gross_return for item in result.daily_returns] == pytest.approx([0.2, -0.05])
    assert [item.net_return for item in result.daily_returns] == pytest.approx([0.196, -0.052])

    long_only = simulate(observations, costs, mode="long_only")
    assert long_only.metrics.trade_count == 1
    assert long_only.trades[0].observation.direction == 1
    assert isinf(long_only.metrics.profit_factor)
    assert long_only.metrics.annualized_daily_sharpe is not None


def test_empty_or_flat_results_have_defined_zero_returns_and_undefined_ratios() -> None:
    empty = simulate((), CostScenario("none", 0.0))
    assert empty.metrics.trade_count == 0
    assert empty.metrics.net_total_return == 0.0
    assert empty.metrics.net_compounded_return == 0.0
    assert empty.metrics.hit_rate is None
    assert empty.metrics.profit_factor is None
    assert empty.metrics.annualized_daily_sharpe is None
    assert empty.metrics.var_95 is None

    flat = simulate(
        (_observation(0, 0, 100.0, 110.0),),
        CostScenario("none", 0.0),
    )
    assert flat.metrics.trade_count == 0
    assert flat.daily_returns[0].net_return == 0.0
    assert flat.metrics.var_95 == 0.0
    assert flat.metrics.cvar_95 == 0.0


def test_observations_and_costs_reject_invalid_inputs_and_sequences() -> None:
    valid = _observation(0, 1, 100.0, 101.0)
    with pytest.raises(ValueError, match="negative"):
        CostScenario("invalid", -0.1)
    with pytest.raises(ValueError, match="direction"):
        _observation(0, 2, 100.0, 101.0)
    with pytest.raises(ValueError, match="positive"):
        _observation(0, 1, 0.0, 101.0)
    with pytest.raises(ValueError, match="day"):
        TradeObservation(
            timestamp=valid.timestamp,
            exit_timestamp=valid.exit_timestamp,
            day=date(2026, 6, 20),
            direction=1,
            entry_price=100.0,
            exit_price=101.0,
        )
    with pytest.raises(ValueError, match="chronological"):
        simulate((_observation(5, 1, 100, 101), valid), CostScenario("none", 0))
    with pytest.raises(ValueError, match="overlap"):
        simulate((_observation(0, 1, 100, 101), _observation(4, 1, 100, 101)), CostScenario("none", 0))

    adjacent = simulate(
        (_observation(0, 1, 100, 101), _observation(5, 1, 100, 101)),
        CostScenario("none", 0),
    )
    assert adjacent.metrics.trade_count == 2


def test_sign_classification_reports_binary_and_ternary_confusion() -> None:
    ternary = sign_classification_metrics(
        [-2, -1, 0, 1, 3, 5],
        [-1, 2, 0, 4, 0, -3],
        mode="ternary",
    )
    assert ternary.labels == (-1, 0, 1)
    assert ternary.confusion == ((1, 0, 1), (0, 1, 0), (1, 1, 1))
    assert ternary.accuracy == pytest.approx(0.5)
    assert ternary.balanced_accuracy == pytest.approx((0.5 + 1.0 + 1 / 3) / 3)
    assert ternary.mcc == pytest.approx(6 / (528**0.5))

    binary = sign_classification_metrics([-1, -1, 1, 1], [-1, 1, 1, 1], mode="binary")
    assert binary.confusion == ((1, 1), (0, 2))
    assert binary.balanced_accuracy == pytest.approx(0.75)
    assert binary.mcc == pytest.approx(1 / (3**0.5))
    with pytest.raises(ValueError, match="zero"):
        sign_classification_metrics([-1, 0, 1], [-1, 1, 1], mode="binary")


def test_correlations_handle_ties_and_degenerate_series() -> None:
    metrics = correlation_metrics([1, 2, 2, 4], [4, 3, 3, 1])
    assert metrics.sample_count == 4
    assert metrics.spearman == pytest.approx(-1.0)
    assert metrics.pearson is not None and metrics.pearson < -0.99

    degenerate = correlation_metrics([1, 1, 1], [1, 2, 3])
    assert degenerate.pearson is None
    assert degenerate.spearman is None


def test_day_block_bootstrap_is_deterministic_and_resamples_complete_blocks() -> None:
    first_day = date(2026, 6, 19)
    second_day = date(2026, 6, 20)
    observations = ((first_day, 1.0), (first_day, 3.0), (second_day, 10.0))

    first = day_block_bootstrap_ci(
        observations,
        statistic=fmean,
        confidence=0.8,
        resamples=200,
        seed=29,
    )
    second = day_block_bootstrap_ci(
        observations,
        statistic=fmean,
        confidence=0.8,
        resamples=200,
        seed=29,
    )

    assert first == second
    assert first.point_estimate == pytest.approx(14 / 3)
    assert first.lower <= first.point_estimate <= first.upper
    assert first.day_count == 2
    assert first.resamples == 200


def test_cash_and_buy_hold_benchmarks_use_one_round_trip() -> None:
    observations = (
        _observation(0, 0, 100.0, 101.0),
        _observation(5, 0, 101.0, 120.0),
    )
    costs = CostScenario("five bps each way", 5.0)
    cash = cash_benchmark(initial_capital=25_000)
    buy_hold = buy_and_hold_benchmark(observations, costs, initial_capital=25_000)

    assert cash.net_return == 0.0
    assert buy_hold.gross_return == pytest.approx(0.2)
    assert buy_hold.net_return == pytest.approx(0.199)
    assert buy_hold.pnl_per_initial_capital == pytest.approx(4_975)
