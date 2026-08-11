"""Deterministic, I/O-free research metrics for fixed-horizon forecasts.

This standalone module models completed, non-overlapping observations rather than an
execution venue. Costs are an explicit return haircut on both sides of every selected
round trip; fills, queue position, spread, market impact, and financing remain outside
the model and must be handled by the caller's research assumptions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from math import fsum, inf, isfinite, prod, sqrt
from numbers import Integral, Real
from random import Random
from statistics import fmean, median, stdev
from typing import Literal

BacktestMode = Literal["long_only", "long_short"]
ClassificationMode = Literal["binary", "ternary"]


@dataclass(frozen=True, slots=True)
class TradeObservation:
    """One fixed, completed forecast observation with no assumed execution state."""

    timestamp: datetime
    exit_timestamp: datetime
    day: date
    direction: int
    entry_price: float
    exit_price: float

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, datetime) or not isinstance(
            self.exit_timestamp, datetime
        ):
            raise TypeError("timestamp and exit_timestamp must be datetime instances")
        if isinstance(self.day, datetime) or not isinstance(self.day, date):
            raise TypeError("day must be a date instance")
        if _is_aware(self.timestamp) != _is_aware(self.exit_timestamp):
            raise ValueError("timestamp and exit_timestamp must use compatible timezone forms")
        if self.exit_timestamp <= self.timestamp:
            raise ValueError("exit_timestamp must be later than timestamp")
        if self.day != self.timestamp.date():
            raise ValueError("day must equal timestamp.date()")
        if isinstance(self.direction, bool) or not isinstance(self.direction, Integral):
            raise TypeError("direction must be an integer")
        if self.direction not in (-1, 0, 1):
            raise ValueError("direction must be -1, 0, or 1")
        object.__setattr__(self, "direction", int(self.direction))
        object.__setattr__(self, "entry_price", _positive_float(self.entry_price, "entry_price"))
        object.__setattr__(self, "exit_price", _positive_float(self.exit_price, "exit_price"))

    @property
    def exposure_minutes(self) -> float:
        """Wall-clock minutes between the fixed entry and exit timestamps."""

        return (self.exit_timestamp - self.timestamp).total_seconds() / 60.0


@dataclass(frozen=True, slots=True)
class CostScenario:
    """A named cost haircut, in basis points, independently charged each way."""

    name: str
    one_way_bps: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("cost scenario name must be non-empty")
        value = _finite_float(self.one_way_bps, "one_way_bps")
        if value < 0.0:
            raise ValueError("one_way_bps cannot be negative")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "one_way_bps", value)

    @property
    def round_trip_return_cost(self) -> float:
        """Fixed return haircut for one entry and one exit."""

        return 2.0 * self.one_way_bps / 10_000.0


@dataclass(frozen=True, slots=True)
class SimulatedTrade:
    observation: TradeObservation
    gross_return: float
    cost_return: float
    net_return: float
    pnl: float


@dataclass(frozen=True, slots=True)
class DailyReturn:
    day: date
    gross_return: float
    net_return: float


@dataclass(frozen=True, slots=True)
class BacktestMetrics:
    """Fixed-notional performance and daily risk statistics.

    ``total_return`` fields sum fixed-initial-notional trade returns. ``compounded_return``
    fields instead reinvest each sequential trade. VaR and CVaR are lower-tail daily return
    thresholds (negative values denote losses), not positive loss magnitudes. Turnover is
    round-trip notional divided by initial capital, so each trade contributes two.
    """

    gross_total_return: float
    net_total_return: float
    gross_compounded_return: float
    net_compounded_return: float
    pnl_per_initial_capital: float
    trade_count: int
    wins: int
    losses: int
    breakeven: int
    hit_rate: float | None
    average_trade_return: float | None
    median_trade_return: float | None
    profit_factor: float | None
    turnover: float
    exposure_minutes: float
    max_drawdown: float
    annualized_daily_sharpe: float | None
    annualized_daily_sortino: float | None
    annualized_calmar: float | None
    var_95: float | None
    cvar_95: float | None


@dataclass(frozen=True, slots=True)
class BacktestResult:
    mode: BacktestMode
    costs: CostScenario
    initial_capital: float
    trades: tuple[SimulatedTrade, ...]
    daily_returns: tuple[DailyReturn, ...]
    metrics: BacktestMetrics


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    gross_return: float
    net_return: float
    pnl_per_initial_capital: float


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Sign classification summary; confusion rows are actual and columns predicted."""

    mode: ClassificationMode
    labels: tuple[int, ...]
    confusion: tuple[tuple[int, ...], ...]
    sample_count: int
    accuracy: float
    balanced_accuracy: float
    mcc: float | None


@dataclass(frozen=True, slots=True)
class CorrelationMetrics:
    sample_count: int
    pearson: float | None
    spearman: float | None


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    seed: int
    day_count: int


def simulate(
    observations: Sequence[TradeObservation],
    costs: CostScenario,
    *,
    mode: BacktestMode = "long_only",
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Simulate sequential fixed-initial-notional trades from completed observations."""

    ordered = _validate_observations(observations)
    if not isinstance(costs, CostScenario):
        raise TypeError("costs must be a CostScenario")
    if mode not in ("long_only", "long_short"):
        raise ValueError("mode must be 'long_only' or 'long_short'")
    capital = _positive_float(initial_capital, "initial_capital")
    daily_gross = {item.day: 0.0 for item in ordered}
    daily_net = {item.day: 0.0 for item in ordered}
    selected: list[SimulatedTrade] = []

    for item in ordered:
        if item.direction == 0 or (mode == "long_only" and item.direction < 0):
            continue
        gross_return = item.direction * (item.exit_price / item.entry_price - 1.0)
        cost_return = costs.round_trip_return_cost
        net_return = gross_return - cost_return
        trade = SimulatedTrade(
            observation=item,
            gross_return=gross_return,
            cost_return=cost_return,
            net_return=net_return,
            pnl=capital * net_return,
        )
        selected.append(trade)
        daily_gross[item.day] += gross_return
        daily_net[item.day] += net_return

    daily = tuple(
        DailyReturn(day=day, gross_return=daily_gross[day], net_return=daily_net[day])
        for day in sorted(daily_net)
    )
    net_returns = tuple(item.net_return for item in selected)
    gross_returns = tuple(item.gross_return for item in selected)
    total_net = fsum(net_returns)
    positive = fsum(value for value in net_returns if value > 0.0)
    negative = fsum(value for value in net_returns if value < 0.0)
    wins = sum(value > 0.0 for value in net_returns)
    losses = sum(value < 0.0 for value in net_returns)
    trade_count = len(selected)
    daily_net_values = tuple(item.net_return for item in daily)
    max_drawdown = _fixed_notional_max_drawdown(net_returns)
    metrics = BacktestMetrics(
        gross_total_return=fsum(gross_returns),
        net_total_return=total_net,
        gross_compounded_return=prod(1.0 + value for value in gross_returns) - 1.0,
        net_compounded_return=prod(1.0 + value for value in net_returns) - 1.0,
        pnl_per_initial_capital=capital * total_net,
        trade_count=trade_count,
        wins=wins,
        losses=losses,
        breakeven=trade_count - wins - losses,
        hit_rate=wins / trade_count if trade_count else None,
        average_trade_return=fmean(net_returns) if net_returns else None,
        median_trade_return=median(net_returns) if net_returns else None,
        profit_factor=_profit_factor(positive, negative),
        turnover=2.0 * trade_count,
        exposure_minutes=fsum(item.observation.exposure_minutes for item in selected),
        max_drawdown=max_drawdown,
        annualized_daily_sharpe=_daily_sharpe(daily_net_values),
        annualized_daily_sortino=_daily_sortino(daily_net_values),
        annualized_calmar=_calmar(daily_net_values, max_drawdown),
        var_95=_percentile(daily_net_values, 0.05) if daily_net_values else None,
        cvar_95=_lower_tail_mean(daily_net_values, 0.05) if daily_net_values else None,
    )
    return BacktestResult(
        mode=mode,
        costs=costs,
        initial_capital=capital,
        trades=tuple(selected),
        daily_returns=daily,
        metrics=metrics,
    )


def cash_benchmark(*, initial_capital: float = 100_000.0) -> BenchmarkResult:
    """Return the zero-return cash baseline after validating the capital scale."""

    _positive_float(initial_capital, "initial_capital")
    return BenchmarkResult(
        name="cash",
        gross_return=0.0,
        net_return=0.0,
        pnl_per_initial_capital=0.0,
    )


def buy_and_hold_benchmark(
    observations: Sequence[TradeObservation],
    costs: CostScenario,
    *,
    initial_capital: float = 100_000.0,
) -> BenchmarkResult:
    """Buy at the first observation entry and sell at the final observation exit."""

    ordered = _validate_observations(observations)
    if not ordered:
        raise ValueError("buy-and-hold requires at least one observation")
    if not isinstance(costs, CostScenario):
        raise TypeError("costs must be a CostScenario")
    capital = _positive_float(initial_capital, "initial_capital")
    gross_return = ordered[-1].exit_price / ordered[0].entry_price - 1.0
    net_return = gross_return - costs.round_trip_return_cost
    return BenchmarkResult(
        name="buy_and_hold",
        gross_return=gross_return,
        net_return=net_return,
        pnl_per_initial_capital=capital * net_return,
    )


def sign_classification_metrics(
    actual: Sequence[Real],
    predicted: Sequence[Real],
    *,
    mode: ClassificationMode = "ternary",
) -> ClassificationMetrics:
    """Measure confusion, balanced accuracy, and MCC after converting values to signs.

    Binary mode requires both series to contain only non-zero signs. Ternary mode treats
    an exact zero as its own flat/abstain class.
    """

    actual_values, predicted_values = _paired_floats(actual, predicted)
    if mode == "binary":
        labels = (-1, 1)
    elif mode == "ternary":
        labels = (-1, 0, 1)
    else:
        raise ValueError("mode must be 'binary' or 'ternary'")
    actual_signs = tuple(_sign(value) for value in actual_values)
    predicted_signs = tuple(_sign(value) for value in predicted_values)
    if mode == "binary" and (0 in actual_signs or 0 in predicted_signs):
        raise ValueError("binary mode does not accept zero-valued signs")

    index = {label: position for position, label in enumerate(labels)}
    cells = [[0 for _ in labels] for _ in labels]
    for actual_sign, predicted_sign in zip(actual_signs, predicted_signs, strict=True):
        cells[index[actual_sign]][index[predicted_sign]] += 1
    confusion = tuple(tuple(row) for row in cells)
    sample_count = len(actual_signs)
    correct = sum(confusion[index][index] for index in range(len(labels)))
    actual_totals = tuple(sum(row) for row in confusion)
    predicted_totals = tuple(
        sum(confusion[row][column] for row in range(len(labels)))
        for column in range(len(labels))
    )
    recalls = tuple(
        confusion[position][position] / count
        for position, count in enumerate(actual_totals)
        if count
    )
    return ClassificationMetrics(
        mode=mode,
        labels=labels,
        confusion=confusion,
        sample_count=sample_count,
        accuracy=correct / sample_count,
        balanced_accuracy=fmean(recalls),
        mcc=_multiclass_mcc(confusion, actual_totals, predicted_totals),
    )


def correlation_metrics(
    actual: Sequence[Real], predicted: Sequence[Real]
) -> CorrelationMetrics:
    """Return Pearson and average-rank Spearman correlations without SciPy."""

    actual_values, predicted_values = _paired_floats(actual, predicted)
    return CorrelationMetrics(
        sample_count=len(actual_values),
        pearson=_pearson(actual_values, predicted_values),
        spearman=_pearson(_average_ranks(actual_values), _average_ranks(predicted_values)),
    )


def day_block_bootstrap_ci(
    observations: Sequence[tuple[date, Real]],
    *,
    statistic: Callable[[Sequence[float]], float] = fmean,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = 17,
) -> BootstrapInterval:
    """Bootstrap whole day blocks with replacement using a deterministic local RNG."""

    if not isinstance(resamples, Integral) or isinstance(resamples, bool) or resamples <= 0:
        raise ValueError("resamples must be a positive integer")
    confidence_value = _finite_float(confidence, "confidence")
    if not 0.0 < confidence_value < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    if not isinstance(seed, Integral) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not callable(statistic):
        raise TypeError("statistic must be callable")

    blocks: dict[date, list[float]] = defaultdict(list)
    for item in observations:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("each bootstrap observation must be a (day, value) tuple")
        day, value = item
        if isinstance(day, datetime) or not isinstance(day, date):
            raise TypeError("bootstrap block keys must be date instances")
        blocks[day].append(_finite_float(value, "bootstrap value"))
    if not blocks:
        raise ValueError("bootstrap requires at least one observation")

    days = tuple(sorted(blocks))
    ordered_values = tuple(value for day in days for value in blocks[day])
    point_estimate = _finite_statistic(statistic, ordered_values)
    rng = Random(int(seed))
    estimates: list[float] = []
    for _ in range(int(resamples)):
        sample: list[float] = []
        for _ in days:
            sampled_day = days[rng.randrange(len(days))]
            sample.extend(blocks[sampled_day])
        estimates.append(_finite_statistic(statistic, sample))
    tail = (1.0 - confidence_value) / 2.0
    return BootstrapInterval(
        point_estimate=point_estimate,
        lower=_percentile(estimates, tail),
        upper=_percentile(estimates, 1.0 - tail),
        confidence=confidence_value,
        resamples=int(resamples),
        seed=int(seed),
        day_count=len(days),
    )


def _validate_observations(
    observations: Sequence[TradeObservation],
) -> tuple[TradeObservation, ...]:
    ordered = tuple(observations)
    previous: TradeObservation | None = None
    for item in ordered:
        if not isinstance(item, TradeObservation):
            raise TypeError("observations must contain TradeObservation instances")
        if previous is not None:
            try:
                if item.timestamp <= previous.timestamp:
                    raise ValueError("observations must be strictly chronological")
                if item.timestamp < previous.exit_timestamp:
                    raise ValueError("observations cannot overlap")
            except TypeError as error:
                raise ValueError("all observation timestamps must use compatible timezones") from error
        previous = item
    return ordered


def _fixed_notional_max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = equity
    maximum = 0.0
    for value in returns:
        equity += value
        peak = max(peak, equity)
        if peak > 0.0:
            maximum = max(maximum, (peak - equity) / peak)
    return maximum


def _profit_factor(positive: float, negative: float) -> float | None:
    if negative < 0.0:
        return positive / abs(negative)
    if positive > 0.0:
        return inf
    return None


def _daily_sharpe(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    volatility = stdev(values)
    if volatility == 0.0:
        return None
    return sqrt(365.0) * fmean(values) / volatility


def _daily_sortino(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    downside_deviation = sqrt(fmean(min(value, 0.0) ** 2 for value in values))
    if downside_deviation == 0.0:
        return None
    return sqrt(365.0) * fmean(values) / downside_deviation


def _calmar(values: Sequence[float], max_drawdown: float) -> float | None:
    if not values or max_drawdown == 0.0:
        return None
    growth = prod(1.0 + value for value in values)
    if growth <= 0.0:
        return None
    annualized_return = growth ** (365.0 / len(values)) - 1.0
    return annualized_return / max_drawdown


def _lower_tail_mean(values: Sequence[float], quantile: float) -> float:
    threshold = _percentile(values, quantile)
    return fmean(value for value in values if value <= threshold)


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_floats(
    actual: Sequence[Real], predicted: Sequence[Real]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have equal length")
    if len(actual) == 0:
        raise ValueError("actual and predicted cannot be empty")
    return (
        tuple(_finite_float(value, "actual value") for value in actual),
        tuple(_finite_float(value, "predicted value") for value in predicted),
    )


def _multiclass_mcc(
    confusion: Sequence[Sequence[int]],
    actual_totals: Sequence[int],
    predicted_totals: Sequence[int],
) -> float | None:
    sample_count = sum(actual_totals)
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    covariance = correct * sample_count - sum(
        predicted * actual
        for predicted, actual in zip(predicted_totals, actual_totals, strict=True)
    )
    predicted_variance = sample_count**2 - sum(value**2 for value in predicted_totals)
    actual_variance = sample_count**2 - sum(value**2 for value in actual_totals)
    denominator = sqrt(predicted_variance * actual_variance)
    return covariance / denominator if denominator else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 2:
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    left_centered = tuple(value - left_mean for value in left)
    right_centered = tuple(value - right_mean for value in right)
    denominator = sqrt(
        fsum(value * value for value in left_centered)
        * fsum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return None
    return fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    ) / denominator


def _average_ranks(values: Sequence[float]) -> tuple[float, ...]:
    positions = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(positions):
        end = start + 1
        while end < len(positions) and values[positions[end]] == values[positions[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in positions[start:end]:
            ranks[position] = average_rank
        start = end
    return tuple(ranks)


def _finite_statistic(
    statistic: Callable[[Sequence[float]], float], values: Sequence[float]
) -> float:
    return _finite_float(statistic(values), "bootstrap statistic")


def _sign(value: float) -> int:
    return 1 if value > 0.0 else -1 if value < 0.0 else 0


def _is_aware(value: datetime) -> bool:
    return value.utcoffset() is not None


def _finite_float(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _positive_float(value: Real, name: str) -> float:
    converted = _finite_float(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive")
    return converted
