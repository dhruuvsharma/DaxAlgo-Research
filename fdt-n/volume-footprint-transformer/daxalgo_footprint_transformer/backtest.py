from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TARGET_NAMES, ModelConfig
from .data import EncodedFootprintSeries, iter_binance_trades
from .export import load_checkpoint
from .training import (
    DirectRidgeBaseline,
    WindowDataset,
    collate_windows,
    direct_ridge_baseline_outputs,
    distribution_metrics,
    persistence_baseline_outputs,
    point_prediction_metrics,
)

BACKTEST_SCHEMA_VERSION = 1
DAY_MS = 86_400_000


@dataclass(frozen=True)
class FrozenPolicy:
    horizon_bars: int = 8
    decision_interval_bars: int = 8
    execution_latency_ms: int = 1_000
    signal_edge_bps: float = 24.0
    require_positive_delta: bool = True
    primary_side_cost_bps: float = 12.0
    stress_side_cost_bps: float = 17.0
    initial_capital_usd: float = 10_000.0

    def validate(self) -> None:
        if self.horizon_bars < 1 or self.decision_interval_bars < 1:
            raise ValueError("policy horizons must be positive")
        if self.decision_interval_bars != self.horizon_bars:
            raise ValueError("frozen study requires non-overlapping horizon-aligned decisions")
        if self.execution_latency_ms < 0:
            raise ValueError("execution latency must be non-negative")
        for label, value in (
            ("signal edge", self.signal_edge_bps),
            ("primary side cost", self.primary_side_cost_bps),
            ("stress side cost", self.stress_side_cost_bps),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{label} must be finite and non-negative")
            if value >= 10_000:
                raise ValueError(f"{label} must be below 10,000 bps")
        if self.stress_side_cost_bps < self.primary_side_cost_bps:
            raise ValueError("stress cost must not be below the primary cost")
        if not math.isfinite(self.initial_capital_usd) or self.initial_capital_usd <= 0:
            raise ValueError("initial capital must be finite and positive")


def first_retained_trade_fills(
    paths: Sequence[Path],
    eligible_times_ms: np.ndarray,
    *,
    quantity_scale: float,
    maximum_wait_ms: int = 60_000,
) -> tuple[np.ndarray, np.ndarray]:
    requested = np.asarray(eligible_times_ms, dtype=np.int64)
    if requested.ndim != 1 or requested.size == 0 or np.any(np.diff(requested) < 0):
        raise ValueError("eligible execution times must be a non-empty chronological vector")
    if maximum_wait_ms < 1:
        raise ValueError("maximum execution wait must be positive")
    prices = np.empty(len(requested), dtype=np.float64)
    fill_times = np.empty(len(requested), dtype=np.int64)
    pending = 0
    for timestamp_ms, price, _, _ in iter_binance_trades(paths, quantity_scale=quantity_scale):
        while pending < len(requested) and timestamp_ms >= requested[pending]:
            if timestamp_ms - requested[pending] >= maximum_wait_ms:
                raise ValueError("no retained trade arrived inside an eligible execution minute")
            prices[pending] = price
            fill_times[pending] = timestamp_ms
            pending += 1
        if pending == len(requested):
            break
    if pending != len(requested):
        raise ValueError(
            f"missing retained-trade proxy prices for {len(requested) - pending} events"
        )
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise ValueError("execution proxy prices must be finite and positive")
    return prices, fill_times


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source_archives(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != "Binance Data Vision spot daily individual trades":
        raise ValueError("profitability study requires live-parity individual-trade archives")
    records = manifest.get("source_files")
    if not isinstance(records, list) or not records:
        raise ValueError("dataset manifest has no source archives")
    resolved: list[Path] = []
    for record in records:
        raw = Path(str(record["path"]))
        candidates = [raw] if raw.is_absolute() else [manifest_path.parent / raw, Path.cwd() / raw]
        path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
        if path is None or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing non-empty source archive: {raw}")
        expected_bytes = int(record["bytes"])
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"source archive size mismatch: {path.name}")
        if _sha256(path) != str(record["sha256"]):
            raise ValueError(f"source archive digest mismatch: {path.name}")
        resolved.append(path)
    return resolved


def validate_dataset_manifest(
    manifest_path: Path, series: EncodedFootprintSeries, *, symbol: str = "BTCUSDT"
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source") != "Binance Data Vision spot daily individual trades":
        raise ValueError("dataset manifest is not the required individual-trade source")
    expected = {
        "encoding": series.config.to_dict(),
        "bars": len(series),
        "encoded_price_rows": int(series.row_offsets[-1]),
        "valid_windows": int(len(series.valid_window_indices())),
        "first_timestamp_ms": int(series.timestamps_ms[0]),
        "last_timestamp_ms": int(series.timestamps_ms[-1]),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"dataset manifest does not match encoded series: {key}")
    records = manifest.get("source_files")
    if not isinstance(records, list) or not records:
        raise ValueError("dataset manifest has no source files")
    expected_prefix = f"{symbol.upper()}-trades-"
    if any(not Path(str(record["path"])).name.startswith(expected_prefix) for record in records):
        raise ValueError("dataset source files do not match the frozen symbol")
    return manifest


def utc_date_ms(value: date) -> int:
    return int(datetime.combine(value, time(), tzinfo=UTC).timestamp() * 1_000)


def validate_frozen_members(
    checkpoint_paths: Sequence[Path], training_results_paths: Sequence[Path]
) -> list[int]:
    if len(checkpoint_paths) != 3 or len(training_results_paths) != 3:
        raise ValueError("the frozen ensemble requires exactly three checkpoints and results")
    if len({path.resolve() for path in checkpoint_paths}) != 3 or len(
        {path.resolve() for path in training_results_paths}
    ) != 3:
        raise ValueError("frozen ensemble members must be distinct artifacts")
    expected_config = ModelConfig.from_preset("fdt-n-tiny").to_dict()
    checkpoint_seeds: list[int] = []
    checkpoint_records: list[dict[str, Any]] = []
    for path in checkpoint_paths:
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("model_config") != expected_config:
            raise ValueError(f"checkpoint is not the frozen Tiny preset: {path}")
        checkpoint_seeds.append(int(saved["train_config"]["seed"]))
        checkpoint_records.append(saved)
    result_seeds: list[int] = []
    result_records: list[dict[str, Any]] = []
    for path in training_results_paths:
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("model_config") != expected_config or result.get("parameter_count") != 59_797:
            raise ValueError(f"training result is not the frozen Tiny preset: {path}")
        result_seeds.append(int(result["train_config"]["seed"]))
        result_records.append(result)
    required = [17, 29, 43]
    if sorted(checkpoint_seeds) != required or sorted(result_seeds) != required:
        raise ValueError("the frozen ensemble must contain seeds 17, 29, and 43 exactly once")
    if checkpoint_seeds != result_seeds:
        raise ValueError("checkpoint and training-result orders must align by seed")
    common_train_config: str | None = None
    common_encoding: str | None = None
    for checkpoint, result in zip(checkpoint_records, result_records, strict=True):
        checkpoint_train = dict(checkpoint["train_config"])
        result_train = dict(result["train_config"])
        checkpoint_train.pop("seed")
        result_train.pop("seed")
        if checkpoint_train != result_train:
            raise ValueError("checkpoint and result training configurations do not match")
        if checkpoint["encoding_config"] != result["encoding_config"]:
            raise ValueError("checkpoint and result encoding configurations do not match")
        if int(checkpoint["epoch"]) != int(result["best_epoch"]):
            raise ValueError("checkpoint epoch does not match its training result")
        serialized_train = json.dumps(result_train, sort_keys=True)
        serialized_encoding = json.dumps(result["encoding_config"], sort_keys=True)
        if common_train_config is None:
            common_train_config = serialized_train
            common_encoding = serialized_encoding
        elif serialized_train != common_train_config or serialized_encoding != common_encoding:
            raise ValueError("frozen ensemble members do not share one training protocol")
    return checkpoint_seeds


def load_preregistered_study(
    preregistration_path: Path,
    development_dataset_path: Path,
    development_manifest_path: Path,
    checkpoint_paths: Sequence[Path],
    training_results_paths: Sequence[Path],
    ridge_path: Path,
) -> tuple[dict[str, Any], FrozenPolicy, date, date, list[int]]:
    record = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("status") != (
        "preregistered_before_fresh_holdout_download"
    ):
        raise ValueError("profitability study preregistration is invalid or not frozen")
    development = record["development"]
    if _sha256(development_dataset_path) != development["dataset_sha256"]:
        raise ValueError("development dataset does not match the preregistration")
    if _sha256(development_manifest_path) != development["manifest_sha256"]:
        raise ValueError("development manifest does not match the preregistration")
    member_seeds = validate_frozen_members(checkpoint_paths, training_results_paths)
    members = record["model"]["members"]
    if [int(member["seed"]) for member in members] != member_seeds:
        raise ValueError("preregistered member order does not match the supplied ensemble")
    for member, checkpoint, result in zip(
        members, checkpoint_paths, training_results_paths, strict=True
    ):
        if _sha256(checkpoint) != member["checkpoint_sha256"]:
            raise ValueError(f"checkpoint hash does not match preregistration: {checkpoint}")
        if _sha256(result) != member["training_results_sha256"]:
            raise ValueError(f"training-result hash does not match preregistration: {result}")
    if _sha256(ridge_path) != record["model"]["ridge_sha256"]:
        raise ValueError("ridge artifact does not match the preregistration")
    policy_record = record["policy"]
    policy = FrozenPolicy(
        horizon_bars=int(policy_record["horizon_bars"]),
        decision_interval_bars=int(policy_record["decision_interval_bars"]),
        execution_latency_ms=int(policy_record["execution_latency_ms"]),
        signal_edge_bps=float(policy_record["signal_edge_bps"]),
        require_positive_delta=bool(policy_record["require_positive_delta"]),
        primary_side_cost_bps=float(policy_record["primary_side_cost_bps"]),
        stress_side_cost_bps=float(policy_record["stress_side_cost_bps"]),
        initial_capital_usd=float(policy_record["initial_capital_usd"]),
    )
    policy.validate()
    holdout = record["holdout"]
    return (
        record,
        policy,
        date.fromisoformat(holdout["evaluation_start_utc"]),
        date.fromisoformat(holdout["evaluation_end_exclusive_utc"]),
        member_seeds,
    )


def select_decision_indices(
    series: EncodedFootprintSeries,
    evaluation_start_ms: int,
    evaluation_end_ms: int,
    policy: FrozenPolicy,
) -> np.ndarray:
    policy.validate()
    if evaluation_end_ms <= evaluation_start_ms:
        raise ValueError("evaluation end must follow its start")
    if policy.horizon_bars > series.config.horizon_bars:
        raise ValueError("policy horizon exceeds the encoded target horizon")
    interval_ms = series.config.interval_seconds * 1_000
    valid = series.valid_window_indices()
    decision_times = series.timestamps_ms[valid] + interval_ms
    exit_times = decision_times + policy.horizon_bars * interval_ms
    aligned = (decision_times // interval_ms) % policy.decision_interval_bars == 0
    selected = valid[
        aligned & (decision_times >= evaluation_start_ms) & (exit_times <= evaluation_end_ms)
    ]
    if selected.size == 0:
        raise ValueError("evaluation interval contains no complete policy decisions")
    return selected


@torch.inference_mode()
def predict_ensemble(
    series: EncodedFootprintSeries,
    indices: np.ndarray,
    checkpoint_paths: Sequence[Path],
    *,
    batch_size: int = 32,
    device_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, list[int]]:
    if not checkpoint_paths:
        raise ValueError("at least one checkpoint is required")
    if batch_size < 1:
        raise ValueError("prediction batch size must be positive")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    dataset = WindowDataset(series, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_windows)
    prediction_sum: np.ndarray | None = None
    targets: np.ndarray | None = None
    masks: np.ndarray | None = None
    parameter_counts: list[int] = []
    for checkpoint_index, checkpoint_path in enumerate(checkpoint_paths):
        model = load_checkpoint(checkpoint_path)
        if model.encoding != series.config:
            raise ValueError(f"checkpoint encoding does not match dataset: {checkpoint_path}")
        parameter_counts.append(model.parameter_count)
        model.to(device)
        model.eval()
        outputs: list[np.ndarray] = []
        target_batches: list[np.ndarray] = []
        mask_batches: list[np.ndarray] = []
        for row, row_mask, bar, target, target_mask in loader:
            row = row.to(device)
            row_mask = row_mask.to(device)
            bar = bar.to(device)
            bar_mask = torch.ones(bar.shape[:2], dtype=torch.bool, device=device)
            outputs.append(model(row, row_mask, bar, bar_mask).cpu().numpy())
            if checkpoint_index == 0:
                target_batches.append(target.numpy())
                mask_batches.append(target_mask.numpy())
        prediction = np.concatenate(outputs)
        if prediction_sum is None:
            prediction_sum = prediction.astype(np.float64)
            targets = np.concatenate(target_batches)
            masks = np.concatenate(mask_batches)
        else:
            prediction_sum += prediction
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if prediction_sum is None or targets is None or masks is None:
        raise RuntimeError("ensemble inference produced no predictions")
    ensemble = prediction_sum / len(checkpoint_paths)
    if not np.isfinite(ensemble).all() or not np.isfinite(targets).all():
        raise ValueError("ensemble inference produced non-finite values")
    if np.any(ensemble[..., 0] > ensemble[..., 1]) or np.any(
        ensemble[..., 1] > ensemble[..., 2]
    ):
        raise ValueError("ensemble inference violated quantile ordering")
    return (
        ensemble,
        targets,
        masks,
        str(device),
        parameter_counts,
    )


def policy_signals(
    predictions: np.ndarray,
    series: EncodedFootprintSeries,
    indices: np.ndarray,
    policy: FrozenPolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim == 4 and values.shape[-1] == 3:
        points = values[..., 1]
    elif values.ndim == 3:
        points = values
    else:
        raise ValueError("policy predictions must be point or three-quantile tensors")
    if points.shape[:2] != (len(indices), series.config.horizon_bars):
        raise ValueError("policy predictions do not align with the selected windows")
    horizon = policy.horizon_bars - 1
    poc_index = TARGET_NAMES.index("poc_offset")
    delta_index = TARGET_NAMES.index("delta_fraction")
    context_poc_prices = (
        series.bar_values[np.asarray(indices, dtype=np.int64), 0] * series.config.row_size
    )
    displacement = (
        np.clip(points[:, horizon, poc_index], -2.0, 2.0)
        * series.config.price_scale_ticks
        * series.config.row_size
    )
    edge_bps = displacement / context_poc_prices * 10_000.0
    delta = points[:, horizon, delta_index]
    positions = edge_bps > policy.signal_edge_bps
    if policy.require_positive_delta:
        positions &= delta > 0.0
    return positions.astype(np.int8), edge_bps, delta


def _iso(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC).isoformat()


def _annualized_ratios(daily_returns: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(daily_returns, dtype=np.float64)
    if values.size < 2:
        return None, None
    standard_deviation = float(values.std(ddof=1))
    sharpe = (
        float(values.mean() / standard_deviation * math.sqrt(365.0))
        if standard_deviation > 1e-12
        else None
    )
    downside = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    sortino = float(values.mean() / downside * math.sqrt(365.0)) if downside > 1e-12 else None
    return sharpe, sortino


def _circular_block_bootstrap(
    daily_returns: np.ndarray,
    *,
    resamples: int = 2_000,
    block_days: int = 3,
    seed: int = 17,
) -> dict[str, object]:
    values = np.asarray(daily_returns, dtype=np.float64)
    if values.size < block_days or resamples < 1:
        return {"resamples": 0, "block_days": block_days, "net_return_ci_95": None}
    generator = np.random.default_rng(seed)
    total_returns = np.empty(resamples, dtype=np.float64)
    sharpes: list[float] = []
    for sample_index in range(resamples):
        sampled: list[float] = []
        while len(sampled) < len(values):
            start = int(generator.integers(0, len(values)))
            sampled.extend(values[(start + offset) % len(values)] for offset in range(block_days))
        draw = np.asarray(sampled[: len(values)], dtype=np.float64)
        total_returns[sample_index] = float(np.prod(1.0 + draw) - 1.0)
        sharpe, _ = _annualized_ratios(draw)
        if sharpe is not None and math.isfinite(sharpe):
            sharpes.append(sharpe)
    return {
        "resamples": resamples,
        "block_days": block_days,
        "seed": seed,
        "net_return_ci_95": [
            float(np.quantile(total_returns, 0.025)),
            float(np.quantile(total_returns, 0.975)),
        ],
        "sharpe_ci_95": (
            [float(np.quantile(sharpes, 0.025)), float(np.quantile(sharpes, 0.975))]
            if sharpes
            else None
        ),
    }


def simulate_blocks(
    entry_times_ms: np.ndarray,
    exit_times_ms: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    positions: np.ndarray,
    *,
    initial_capital_usd: float,
    side_cost_bps: float,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    arrays = [
        np.asarray(entry_times_ms, dtype=np.int64),
        np.asarray(exit_times_ms, dtype=np.int64),
        np.asarray(entry_prices, dtype=np.float64),
        np.asarray(exit_prices, dtype=np.float64),
        np.asarray(positions, dtype=np.int8),
    ]
    if len({len(array) for array in arrays}) != 1 or not len(arrays[0]):
        raise ValueError("backtest block arrays must be aligned and non-empty")
    if np.any(np.diff(arrays[0]) <= 0) or np.any(arrays[1] <= arrays[0]):
        raise ValueError("backtest blocks must be chronological with positive duration")
    if len(arrays[0]) > 1 and np.any(arrays[0][1:] < arrays[1][:-1]):
        raise ValueError("backtest blocks must not overlap")
    if np.any(arrays[2] <= 0) or np.any(arrays[3] <= 0):
        raise ValueError("backtest execution prices must be positive")
    if np.any((arrays[4] != 0) & (arrays[4] != 1)):
        raise ValueError("spot policy positions must be long or flat")
    entry_times, exit_times, entries, exits, selected = arrays
    side_cost = side_cost_bps / 10_000.0
    gross_equity = 1.0
    net_equity = 1.0
    equity_curve = [net_equity]
    trades: list[dict[str, object]] = []
    daily_factors: dict[int, float] = {}
    for index in range(len(entry_times)):
        gross_factor = exits[index] / entries[index] if selected[index] else 1.0
        net_factor = (1.0 - side_cost) ** 2 * gross_factor if selected[index] else 1.0
        equity_before = net_equity
        gross_equity *= gross_factor
        net_equity *= net_factor
        equity_curve.append(net_equity)
        day = int(entry_times[index] // DAY_MS)
        daily_factors[day] = daily_factors.get(day, 1.0) * net_factor
        if selected[index]:
            trades.append(
                {
                    "entry_time": _iso(int(entry_times[index])),
                    "exit_time": _iso(int(exit_times[index])),
                    "entry_price": float(entries[index]),
                    "exit_price": float(exits[index]),
                    "gross_return": float(gross_factor - 1.0),
                    "net_return": float(net_factor - 1.0),
                    "net_pnl_usd": float(
                        initial_capital_usd * (net_equity - equity_before)
                    ),
                }
            )
    curve = np.asarray(equity_curve, dtype=np.float64)
    peaks = np.maximum.accumulate(curve)
    max_drawdown = float(np.min(curve / peaks - 1.0))
    daily_returns = np.asarray(
        [daily_factors[key] - 1.0 for key in sorted(daily_factors)], dtype=np.float64
    )
    sharpe, sortino = _annualized_ratios(daily_returns)
    trade_pnls = np.asarray([trade["net_pnl_usd"] for trade in trades], dtype=np.float64)
    positive = float(trade_pnls[trade_pnls > 0].sum()) if trade_pnls.size else 0.0
    negative = float(-trade_pnls[trade_pnls < 0].sum()) if trade_pnls.size else 0.0
    metrics = {
        "blocks": int(len(entry_times)),
        "trades": int(len(trades)),
        "exposure_fraction": float(selected.mean()),
        "turnover_sides": int(2 * len(trades)),
        "side_cost_bps": float(side_cost_bps),
        "gross_return": float(gross_equity - 1.0),
        "net_return": float(net_equity - 1.0),
        "net_pnl_usd": float(initial_capital_usd * (net_equity - 1.0)),
        "trade_hit_rate": float(np.mean(trade_pnls > 0)) if trade_pnls.size else None,
        "profit_factor": float(positive / negative) if negative > 0 else None,
        "block_boundary_max_drawdown": max_drawdown,
        "daily_sharpe_annualized": sharpe,
        "daily_sortino_annualized": sortino,
        "daily_observations": int(len(daily_returns)),
        "bootstrap": _circular_block_bootstrap(daily_returns),
    }
    return metrics, trades, curve, daily_returns


def simulate_buy_and_hold(
    entry_times_ms: np.ndarray,
    exit_times_ms: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    *,
    initial_capital_usd: float,
    side_cost_bps: float,
) -> dict[str, object]:
    start_price = float(entry_prices[0])
    side_cost = side_cost_bps / 10_000.0
    marked = (1.0 - side_cost) * np.asarray(exit_prices, dtype=np.float64) / start_price
    marked[-1] *= 1.0 - side_cost
    curve = np.concatenate(([1.0], marked))
    peaks = np.maximum.accumulate(curve)
    daily_marks: dict[int, float] = {}
    for timestamp, value in zip(entry_times_ms, marked, strict=True):
        daily_marks[int(timestamp // DAY_MS)] = float(value)
    ordered = np.asarray([daily_marks[key] for key in sorted(daily_marks)], dtype=np.float64)
    daily_returns = np.diff(np.concatenate(([1.0], ordered))) / np.concatenate(
        ([1.0], ordered[:-1])
    )
    sharpe, sortino = _annualized_ratios(daily_returns)
    gross_return = float(exit_prices[-1] / entry_prices[0] - 1.0)
    net_return = float(marked[-1] - 1.0)
    return {
        "blocks": int(len(entry_times_ms)),
        "trades": 1,
        "exposure_fraction": 1.0,
        "turnover_sides": 2,
        "side_cost_bps": float(side_cost_bps),
        "gross_return": gross_return,
        "net_return": net_return,
        "net_pnl_usd": float(initial_capital_usd * net_return),
        "block_boundary_max_drawdown": float(np.min(curve / peaks - 1.0)),
        "daily_sharpe_annualized": sharpe,
        "daily_sortino_annualized": sortino,
        "daily_observations": int(len(daily_returns)),
        "bootstrap": _circular_block_bootstrap(daily_returns),
    }


def _write_trades(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "strategy",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "gross_return",
        "net_return",
        "net_pnl_usd",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _format_percent(value: object) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}%"


def _format_number(value: object, digits: int = 2) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _format_net_return_interval(metric: dict[str, Any]) -> str:
    interval = metric.get("bootstrap", {}).get("net_return_ci_95")
    if interval is None:
        return "-"
    return f"[{_format_percent(interval[0])}, {_format_percent(interval[1])}]"


def _break_even_side_cost_bps(metric: dict[str, Any]) -> float | None:
    trades = int(metric.get("trades", 0))
    gross_return = float(metric.get("gross_return", 0.0))
    if trades < 1 or gross_return <= -1.0:
        return None
    return 10_000.0 * (1.0 - (1.0 + gross_return) ** (-1.0 / (2.0 * trades)))


def _write_investor_report(path: Path, result: dict[str, Any]) -> None:
    financial = result["financial_metrics"]
    economic_rows = []
    risk_rows = []
    for key, label in (
        ("fdt_n_tiny_ensemble_primary", "Tiny FDT-N ensemble"),
        ("direct_ridge_primary", "Direct ridge"),
        ("buy_and_hold_primary", "BTC buy-and-hold"),
        ("fdt_n_tiny_ensemble_stress", "Tiny FDT-N stress costs"),
        ("no_trade", "Cash / no trade"),
    ):
        metric = financial[key]
        economic_rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    _format_percent(metric["gross_return"]),
                    _format_percent(metric["net_return"]),
                    f"${metric['net_pnl_usd']:,.2f}",
                    str(metric["trades"]),
                    _format_percent(metric.get("exposure_fraction", 0.0)),
                    _format_percent(metric.get("block_boundary_max_drawdown")),
                )
            )
            + " |"
        )
        risk_rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    str(metric.get("turnover_sides", 0)),
                    _format_percent(metric.get("trade_hit_rate")),
                    _format_number(metric.get("profit_factor")),
                    _format_number(metric.get("daily_sharpe_annualized")),
                    _format_net_return_interval(metric),
                )
            )
            + " |"
        )
    pooled = result["forecast_metrics"]["fdt_n_tiny_ensemble"]["pooled"]
    forecast_rows = []
    for key, label in (
        ("fdt_n_tiny_ensemble", "Tiny FDT-N ensemble"),
        ("direct_ridge", "Direct ridge"),
        ("persistence", "Persistence"),
    ):
        metric = result["forecast_metrics"][key]["pooled"]
        forecast_rows.append(
            "| "
            + " | ".join(
                (
                    label,
                    f"{metric['median_mae_normalized']:.6f}",
                    _format_percent(metric.get("poc_directional_hit_rate")),
                    _format_percent(metric.get("central_80_coverage")),
                )
            )
            + " |"
        )
    policy = result["policy"]
    evaluation = result["evaluation"]
    model = result["model"]
    provenance = result["data_provenance"]
    calibration = result["development_calibration"]
    primary = financial["fdt_n_tiny_ensemble_primary"]
    break_even_cost = _break_even_side_cost_bps(primary)
    ridge_mae = result["forecast_metrics"]["direct_ridge"]["pooled"][
        "median_mae_normalized"
    ]
    persistence_mae = result["forecast_metrics"]["persistence"]["pooled"][
        "median_mae_normalized"
    ]
    ridge_improvement_pct = 100.0 * (ridge_mae - pooled["median_mae_normalized"]) / ridge_mae
    persistence_improvement_pct = (
        100.0 * (persistence_mae - pooled["median_mae_normalized"]) / persistence_mae
    )
    risk_header = (
        "| Strategy | Charged sides | Net trade hit rate | Profit factor | "
        "Annualized daily Sharpe | 95% bootstrap net-return interval |"
    )
    text = f"""# FDT-N Tiny fresh-holdout evidence report

Status: descriptive research result; not a live track record or investment recommendation.

## Executive decision

**No deployment or capital-allocation recommendation.** The frozen policy produced
{_format_percent(primary['gross_return'])} gross return but {_format_percent(primary['net_return'])}
net return at {policy['primary_side_cost_bps']:.1f} bps per side. The accounting-implied break-even
cost was only {break_even_cost:.2f} bps per side. This experiment supports a reproducible
forecasting research process, but not a claim of profitable or production-ready alpha.

## System under test

- Input: 64 sealed one-minute BTCUSDT footprint bars. Each bar retains price-level buy/sell volume,
  delta and imbalance structure plus deterministic POC, value-area, volume, quality, and time
  aggregates.
- Model: a hierarchical footprint-row pooler, gated bar-feature fusion, causal temporal encoder, and
  direct learned horizon queries. Each ensemble member has {model['parameter_counts'][0]:,}
  parameters; seeds {', '.join(str(seed) for seed in model['seeds'])} are equally weighted.
- Output: q10/q50/q90 marginal forecasts at minutes +1 through +8 for POC, low, high, buy POC,
  sell POC, log volume, and delta fraction. These marginals are forecasts, not orders.
- Trading conversion: the +8 median POC and delta forecasts drive a separate frozen long/flat rule.

## Evaluation scope

- Designated post-development period: {evaluation['start_utc']} to
  {evaluation['end_exclusive_utc']} exclusive ({primary['daily_observations']} UTC daily
  observations).
- {evaluation['decision_examples']:,} non-overlapping decisions; first proxy entry
  {evaluation['first_entry']}; final proxy exit {evaluation['last_exit']}.
- Binance Data Vision BTCUSDT spot individual trades; {provenance['source_archives']} verified daily
  archives.

## Primary economic results

| Strategy | Gross return | Net return | Net PnL | Trades | Exposure | Boundary max drawdown |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(economic_rows)}

{risk_header}
|---|---:|---:|---:|---:|---:|
{chr(10).join(risk_rows)}

Initial capital is ${policy['initial_capital_usd']:,.0f}. Primary execution costs are
{policy['primary_side_cost_bps']:.1f} bps per side; stress costs are
{policy['stress_side_cost_bps']:.1f} bps per side. All model and ridge trades use identical
first-retained-trade proxy-price rules and cost assumptions. Daily Sharpe is annualized with
sqrt(365), sample standard deviation, and a zero risk-free rate; only 30 daily observations make it
descriptive rather than an institutional risk estimate. Drawdown is sampled only at block
boundaries.

## Fresh-holdout forecast diagnostics

| Forecaster | Pooled normalized median MAE | POC directional hit rate | Central 80% coverage |
|---|---:|---:|---:|
{chr(10).join(forecast_rows)}

FDT-N's pooled MAE was {ridge_improvement_pct:.2f}% lower than ridge and
{persistence_improvement_pct:.2f}%
lower than persistence as a point estimate. No fresh-period paired interval establishes superiority,
and POC directional hit rate is a forecast diagnostic rather than a trade win rate.

## Development calibration disclosure

The fixed 0--5 bps threshold grid selected {calibration['selected_signal_edge_bps']:.1f} bps under
the preregistered rule. On development data it made {calibration['selected_trades']} trades,
returned {_format_percent(calibration['selected_gross_return'])} gross, and
{_format_percent(calibration['selected_net_return'])} net. The negative development economics were
known and retained before the designated post-development data was downloaded.

## Frozen protocol

- Equal-weight ensemble of Tiny seeds 17, 29, and 43.
- BTCUSDT spot, long/flat, no leverage or short funding.
- Horizon {policy['horizon_bars']} on UTC-aligned {policy['decision_interval_bars']}-minute blocks.
- Long only when median POC displacement exceeds {policy['signal_edge_bps']:.1f} bps and median
  delta fraction is positive.
- Decision uses a sealed footprint bar; entry waits {policy['execution_latency_ms']} milliseconds
  before using the first retained individual trade as a proxy price. Exit applies the same latency
  after the horizon boundary.
- Each block is an independent round trip, so consecutive long blocks pay two-sided costs again.

## Diligence boundary

This is the first recorded evaluation on the designated post-development period. It is one
instrument, one month, and one rising BTC regime. The local preregistration record predates the
fresh download but does not have an external immutable timestamp. The test does not establish
capacity, market impact, quote-level executable fills, spread, queue position, partial fills,
intrablock drawdown, operational resilience, or a multi-regime institutional track record. The full
audit trail
is in `results.json`, `backtest-series.npz`, and `trades.csv`; the preregistration SHA-256 is
`{provenance['preregistration_sha256']}` and the holdout dataset SHA-256 is
`{provenance['dataset_sha256']}`.
"""
    path.write_text(text, encoding="utf-8")


def run_profitability_study(
    *,
    preregistration_path: Path,
    development_dataset_path: Path,
    dataset_path: Path,
    dataset_manifest_path: Path,
    development_manifest_path: Path,
    checkpoint_paths: Sequence[Path],
    training_results_paths: Sequence[Path],
    ridge_path: Path,
    output_dir: Path,
    batch_size: int = 32,
    device_name: str | None = None,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"backtest output directory already exists: {output_dir}")
    preregistration, frozen, evaluation_start, evaluation_end, member_seeds = (
        load_preregistered_study(
            preregistration_path,
            development_dataset_path,
            development_manifest_path,
            checkpoint_paths,
            training_results_paths,
            ridge_path,
        )
    )
    development_manifest = json.loads(development_manifest_path.read_text(encoding="utf-8"))
    evaluation_start_ms = utc_date_ms(evaluation_start)
    if evaluation_start_ms <= int(development_manifest["last_timestamp_ms"]):
        raise ValueError("fresh holdout must begin after the complete development dataset")
    series = EncodedFootprintSeries.load(dataset_path)
    validate_dataset_manifest(
        dataset_manifest_path, series, symbol=str(preregistration["holdout"]["symbol"])
    )
    if frozen.execution_latency_ms >= series.config.interval_seconds * 1_000:
        raise ValueError("execution latency must remain inside the next boundary minute")
    indices = select_decision_indices(
        series,
        evaluation_start_ms,
        utc_date_ms(evaluation_end),
        frozen,
    )
    evaluation_days = (evaluation_end - evaluation_start).days
    holdout_record = preregistration["holdout"]
    interval_ms = series.config.interval_seconds * 1_000
    decision_times = series.timestamps_ms[indices] + interval_ms
    expected_decisions = evaluation_days * DAY_MS // (
        frozen.decision_interval_bars * interval_ms
    )
    if (
        evaluation_days < 1
        or evaluation_days != int(holdout_record["expected_days"])
        or expected_decisions != int(holdout_record["expected_decisions"])
        or len(indices) != expected_decisions
        or int(decision_times[0]) != evaluation_start_ms
        or int(decision_times[-1] + frozen.decision_interval_bars * interval_ms)
        != utc_date_ms(evaluation_end)
        or np.any(np.diff(decision_times) != frozen.decision_interval_bars * interval_ms)
    ):
        raise ValueError("fresh holdout does not contain the complete preregistered decision grid")
    predictions, targets, masks, device, parameter_counts = predict_ensemble(
        series,
        indices,
        checkpoint_paths,
        batch_size=batch_size,
        device_name=device_name,
    )
    ridge = DirectRidgeBaseline.load(ridge_path)
    ridge_predictions, ridge_targets, ridge_masks = direct_ridge_baseline_outputs(
        ridge, series, indices
    )
    if not (np.array_equal(targets, ridge_targets) and np.array_equal(masks, ridge_masks)):
        raise RuntimeError("ridge and ensemble evaluation windows do not align")
    persistence_predictions, persistence_targets, persistence_masks = persistence_baseline_outputs(
        series, indices
    )
    if not (
        np.array_equal(targets, persistence_targets)
        and np.array_equal(masks, persistence_masks)
    ):
        raise RuntimeError("persistence and ensemble evaluation windows do not align")

    archives = resolve_source_archives(dataset_manifest_path)
    if len(archives) != int(holdout_record["expected_source_archives"]):
        raise ValueError("holdout archive count does not match the preregistration")
    expected_names = {
        f"{holdout_record['symbol']}-trades-{day.isoformat()}.zip"
        for day in (
            date.fromordinal(ordinal)
            for ordinal in range(
                date.fromisoformat(holdout_record["data_start_utc"]).toordinal(),
                date.fromisoformat(holdout_record["data_end_utc"]).toordinal() + 1,
            )
        )
    }
    if {path.name for path in archives} != expected_names:
        raise ValueError("holdout archive dates do not match the preregistration")
    entry_eligible_times = decision_times + frozen.execution_latency_ms
    exit_eligible_times = (
        decision_times + frozen.horizon_bars * interval_ms + frozen.execution_latency_ms
    )
    eligible_times = np.concatenate((entry_eligible_times, exit_eligible_times))
    order = np.argsort(eligible_times, kind="stable")
    sorted_prices, sorted_fill_times = first_retained_trade_fills(
        archives,
        eligible_times[order],
        quantity_scale=series.config.quantity_scale,
    )
    prices = np.empty_like(sorted_prices)
    fill_times = np.empty_like(sorted_fill_times)
    prices[order] = sorted_prices
    fill_times[order] = sorted_fill_times
    entry_prices, exit_prices = np.split(prices, 2)
    entry_times, exit_times = np.split(fill_times, 2)
    ensemble_positions, ensemble_edge_bps, ensemble_delta = policy_signals(
        predictions, series, indices, frozen
    )
    ridge_positions, ridge_edge_bps, ridge_delta = policy_signals(
        ridge_predictions, series, indices, frozen
    )

    ensemble_primary, ensemble_trades, ensemble_curve, ensemble_daily = simulate_blocks(
        entry_times,
        exit_times,
        entry_prices,
        exit_prices,
        ensemble_positions,
        initial_capital_usd=frozen.initial_capital_usd,
        side_cost_bps=frozen.primary_side_cost_bps,
    )
    ensemble_stress, stress_trades, stress_curve, stress_daily = simulate_blocks(
        entry_times,
        exit_times,
        entry_prices,
        exit_prices,
        ensemble_positions,
        initial_capital_usd=frozen.initial_capital_usd,
        side_cost_bps=frozen.stress_side_cost_bps,
    )
    ridge_primary, ridge_trades, ridge_curve, ridge_daily = simulate_blocks(
        entry_times,
        exit_times,
        entry_prices,
        exit_prices,
        ridge_positions,
        initial_capital_usd=frozen.initial_capital_usd,
        side_cost_bps=frozen.primary_side_cost_bps,
    )
    buy_and_hold = simulate_buy_and_hold(
        entry_times,
        exit_times,
        entry_prices,
        exit_prices,
        initial_capital_usd=frozen.initial_capital_usd,
        side_cost_bps=frozen.primary_side_cost_bps,
    )
    for label, metric in (
        ("ensemble", ensemble_primary),
        ("ensemble stress", ensemble_stress),
        ("ridge", ridge_primary),
        ("buy and hold", buy_and_hold),
    ):
        if metric["daily_observations"] != evaluation_days:
            raise RuntimeError(f"{label} daily returns do not match the evaluation day count")

    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output_dir / "backtest-series.npz",
        decision_indices=indices,
        decision_times_ms=decision_times,
        entry_eligible_times_ms=entry_eligible_times,
        exit_eligible_times_ms=exit_eligible_times,
        entry_times_ms=entry_times,
        exit_times_ms=exit_times,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        ensemble_predictions=predictions.astype(np.float32),
        targets=targets,
        target_masks=masks,
        ridge_predictions=ridge_predictions.astype(np.float32),
        ensemble_positions=ensemble_positions,
        ridge_positions=ridge_positions,
        ensemble_edge_bps=ensemble_edge_bps,
        ridge_edge_bps=ridge_edge_bps,
        ensemble_delta=ensemble_delta,
        ridge_delta=ridge_delta,
        ensemble_equity=ensemble_curve,
        ensemble_stress_equity=stress_curve,
        ridge_equity=ridge_curve,
        ensemble_daily_returns=ensemble_daily,
        ensemble_stress_daily_returns=stress_daily,
        ridge_daily_returns=ridge_daily,
    )
    trade_rows = [dict(strategy="fdt_n_tiny_ensemble", **trade) for trade in ensemble_trades]
    trade_rows.extend(dict(strategy="direct_ridge", **trade) for trade in ridge_trades)
    _write_trades(output_dir / "trades.csv", trade_rows)

    result: dict[str, Any] = {
        "schema_version": BACKTEST_SCHEMA_VERSION,
        "study_id": preregistration["study_id"],
        "study": "FDT-N Tiny fresh-holdout BTCUSDT spot profitability",
        "status": "descriptive_research_not_live_track_record",
        "evaluation": {
            "start_utc": evaluation_start.isoformat(),
            "end_exclusive_utc": evaluation_end.isoformat(),
            "decision_examples": int(len(indices)),
            "first_entry": _iso(int(entry_times[0])),
            "last_exit": _iso(int(exit_times[-1])),
        },
        "policy": asdict(frozen),
        "development_calibration": preregistration["development"]["calibration"],
        "model": {
            "ensemble": "equal_weight",
            "checkpoints": [str(path) for path in checkpoint_paths],
            "checkpoint_sha256": [_sha256(path) for path in checkpoint_paths],
            "training_results": [str(path) for path in training_results_paths],
            "training_results_sha256": [_sha256(path) for path in training_results_paths],
            "ridge_artifact": str(ridge_path),
            "ridge_sha256": _sha256(ridge_path),
            "seeds": member_seeds,
            "parameter_counts": parameter_counts,
            "device": device,
        },
        "data_provenance": {
            "preregistration": str(preregistration_path),
            "preregistration_sha256": _sha256(preregistration_path),
            "development_dataset": str(development_dataset_path),
            "development_dataset_sha256": _sha256(development_dataset_path),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "manifest": str(dataset_manifest_path),
            "manifest_sha256": _sha256(dataset_manifest_path),
            "development_manifest": str(development_manifest_path),
            "development_manifest_sha256": _sha256(development_manifest_path),
            "source": "Binance Data Vision spot daily individual trades",
            "source_archives": len(archives),
            "execution_price": "first live-parity retained trade after frozen latency",
        },
        "forecast_metrics": {
            "fdt_n_tiny_ensemble": distribution_metrics(predictions, targets, masks),
            "direct_ridge": point_prediction_metrics(ridge_predictions, targets, masks),
            "persistence": point_prediction_metrics(persistence_predictions, targets, masks),
        },
        "financial_metrics": {
            "fdt_n_tiny_ensemble_primary": ensemble_primary,
            "fdt_n_tiny_ensemble_stress": ensemble_stress,
            "direct_ridge_primary": ridge_primary,
            "buy_and_hold_primary": buy_and_hold,
            "no_trade": {
                "gross_return": 0.0,
                "net_return": 0.0,
                "net_pnl_usd": 0.0,
                "trades": 0,
                "block_boundary_max_drawdown": 0.0,
            },
        },
        "artifacts": {
            "series": "backtest-series.npz",
            "trades": "trades.csv",
            "investor_report": "investor-report.md",
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    _write_investor_report(output_dir / "investor-report.md", result)
    result["artifacts"].update(
        {
            "series_sha256": _sha256(output_dir / "backtest-series.npz"),
            "trades_sha256": _sha256(output_dir / "trades.csv"),
            "investor_report_sha256": _sha256(output_dir / "investor-report.md"),
        }
    )
    (output_dir / "results.json").write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # The stress trade rows are intentionally not duplicated in CSV; they have identical fills and
    # positions, and their cost-adjusted summary plus complete equity curve are preserved above.
    if len(stress_trades) != len(ensemble_trades):
        raise RuntimeError("stress costs unexpectedly changed the frozen trade decisions")
    return result
