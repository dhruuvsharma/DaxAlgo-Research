from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .backtest import (
    DAY_MS,
    FrozenPolicy,
    first_retained_trade_fills,
    policy_signals,
    predict_ensemble,
    resolve_source_archives,
    simulate_blocks,
    validate_dataset_manifest,
)
from .config import EncodingConfig, ModelConfig
from .data import (
    EncodedFootprintSeries,
    build_footprint_bars,
    iter_binance_trades,
    resample_footprint_bars,
    write_dataset_manifest,
)
from .training import chronological_split, distribution_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_screen_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != 1 or protocol.get("status") != (
        "declared_development_screen_before_variant_training"
    ):
        raise ValueError("screen protocol is not a declared schema-v1 development screen")
    candidates = protocol.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("screen protocol has no candidates")
    names = [str(candidate.get("name")) for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("screen candidate names must be unique")
    split = protocol["development"]["split"]
    train_fraction = float(split["train_fraction"])
    validation_fraction = float(split["validation_fraction"])
    reserved = float(split["reserved_screen_fraction"])
    if not math.isclose(train_fraction + validation_fraction + reserved, 1.0):
        raise ValueError("screen split fractions must sum to one")
    return protocol


def _candidate(protocol: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [candidate for candidate in protocol["candidates"] if candidate["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown screen candidate: {name}")
    return matches[0]


def _encoding(protocol: dict[str, Any], candidate: dict[str, Any]) -> EncodingConfig:
    common = protocol["encoding"]
    return EncodingConfig(
        lookback_bars=int(candidate["lookback_bars"]),
        horizon_bars=int(candidate["horizon_bars"]),
        max_rows=int(common["max_rows"]),
        row_size=float(common["row_size"]),
        interval_seconds=int(candidate["interval_seconds"]),
        imbalance_ratio=float(common["imbalance_ratio"]),
    )


def _unique_source_archives(
    protocol: dict[str, Any], manifest_paths: Sequence[Path]
) -> list[Path]:
    expected_hashes = [record["sha256"] for record in protocol["development"]["source_manifests"]]
    actual_hashes = [_sha256(path) for path in manifest_paths]
    if actual_hashes != expected_hashes:
        raise ValueError("source manifest hashes do not match the declared screen protocol")
    by_name: dict[str, tuple[Path, str]] = {}
    for manifest_path in manifest_paths:
        for archive in resolve_source_archives(manifest_path):
            digest = _sha256(archive)
            prior = by_name.get(archive.name)
            if prior is not None and prior[1] != digest:
                raise ValueError(f"duplicate source archive disagrees: {archive.name}")
            by_name[archive.name] = (archive, digest)
    archives = [record[0] for _, record in sorted(by_name.items())]
    expected_count = int(protocol["development"]["expected_unique_archives"])
    if len(archives) != expected_count:
        raise ValueError("unique source archive count does not match the screen protocol")
    return archives


def screen_sample_gate(
    series: EncodedFootprintSeries,
    protocol: dict[str, Any],
) -> dict[str, object]:
    split = protocol["development"]["split"]
    train, validation, reserved = chronological_split(
        series,
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
    )
    expected_windows = max(
        0, len(series) - series.config.lookback_bars - series.config.horizon_bars + 1
    )
    valid_windows = len(series.valid_window_indices())
    continuity = valid_windows / expected_windows if expected_windows else 0.0
    dependence_bars = max(
        series.config.lookback_bars + series.config.horizon_bars,
        math.ceil(4 * 3_600 / series.config.interval_seconds),
    )
    gate = protocol["sample_gate"]
    blocks = {
        "train": len(train) / dependence_bars,
        "validation": len(validation) / dependence_bars,
        "reserved_screen": len(reserved) / dependence_bars,
    }
    passed = (
        blocks["train"] >= int(gate["minimum_train_blocks"])
        and blocks["validation"] >= int(gate["minimum_validation_blocks"])
        and blocks["reserved_screen"] >= int(gate["minimum_reserved_screen_blocks"])
        and continuity >= float(gate["minimum_continuous_window_fraction"])
    )
    return {
        "passed": passed,
        "dependence_bars": dependence_bars,
        "continuous_window_fraction": continuity,
        "valid_windows": valid_windows,
        "expected_windows": expected_windows,
        "examples": {
            "train": len(train),
            "validation": len(validation),
            "reserved_screen": len(reserved),
        },
        "dependence_blocks": blocks,
    }


def prepare_screen_datasets(
    protocol_path: Path,
    source_manifest_paths: Sequence[Path],
    output_dir: Path,
    candidate_names: Sequence[str],
) -> dict[str, object]:
    protocol = load_screen_protocol(protocol_path)
    if not candidate_names:
        raise ValueError("at least one screen candidate must be requested")
    candidates = [_candidate(protocol, name) for name in candidate_names]
    if any(candidate["stage"] == "existing_reference" for candidate in candidates):
        raise ValueError("existing reference candidates are not rebuilt by the screen")
    archives = _unique_source_archives(protocol, source_manifest_paths)
    configurations = [_encoding(protocol, candidate) for candidate in candidates]
    base_interval = min(config.interval_seconds for config in configurations)
    common = protocol["encoding"]
    base_config = EncodingConfig(
        lookback_bars=2,
        horizon_bars=1,
        max_rows=int(common["max_rows"]),
        row_size=float(common["row_size"]),
        interval_seconds=base_interval,
        imbalance_ratio=float(common["imbalance_ratio"]),
    )
    base_bars = build_footprint_bars(
        iter_binance_trades(archives, base_config.quantity_scale), base_config
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    bars_by_interval = {base_interval: base_bars}
    for candidate, config in zip(candidates, configurations, strict=True):
        candidate_dir = output_dir / str(candidate["name"])
        if candidate_dir.exists():
            raise FileExistsError(f"screen dataset directory already exists: {candidate_dir}")
        candidate_dir.mkdir(parents=True)
        bars = bars_by_interval.get(config.interval_seconds)
        if bars is None:
            bars = resample_footprint_bars(
                base_bars,
                config.interval_seconds,
                imbalance_ratio=config.imbalance_ratio,
            )
            bars_by_interval[config.interval_seconds] = bars
        series = EncodedFootprintSeries.from_bars(bars, config)
        gate = screen_sample_gate(series, protocol)
        if not gate["passed"]:
            raise ValueError(
                f"candidate does not pass the declared sample gate: {candidate['name']}"
            )
        dataset_path = candidate_dir / "dataset.npz"
        manifest_path = candidate_dir / "dataset.manifest.json"
        series.save(dataset_path)
        write_dataset_manifest(
            manifest_path,
            archives,
            series,
            "Binance Data Vision spot daily individual trades",
        )
        prepared = {
            "schema_version": 1,
            "candidate": candidate["name"],
            "protocol_sha256": _sha256(protocol_path),
            "dataset": str(dataset_path),
            "dataset_sha256": _sha256(dataset_path),
            "manifest": str(manifest_path),
            "manifest_sha256": _sha256(manifest_path),
            "encoding": config.to_dict(),
            "bars": len(series),
            "sample_gate": gate,
        }
        (candidate_dir / "prepared.json").write_text(
            json.dumps(prepared, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summaries.append(prepared)
    return {"protocol_sha256": _sha256(protocol_path), "candidates": summaries}


def selection_score(selection_result: dict[str, Any]) -> float:
    horizon = str(int(selection_result["encoding_config"]["horizon_bars"]))
    model = selection_result["validation"]["by_horizon_target"][horizon]
    ridge = selection_result["baselines"]["direct_ridge"]["validation"][
        "by_horizon_target"
    ][horizon]
    improvements = []
    for target in ("poc_offset", "delta_fraction"):
        model_mae = float(model[target]["median_mae_normalized"])
        ridge_mae = float(ridge[target]["median_mae_normalized"])
        if ridge_mae <= 0 or not math.isfinite(model_mae + ridge_mae):
            raise ValueError("selection metrics must be finite with a positive ridge MAE")
        improvements.append((ridge_mae - model_mae) / ridge_mae)
    return float(np.mean(improvements))


def execution_coverage_mask(
    decision_times_ms: np.ndarray,
    *,
    interval_ms: int,
    horizon_bars: int,
    execution_latency_ms: int,
    source_end_exclusive_ms: int,
) -> np.ndarray:
    decisions = np.asarray(decision_times_ms, dtype=np.int64)
    if min(interval_ms, horizon_bars, source_end_exclusive_ms) < 1 or execution_latency_ms < 0:
        raise ValueError("execution coverage bounds must be positive")
    return (
        decisions + horizon_bars * interval_ms + execution_latency_ms
        < source_end_exclusive_ms
    )


def _futility_stop_length(
    entry_eligible_times: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    positions: np.ndarray,
    *,
    interval_ms: int,
    horizon_bars: int,
    side_cost_bps: float,
    minimum_days: int,
    minimum_decisions: int,
    minimum_trades: int,
    net_return_floor: float,
) -> tuple[int, dict[str, object] | None]:
    expected_decisions_per_day = DAY_MS // (interval_ms * horizon_bars)
    equity = 1.0
    trades = 0
    complete_days = 0
    side_cost = side_cost_bps / 10_000.0
    day_ids = entry_eligible_times // DAY_MS
    for day_id in np.unique(day_ids):
        day_indices = np.flatnonzero(day_ids == day_id)
        for index in day_indices:
            if positions[index]:
                equity *= (1.0 - side_cost) ** 2 * exit_prices[index] / entry_prices[index]
                trades += 1
        if len(day_indices) == expected_decisions_per_day:
            complete_days += 1
        observed = int(day_indices[-1]) + 1
        if (
            complete_days >= minimum_days
            and observed >= minimum_decisions
            and trades >= minimum_trades
            and equity - 1.0 <= net_return_floor
        ):
            return observed, {
                "triggered": True,
                "complete_utc_days": complete_days,
                "decisions": observed,
                "trades": trades,
                "net_return_at_checkpoint": equity - 1.0,
                "last_complete_day_id": int(day_id),
            }
    return len(entry_eligible_times), None


def _write_trades(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
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


def evaluate_screen_candidate(
    protocol_path: Path,
    candidate_name: str,
    dataset_path: Path,
    dataset_manifest_path: Path,
    checkpoint_path: Path,
    selection_result_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 32,
    device_name: str | None = None,
) -> dict[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"screen output directory already exists: {output_dir}")
    protocol = load_screen_protocol(protocol_path)
    candidate = _candidate(protocol, candidate_name)
    series = EncodedFootprintSeries.load(dataset_path)
    if series.config != _encoding(protocol, candidate):
        raise ValueError("candidate dataset encoding does not match the screen protocol")
    validate_dataset_manifest(dataset_manifest_path, series)
    prepared_path = dataset_path.parent / "prepared.json"
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if (
        prepared.get("candidate") != candidate_name
        or prepared.get("protocol_sha256") != _sha256(protocol_path)
        or prepared.get("dataset_sha256") != _sha256(dataset_path)
        or prepared.get("manifest_sha256") != _sha256(dataset_manifest_path)
    ):
        raise ValueError("prepared dataset record does not match the candidate evidence")
    gate = screen_sample_gate(series, protocol)
    if not gate["passed"]:
        raise ValueError("candidate dataset does not pass the declared sample gate")
    selection = json.loads(selection_result_path.read_text(encoding="utf-8"))
    if selection.get("status") != "development_selection_only_no_screen_access":
        raise ValueError("candidate was not trained in validation-only selection mode")
    if selection.get("encoding_config") != series.config.to_dict():
        raise ValueError("selection result encoding does not match the candidate dataset")
    if selection.get("model_config") != ModelConfig.from_preset("fdt-n-tiny").to_dict():
        raise ValueError("screen candidate is not the declared Tiny architecture")
    recorded_checkpoint = Path(str(selection.get("checkpoint", "")))
    if not recorded_checkpoint.is_file() or _sha256(recorded_checkpoint) != _sha256(
        checkpoint_path
    ):
        raise ValueError("selection result does not bind the supplied checkpoint")
    training = protocol["training"]
    train_config = selection["train_config"]
    split = protocol["development"]["split"]
    expected_training = {
        "seed": int(training["round_one_seeds"][0]),
        "epochs": int(training["epochs"]),
        "batch_size": int(training["batch_size"]),
        "patience": int(training["patience"]),
        "bootstrap_resamples": int(training["bootstrap_resamples"]),
        "train_fraction": float(split["train_fraction"]),
        "validation_fraction": float(split["validation_fraction"]),
        "selection_only": True,
    }
    if any(train_config.get(key) != value for key, value in expected_training.items()):
        raise ValueError("selection training settings do not match the screen protocol")
    _, _, reserved_indices = chronological_split(
        series,
        train_fraction=float(split["train_fraction"]),
        validation_fraction=float(split["validation_fraction"]),
    )
    interval_ms = series.config.interval_seconds * 1_000
    decision_times = series.timestamps_ms[reserved_indices] + interval_ms
    aligned = (decision_times // interval_ms) % series.config.horizon_bars == 0
    decision_times = decision_times[aligned]
    indices = reserved_indices[aligned]
    economics = protocol["economics"]
    execution_complete = execution_coverage_mask(
        decision_times,
        interval_ms=interval_ms,
        horizon_bars=series.config.horizon_bars,
        execution_latency_ms=int(economics["execution_latency_ms"]),
        source_end_exclusive_ms=int(series.timestamps_ms[-1] + interval_ms),
    )
    indices = indices[execution_complete]
    decision_times = decision_times[execution_complete]
    predictions, targets, masks, device, parameter_counts = predict_ensemble(
        series,
        indices,
        [checkpoint_path],
        batch_size=batch_size,
        device_name=device_name,
    )
    policy = FrozenPolicy(
        horizon_bars=series.config.horizon_bars,
        decision_interval_bars=series.config.horizon_bars,
        execution_latency_ms=int(economics["execution_latency_ms"]),
        signal_edge_bps=float(economics["signal_edge_bps"]),
        require_positive_delta=bool(economics["require_positive_delta"]),
        primary_side_cost_bps=float(economics["primary_side_cost_bps"]),
        stress_side_cost_bps=float(economics["stress_side_cost_bps"]),
        initial_capital_usd=float(economics["initial_capital_usd"]),
    )
    policy.validate()
    positions, edge_bps, delta = policy_signals(predictions, series, indices, policy)
    entry_eligible_times = decision_times + policy.execution_latency_ms
    exit_eligible_times = (
        decision_times + policy.horizon_bars * interval_ms + policy.execution_latency_ms
    )
    eligible = np.concatenate((entry_eligible_times, exit_eligible_times))
    order = np.argsort(eligible, kind="stable")
    prices_sorted, times_sorted = first_retained_trade_fills(
        resolve_source_archives(dataset_manifest_path),
        eligible[order],
        quantity_scale=series.config.quantity_scale,
    )
    prices = np.empty_like(prices_sorted)
    times = np.empty_like(times_sorted)
    prices[order] = prices_sorted
    times[order] = times_sorted
    entry_prices, exit_prices = np.split(prices, 2)
    entry_times, exit_times = np.split(times, 2)
    stop, futility = _futility_stop_length(
        entry_eligible_times,
        entry_prices,
        exit_prices,
        positions,
        interval_ms=interval_ms,
        horizon_bars=policy.horizon_bars,
        side_cost_bps=policy.primary_side_cost_bps,
        minimum_days=int(economics["minimum_complete_utc_days_before_futility"]),
        minimum_decisions=int(economics["minimum_decisions_before_futility"]),
        minimum_trades=int(economics["minimum_closed_trades_before_futility"]),
        net_return_floor=float(economics["futility_net_return"]),
    )
    primary, trades, equity, daily = simulate_blocks(
        entry_times[:stop],
        exit_times[:stop],
        entry_prices[:stop],
        exit_prices[:stop],
        positions[:stop],
        initial_capital_usd=policy.initial_capital_usd,
        side_cost_bps=policy.primary_side_cost_bps,
    )
    stress, _, stress_equity, stress_daily = simulate_blocks(
        entry_times[:stop],
        exit_times[:stop],
        entry_prices[:stop],
        exit_prices[:stop],
        positions[:stop],
        initial_capital_usd=policy.initial_capital_usd,
        side_cost_bps=policy.stress_side_cost_bps,
    )
    if futility is not None:
        status = "eliminated_at_frozen_futility_gate"
    elif primary["trades"] < int(economics["minimum_complete_period_trades"]):
        status = "complete_but_insufficient_trade_count"
    elif primary["net_return"] <= 0:
        status = "complete_but_not_profitable"
    else:
        status = "survives_development_economic_screen"
    output_dir.mkdir(parents=True)
    _write_trades(output_dir / "trades.csv", trades)
    np.savez_compressed(
        output_dir / "screen-series.npz",
        decision_indices=indices[:stop],
        decision_times_ms=decision_times[:stop],
        entry_times_ms=entry_times[:stop],
        exit_times_ms=exit_times[:stop],
        entry_prices=entry_prices[:stop],
        exit_prices=exit_prices[:stop],
        positions=positions[:stop],
        edge_bps=edge_bps[:stop],
        delta=delta[:stop],
        equity=equity,
        stress_equity=stress_equity,
        daily_returns=daily,
        stress_daily_returns=stress_daily,
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "study": "FDT-N Tiny multi-timeframe development screen",
        "status": status,
        "candidate": candidate,
        "protocol_sha256": _sha256(protocol_path),
        "dataset_sha256": _sha256(dataset_path),
        "dataset_manifest_sha256": _sha256(dataset_manifest_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "selection_result_sha256": _sha256(selection_result_path),
        "selection_score": selection_score(selection),
        "device": device,
        "parameter_count": parameter_counts[0],
        "policy": asdict(policy),
        "sample_gate": gate,
        "planned_reserved_decisions": int(len(indices)),
        "evaluated_decisions": int(stop),
        "truncated": futility is not None,
        "futility": futility,
        "forecast_metrics": distribution_metrics(
            predictions[:stop], targets[:stop], masks[:stop]
        ),
        "financial_metrics": {"primary": primary, "stress": stress},
        "artifacts": {
            "series": "screen-series.npz",
            "trades": "trades.csv",
        },
    }
    result["artifacts"].update(
        {
            "series_sha256": _sha256(output_dir / "screen-series.npz"),
            "trades_sha256": _sha256(output_dir / "trades.csv"),
        }
    )
    (output_dir / "results.json").write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
