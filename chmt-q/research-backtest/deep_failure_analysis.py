"""Evidence-first failure analysis for the frozen CHMT-Q development study.

The analyzer reads only the already-generated train replay, selected validation,
and untouched development-forward prediction tape. It explicitly rejects the
sealed outer holdout and writes a new, content-addressable diagnostic artifact.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PARTITIONS = (
    "in_sample_train_replay",
    "selected_validation",
    "untouched_development_forward",
)
TARGETS = (
    "total_poc_offset",
    "buy_poc_offset",
    "sell_poc_offset",
    "log_volume_residual",
    "delta_fraction",
)
QUANTILES = (0.1, 0.5, 0.9)
MODEL_PARAMETERS = 2_089_554


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--holdout-integrity-status",
        choices=("pristine", "inadvertently_read_not_used"),
        required=True,
    )
    args = parser.parse_args()

    sources = {
        "predictions": args.predictions.resolve(strict=True),
        "results": args.results.resolve(strict=True),
        "protocol": args.protocol.resolve(strict=True),
        "scaler": args.scaler.resolve(strict=True),
    }
    run_dir = args.run_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)

    results = _read_json(sources["results"])
    protocol = _read_json(sources["protocol"])
    scaler = _read_json(sources["scaler"])
    _validate_holdout_boundary(results, protocol)
    tape = load_prediction_tape(sources["predictions"])
    expected_counts = {name: int(results["row_counts"][name]) for name in PARTITIONS}
    actual_counts = {name: int(tape[name]["actual"].shape[0]) for name in PARTITIONS}
    if actual_counts != expected_counts:
        raise ValueError(
            f"prediction counts differ from verified results: {actual_counts} != {expected_counts}"
        )

    partition_metrics = {
        name: analyze_partition(tape[name]) for name in PARTITIONS
    }
    training = analyze_training(run_dir)
    model_contract = analyze_contract(tape, scaler, results, protocol)
    economic = analyze_economics(results, protocol)
    scoring = analyze_scoring(results)
    findings = build_ranked_findings(
        partition_metrics=partition_metrics,
        training=training,
        contract=model_contract,
        economic=economic,
        scoring=scoring,
        results=results,
    )

    payload = {
        "schema_version": 1,
        "study_kind": "chmt_q_deep_failure_analysis",
        "historical_study_outer_holdout_accessed": False,
        "outer_holdout_used_in_failure_metrics": False,
        "current_audit_holdout_integrity": {
            "status": args.holdout_integrity_status,
            "operationally_pristine": args.holdout_integrity_status == "pristine",
            "disclosure": (
                "An auxiliary schema/count audit physically read projected target fields in "
                "the Jun 25-30 shards. No values were printed, retained, analyzed, or used in "
                "this report. The period must nevertheless be treated as accessed and cannot "
                "serve as a pristine final holdout."
                if args.holdout_integrity_status == "inadvertently_read_not_used"
                else "No current-audit access is known."
            ),
        },
        "scope": {
            "partitions": list(PARTITIONS),
            "excluded_partition": "sealed_outer_holdout_2026-06-25_to_2026-06-30",
            "diagnostic_purpose": "explain weak forecasting and unusable trading economics",
        },
        "source_artifacts": {
            name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in sources.items()
        },
        "run_directory": str(run_dir),
        "row_counts": actual_counts,
        "model_contract": model_contract,
        "partition_metrics": partition_metrics,
        "training_diagnostics": training,
        "scoring_diagnostics": scoring,
        "economic_diagnostics": economic,
        "ranked_findings": findings,
        "revision_blueprint": revision_blueprint(),
    }
    serializable = _clean(payload)
    encoded = (json.dumps(serializable, indent=2, sort_keys=True) + "\n").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(encoded)
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "outer_holdout_used_in_failure_metrics": False,
                "holdout_integrity_status": args.holdout_integrity_status,
            },
            sort_keys=True,
        )
    )
    return 0


def load_prediction_tape(path: Path) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, list[Any]]] = {
        name: defaultdict(list) for name in PARTITIONS
    }
    seen_ids: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            evidence = row.get("evidence_class")
            validate_evidence_class(evidence)
            if row.get("target_names") != list(TARGETS):
                raise ValueError(f"unexpected target order at prediction row {line_number}")
            if tuple(row.get("quantile_levels", ())) != QUANTILES:
                raise ValueError(f"unexpected quantile order at prediction row {line_number}")
            example_id = str(row["example_id"])
            if example_id in seen_ids:
                raise ValueError(f"duplicate example_id in prediction tape: {example_id}")
            seen_ids.add(example_id)
            bucket = buckets[evidence]
            bucket["example_id"].append(example_id)
            bucket["session"].append(str(row["session_id"]))
            bucket["decision_ns"].append(int(row["decision_ns"]))
            bucket["anchor_price"].append(float(row["anchor_price"]))
            bucket["tick_size"].append(float(row["tick_size"]))
            bucket["actual"].append(row["actual_targets_physical"])
            bucket["actual_mask"].append(row["actual_target_mask"])
            bucket["predicted"].append(row["predicted_quantiles_physical"])
            bucket["presence_actual"].append(row["actual_presence_labels"])
            bucket["presence_mask"].append(row["actual_presence_mask"])
            bucket["presence_predicted"].append(
                row["predicted_presence_probabilities"]
            )

    converted: dict[str, dict[str, Any]] = {}
    for name in PARTITIONS:
        bucket = buckets[name]
        if not bucket["actual"]:
            raise ValueError(f"prediction tape has no rows for {name}")
        decision_ns = np.asarray(bucket["decision_ns"], dtype=np.int64)
        if np.any(np.diff(decision_ns) <= 0):
            raise ValueError(f"prediction rows are not strictly chronological in {name}")
        converted[name] = {
            "example_id": tuple(bucket["example_id"]),
            "session": np.asarray(bucket["session"], dtype=object),
            "decision_ns": decision_ns,
            "anchor_price": np.asarray(bucket["anchor_price"], dtype=np.float64),
            "tick_size": np.asarray(bucket["tick_size"], dtype=np.float64),
            "actual": np.asarray(bucket["actual"], dtype=np.float64),
            "actual_mask": np.asarray(bucket["actual_mask"], dtype=np.bool_),
            "predicted": np.asarray(bucket["predicted"], dtype=np.float64),
            "presence_actual": np.asarray(bucket["presence_actual"], dtype=np.float64),
            "presence_mask": np.asarray(bucket["presence_mask"], dtype=np.bool_),
            "presence_predicted": np.asarray(
                bucket["presence_predicted"], dtype=np.float64
            ),
        }
    return converted


def validate_evidence_class(value: Any) -> None:
    if value not in PARTITIONS:
        raise ValueError(
            "prediction tape contains an unauthorized or unknown evidence class: "
            f"{value!r}"
        )


def analyze_partition(data: dict[str, Any]) -> dict[str, Any]:
    actual = data["actual"]
    predicted = data["predicted"]
    mask = data["actual_mask"]
    sessions = data["session"]
    cells: list[list[dict[str, Any]]] = []
    for horizon in range(5):
        horizon_cells: list[dict[str, Any]] = []
        for target in range(5):
            valid = mask[:, horizon, target]
            a = actual[valid, horizon, target]
            q = predicted[valid, horizon, target, :]
            session = sessions[valid]
            metrics = cell_metrics(a, q)
            metrics["dependence"] = dependence_metrics(a, q[:, 1], session)
            if target == 4:
                metrics["bounded_output"] = {
                    "q10_near_minus_one_fraction": float(np.mean(q[:, 0] <= -0.99)),
                    "median_near_bound_fraction": float(np.mean(np.abs(q[:, 1]) >= 0.99)),
                    "q90_near_plus_one_fraction": float(np.mean(q[:, 2] >= 0.99)),
                }
            horizon_cells.append(metrics)
        cells.append(horizon_cells)

    presence = analyze_presence(data)
    stride_seconds = np.diff(data["decision_ns"]).astype(np.float64) / 1e9
    local_stride = stride_seconds[(stride_seconds > 0) & (stride_seconds < 3600)]
    tick_sizes = np.unique(data["tick_size"])
    if tick_sizes.size != 1:
        raise ValueError("partition contains multiple tick sizes")
    return {
        "row_count": int(actual.shape[0]),
        "session_count": int(np.unique(sessions).size),
        "first_decision_ns": int(data["decision_ns"][0]),
        "last_decision_ns": int(data["decision_ns"][-1]),
        "median_local_decision_stride_seconds": float(np.median(local_stride)),
        "tick_size": float(tick_sizes[0]),
        "cells": cells,
        "presence": presence,
        "daily_h5_total_poc": daily_metrics(
            actual[:, 4, 0], predicted[:, 4, 0, 1], sessions
        ),
    }


def cell_metrics(actual: np.ndarray, quantiles: np.ndarray) -> dict[str, Any]:
    if actual.ndim != 1 or quantiles.shape != (actual.size, 3) or actual.size == 0:
        raise ValueError("cell metrics require non-empty [N] targets and [N,3] quantiles")
    median = quantiles[:, 1]
    residual = median - actual
    actual_std = float(np.std(actual))
    prediction_std = float(np.std(median))
    covariance = float(np.mean((median - median.mean()) * (actual - actual.mean())))
    variance = float(np.var(median))
    slope = covariance / variance if variance > 0.0 else None
    intercept = float(actual.mean() - slope * median.mean()) if slope is not None else None
    sign = binary_sign_metrics(actual, median)
    return {
        "count": int(actual.size),
        "actual": distribution(actual),
        "prediction_median": distribution(median),
        "median_bias": float(np.mean(residual)),
        "median_mae": float(np.mean(np.abs(residual))),
        "median_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pearson": safe_pearson(actual, median),
        "spearman": safe_spearman(actual, median),
        "prediction_to_actual_std_ratio": (
            prediction_std / actual_std if actual_std > 0.0 else None
        ),
        "prediction_to_actual_variance_ratio": (
            variance / float(np.var(actual)) if actual_std > 0.0 else None
        ),
        "calibration_actual_on_prediction": {
            "intercept": intercept,
            "slope": slope,
        },
        "mean_interval_width": float(np.mean(quantiles[:, 2] - quantiles[:, 0])),
        "interval_coverage": float(
            np.mean((quantiles[:, 0] <= actual) & (actual <= quantiles[:, 2]))
        ),
        "interval_crosses_zero_fraction": float(
            np.mean((quantiles[:, 0] <= 0.0) & (quantiles[:, 2] >= 0.0))
        ),
        "empirical_cdf_at_q10_q50_q90": [
            float(np.mean(actual <= quantiles[:, index])) for index in range(3)
        ],
        "sign": sign,
    }


def binary_sign_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    actual_positive = actual > 0.0
    predicted_positive = predicted > 0.0
    tp = int(np.sum(actual_positive & predicted_positive))
    tn = int(np.sum(~actual_positive & ~predicted_positive))
    fp = int(np.sum(~actual_positive & predicted_positive))
    fn = int(np.sum(actual_positive & ~predicted_positive))
    positive_rate = float(np.mean(actual_positive))
    tpr = tp / (tp + fn) if tp + fn else None
    tnr = tn / (tn + fp) if tn + fp else None
    balanced = (tpr + tnr) / 2.0 if tpr is not None and tnr is not None else None
    return {
        "accuracy": (tp + tn) / actual.size,
        "balanced_accuracy": balanced,
        "majority_accuracy": max(positive_rate, 1.0 - positive_rate),
        "mcc": matthews_correlation(tp, tn, fp, fn),
        "actual_positive_fraction": positive_rate,
        "predicted_positive_fraction": float(np.mean(predicted_positive)),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def matthews_correlation(tp: int, tn: int, fp: int, fn: int) -> float | None:
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else None


def dependence_metrics(
    actual: np.ndarray, predicted: np.ndarray, sessions: np.ndarray
) -> dict[str, Any]:
    residual = predicted - actual
    return {
        "actual": series_dependence(actual, sessions),
        "prediction": series_dependence(predicted, sessions),
        "residual": series_dependence(residual, sessions),
    }


def series_dependence(values: np.ndarray, sessions: np.ndarray) -> dict[str, Any]:
    autocorrelations = {
        str(lag): within_session_lag_correlation(values, sessions, lag)
        for lag in (1, 5, 60)
    }
    positive_sum = 0.0
    positive_lags = 0
    for lag in range(1, 61):
        correlation = within_session_lag_correlation(values, sessions, lag)
        if correlation is None or correlation <= 0.0:
            break
        positive_sum += correlation
        positive_lags += 1
    effective = values.size / (1.0 + 2.0 * positive_sum)
    return {
        "autocorrelation": autocorrelations,
        "positive_sequence_lags": positive_lags,
        "heuristic_effective_sample_size": float(effective),
        "heuristic_effective_fraction": float(effective / values.size),
        "method": "initial-positive autocorrelation sequence through lag 60; descriptive, not an IID proof",
    }


def within_session_lag_correlation(
    values: np.ndarray, sessions: np.ndarray, lag: int
) -> float | None:
    if values.size <= lag:
        return None
    same_session = sessions[:-lag] == sessions[lag:]
    if not bool(np.any(same_session)):
        return None
    return safe_pearson(values[:-lag][same_session], values[lag:][same_session])


def analyze_presence(data: dict[str, Any]) -> list[list[dict[str, Any]]]:
    actual = data["presence_actual"]
    predicted = data["presence_predicted"]
    mask = data["presence_mask"]
    result: list[list[dict[str, Any]]] = []
    for horizon in range(5):
        row: list[dict[str, Any]] = []
        for side in range(2):
            valid = mask[:, horizon, side]
            y = actual[valid, horizon, side]
            p = predicted[valid, horizon, side]
            row.append(
                {
                    "count": int(y.size),
                    "positive_fraction": float(np.mean(y)),
                    "predicted_probability_mean": float(np.mean(p)),
                    "brier": float(np.mean(np.square(p - y))),
                    "has_both_classes": bool(np.any(y == 0.0) and np.any(y == 1.0)),
                }
            )
        result.append(row)
    return result


def daily_metrics(
    actual: np.ndarray, predicted: np.ndarray, sessions: np.ndarray
) -> list[dict[str, Any]]:
    result = []
    for session in np.unique(sessions):
        selected = sessions == session
        a = actual[selected]
        p = predicted[selected]
        result.append(
            {
                "session": str(session),
                "count": int(a.size),
                "median_bias": float(np.mean(p - a)),
                "median_mae": float(np.mean(np.abs(p - a))),
                "pearson": safe_pearson(a, p),
                "spearman": safe_spearman(a, p),
                "sign_accuracy": float(np.mean((a > 0.0) == (p > 0.0))),
                "actual_std": float(np.std(a)),
                "prediction_std": float(np.std(p)),
            }
        )
    return result


def analyze_training(run_dir: Path) -> dict[str, Any]:
    epoch_paths = sorted((run_dir / "epochs").glob("epoch-*.json"))
    if not epoch_paths:
        raise ValueError("run directory contains no epoch artifacts")
    run_config = _read_json(run_dir / "config.json")["payload"]
    epochs = []
    all_gradients: list[float] = []
    for path in epoch_paths:
        payload = _read_json(path)["payload"]
        validation = payload["validation"]
        windows = payload["windows"]
        gradients = np.asarray([item["gradient_norm"] for item in windows], dtype=float)
        all_gradients.extend(float(value) for value in gradients)
        attention = payload["telemetry"]["attention_logits"]
        max_attention = max(
            max(float(value) for value in item["max_abs_by_head"])
            for item in attention.values()
        )
        epochs.append(
            {
                "epoch": int(payload["epoch"]),
                "optimizer_step": int(payload["optimizer_step"]),
                "primary_metric": float(validation["primary_metric"]),
                "tie_breaker_median_mae": float(
                    validation["tie_breaker_median_mae"]
                ),
                "validation_loss": float(validation["loss"]),
                "validation_pinball": float(validation["pinball"]),
                "validation_median_huber": float(validation["median_huber"]),
                "validation_presence": float(validation["presence"]),
                "mean_training_window_pinball": float(
                    np.mean([item["pinball"] for item in windows])
                ),
                "last_learning_rate": float(windows[-1]["learning_rate"]),
                "last_auxiliary_scale": float(windows[-1]["auxiliary_scale"]),
                "improved": bool(payload["selection"]["improved"]),
                "best_epoch_so_far": int(payload["selection"]["best_epoch"]),
                "validations_without_improvement": int(
                    payload["selection"]["validations_without_improvement"]
                ),
                "gradient_norm_before_clip": {
                    "median": float(np.median(gradients)),
                    "p95": float(np.quantile(gradients, 0.95)),
                    "max": float(np.max(gradients)),
                    "fraction_above_configured_clip_1_0": float(
                        np.mean(gradients > 1.0)
                    ),
                },
                "maximum_attention_logit": max_attention,
                "raw_modality_gate_means_bar_l1_l2": [
                    float(value)
                    for value in payload["telemetry"]["modality_gate_means"]
                ],
            }
        )
    summary = _read_json(run_dir / "summary.json")["result"]
    best_epoch = int(summary["best_epoch"])
    best = next(item for item in epochs if item["epoch"] == best_epoch)
    final = epochs[-1]
    gradient_array = np.asarray(all_gradients, dtype=float)
    steps_per_epoch = int(epochs[0]["optimizer_step"])
    planned_steps = steps_per_epoch * int(run_config["max_epochs"])
    auxiliary_decay_start_step = int(
        round(planned_steps * float(run_config["model"]["auxiliary_decay_start"]))
    )
    return {
        "seed_count": 1,
        "fold_count": 1,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "epochs_completed": len(epochs),
        "stopped_early": bool(summary["stopped_early"]),
        "best_validation_primary_metric": float(summary["best_metric"]),
        "empirical_baseline_validation_primary_metric": float(
            summary["baseline_primary_metric"]
        ),
        "best_validation_improvement_percent": 100.0
        * (
            float(summary["baseline_primary_metric"])
            - float(summary["best_metric"])
        )
        / float(summary["baseline_primary_metric"]),
        "final_minus_best_primary_metric": float(
            final["primary_metric"] - best["primary_metric"]
        ),
        "planned_max_epochs": int(run_config["max_epochs"]),
        "planned_optimizer_steps": planned_steps,
        "completed_optimizer_steps": int(final["optimizer_step"]),
        "completed_schedule_fraction": float(final["optimizer_step"] / planned_steps),
        "auxiliary_decay_start_step": auxiliary_decay_start_step,
        "auxiliary_decay_started": bool(
            final["optimizer_step"] >= auxiliary_decay_start_step
        ),
        "final_learning_rate": float(final["last_learning_rate"]),
        "intended_minimum_learning_rate": float(
            run_config["training"]["minimum_learning_rate"]
        ),
        "configured_gradient_clip_norm": float(
            run_config["training"]["gradient_clip_norm"]
        ),
        "all_recorded_preclip_gradient_norms": {
            "count": int(gradient_array.size),
            "minimum": float(np.min(gradient_array)),
            "median": float(np.median(gradient_array)),
            "p90": float(np.quantile(gradient_array, 0.90)),
            "maximum": float(np.max(gradient_array)),
            "fraction_above_clip": float(
                np.mean(
                    gradient_array
                    > float(run_config["training"]["gradient_clip_norm"])
                )
            ),
        },
        "selection_window_note": "checkpoint selection used two UTC days and one seed",
    }


def analyze_contract(
    tape: dict[str, dict[str, Any]],
    scaler: dict[str, Any],
    results: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    train = tape["in_sample_train_replay"]
    train_rows = int(train["actual"].shape[0])
    train_sessions = int(np.unique(train["session"]).size)
    target_block = scaler["payload"]["inner_scaler"]["payload"]["blocks"][
        "targets"
    ]
    inner_scaler = scaler["payload"]["inner_scaler"]["payload"]
    l1_block = inner_scaler["blocks"]["l1"]
    minimum_scale = float(inner_scaler["minimum_scale"])
    minimum_scale_l1_indices = [
        index
        for index, value in enumerate(l1_block["scales"])
        if math.isclose(float(value), minimum_scale, rel_tol=0.0, abs_tol=0.0)
    ]
    horizon_weights = np.asarray(
        [1.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(3.0), 0.5, 1.0 / math.sqrt(5.0)]
    )
    target_weights = np.asarray([1.0, 1.0, 1.0, 0.5, 0.5])
    h5_total_poc_objective_share = float(
        horizon_weights[4]
        * target_weights[0]
        / (horizon_weights.sum() * target_weights.sum())
    )
    return {
        "parameter_count": MODEL_PARAMETERS,
        "train_rows": train_rows,
        "train_sessions": train_sessions,
        "parameters_per_train_row": MODEL_PARAMETERS / train_rows,
        "parameters_per_train_session": MODEL_PARAMETERS / train_sessions,
        "h5_total_poc_quantile_objective_weight_fraction": h5_total_poc_objective_share,
        "targets": list(TARGETS),
        "contains_direct_return_target": False,
        "contains_executable_bid_ask_return_target": False,
        "contains_trade_action_target": False,
        "protocol_output_contract": protocol["model_output_contract"],
        "bar_lookback_minutes": 64,
        "microstructure_lookback_seconds": 120,
        "adjacent_bar_context_overlap_fraction": 63.0 / 64.0,
        "adjacent_five_minute_label_overlap_fraction": 4.0 / 5.0,
        "training_decision_stride_seconds": 60.0,
        "target_scaler": {
            "centers": target_block["centers"],
            "scales": target_block["scales"],
            "identity_indices": target_block["identity_indices"],
        },
        "input_scaler": {
            "clip_bounds": [float(inner_scaler["clip_min"]), float(inner_scaler["clip_max"])],
            "minimum_scale": minimum_scale,
            "l1_minimum_scale_feature_indices": minimum_scale_l1_indices,
        },
        "outer_holdout_example_count": int(results["outer_holdout_example_count"]),
        "historical_study_outer_holdout_accessed": False,
    }


def analyze_economics(
    results: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    partition = "untouched_development_forward"
    economy = results["economics"][partition]
    policies = economy["policies"]
    long_flat = policies["median_long_flat"]
    long_short = policies["median_long_short"]
    interval = policies["interval_long_flat"]["base"]
    base_cost = float(protocol["cost_scenarios_one_way_bps"]["base"])

    def policy_snapshot(value: dict[str, Any]) -> dict[str, Any]:
        frictionless = value["frictionless"]
        base = value["base"]
        break_even = frictionless["break_even_one_way_bps"]
        return {
            "active_fraction": float(base["active_fraction"]),
            "trade_count": int(base["metrics"]["trade_count"]),
            "frictionless_compounded_return": float(
                frictionless["metrics"]["gross_compounded_return"]
            ),
            "base_cost_compounded_return": float(
                base["metrics"]["net_compounded_return"]
            ),
            "break_even_one_way_bps": float(break_even),
            "gross_round_trip_edge_bps": 2.0 * float(break_even),
            "base_cost_to_break_even_multiple": base_cost / float(break_even),
        }

    candle = results["candle_proxy"][partition][4]
    classification = candle["anchor_to_close_direction_proxy"]
    correlation = candle["predicted_poc_vs_anchor_close_correlation"]
    return {
        "partition": partition,
        "days": 4,
        "base_one_way_cost_bps": base_cost,
        "base_round_trip_cost_bps": 2.0 * base_cost,
        "interval_policy": {
            "active_fraction": float(interval["active_fraction"]),
            "trade_count": int(interval["metrics"]["trade_count"]),
            "reason": "every five-minute q10-q90 total-POC interval crossed zero",
        },
        "median_long_flat": policy_snapshot(long_flat),
        "median_long_short": policy_snapshot(long_short),
        "five_minute_direction_proxy": {
            "accuracy": float(classification["accuracy"]),
            "balanced_accuracy": float(classification["balanced_accuracy"]),
            "mcc": float(classification["mcc"]),
            "pearson": float(correlation["pearson"]),
            "spearman": float(correlation["spearman"]),
        },
        "execution_limitations": protocol["execution_proxy"]["limitations"],
    }


def analyze_scoring(results: dict[str, Any]) -> dict[str, Any]:
    analyzed: dict[str, Any] = {}
    for partition in PARTITIONS:
        forecast = results["forecast"][partition]
        model = forecast["model"]
        baseline = forecast["empirical_marginal"]
        model_pinball = np.asarray(model["normalized"]["pinball"], dtype=float)
        baseline_pinball = np.asarray(
            baseline["normalized"]["pinball"], dtype=float
        )
        model_mae = np.asarray(model["physical"]["median_mae"], dtype=float)
        baseline_mae = np.asarray(
            baseline["physical"]["median_mae"], dtype=float
        )
        analyzed[partition] = {
            "weighted_normalized_pinball": {
                "model": float(model["normalized"]["weighted_primary_metric"]),
                "empirical_baseline": float(
                    baseline["normalized"]["weighted_primary_metric"]
                ),
                "model_improvement_percent": float(
                    forecast["relative_model_improvement_percent"]
                ),
            },
            "cell_pinball_improvement_percent": (
                100.0 * (baseline_pinball - model_pinball) / baseline_pinball
            ).tolist(),
            "cell_median_mae_improvement_percent": (
                100.0 * (baseline_mae - model_mae) / baseline_mae
            ).tolist(),
        }
    return analyzed


def build_ranked_findings(
    *,
    partition_metrics: dict[str, Any],
    training: dict[str, Any],
    contract: dict[str, Any],
    economic: dict[str, Any],
    scoring: dict[str, Any],
    results: dict[str, Any],
) -> list[dict[str, Any]]:
    forward = partition_metrics["untouched_development_forward"]
    h5_poc = forward["cells"][4][0]
    h5_delta = forward["cells"][4][4]
    model_forecast = results["forecast"]["untouched_development_forward"]["model"]
    baseline_forecast = results["forecast"]["untouched_development_forward"][
        "empirical_marginal"
    ]
    forward_scoring = scoring["untouched_development_forward"]
    return [
        {
            "rank": 1,
            "classification": "proven decisive cause",
            "title": "The optimized task is not the trading task",
            "evidence": {
                "direct_return_target": contract["contains_direct_return_target"],
                "target_names": contract["targets"],
                "h5_total_poc_objective_weight_percent": 100.0
                * contract["h5_total_poc_quantile_objective_weight_fraction"],
                "five_minute_poc_to_future_close_spearman": economic[
                    "five_minute_direction_proxy"
                ]["spearman"],
                "five_minute_direction_mcc": economic["five_minute_direction_proxy"][
                    "mcc"
                ],
            },
            "implication": "Lower footprint pinball loss does not imply a forecast of executable return or net PnL.",
        },
        {
            "rank": 2,
            "classification": "proven decisive cause",
            "title": "The observed gross edge is over fifty times too small for modeled costs",
            "evidence": {
                "base_round_trip_cost_bps": economic["base_round_trip_cost_bps"],
                "long_flat_gross_round_trip_edge_bps": economic["median_long_flat"][
                    "gross_round_trip_edge_bps"
                ],
                "cost_to_break_even_multiple": economic["median_long_flat"][
                    "base_cost_to_break_even_multiple"
                ],
                "base_cost_compounded_return": economic["median_long_flat"][
                    "base_cost_compounded_return"
                ],
            },
            "implication": "No threshold tuning can rescue a signal whose measured edge is structurally below fees and slippage.",
        },
        {
            "rank": 3,
            "classification": "proven major cause",
            "title": "Uncertainty is too broad to authorize a trade",
            "evidence": {
                "h5_total_poc_mean_interval_width_ticks": h5_poc[
                    "mean_interval_width"
                ],
                "h5_total_poc_mean_interval_width_usdt": h5_poc[
                    "mean_interval_width"
                ]
                * forward["tick_size"],
                "h5_total_poc_interval_crosses_zero_fraction": h5_poc[
                    "interval_crosses_zero_fraction"
                ],
                "primary_interval_policy_trade_count": economic["interval_policy"][
                    "trade_count"
                ],
            },
            "implication": "The uncertainty-aware primary policy correctly abstains on every decision.",
        },
        {
            "rank": 4,
            "classification": "proven major cause",
            "title": "Conditional median forecasts are weak and strongly shrunk",
            "evidence": {
                "h5_total_poc_pearson": h5_poc["pearson"],
                "h5_total_poc_spearman": h5_poc["spearman"],
                "prediction_to_actual_std_ratio": h5_poc[
                    "prediction_to_actual_std_ratio"
                ],
                "aggregate_weighted_pinball_improvement_percent": forward_scoring[
                    "weighted_normalized_pinball"
                ]["model_improvement_percent"],
                "h5_total_poc_median_mae_improvement_percent": forward_scoring[
                    "cell_median_mae_improvement_percent"
                ][4][0],
                "h5_delta_median_mae_improvement_percent": forward_scoring[
                    "cell_median_mae_improvement_percent"
                ][4][4],
                "model_h5_poc_median_mae": model_forecast["physical"][
                    "median_mae"
                ][4][0],
                "baseline_h5_poc_median_mae": baseline_forecast["physical"][
                    "median_mae"
                ][4][0],
            },
            "implication": "Aggregate pinball improvement is not the same as strong point-forecast or directional information.",
        },
        {
            "rank": 5,
            "classification": "proven design limitation",
            "title": "The nominal sample count materially overstates independent information",
            "evidence": {
                "train_rows": contract["train_rows"],
                "train_sessions": contract["train_sessions"],
                "parameters": contract["parameter_count"],
                "parameters_per_train_row": contract["parameters_per_train_row"],
                "adjacent_bar_context_overlap_fraction": contract[
                    "adjacent_bar_context_overlap_fraction"
                ],
                "h5_actual_heuristic_effective_fraction": h5_poc["dependence"][
                    "actual"
                ]["heuristic_effective_fraction"],
            },
            "implication": "A 2.09M-parameter model was fit on 18 highly overlapping sessions, not 25,758 independent markets.",
        },
        {
            "rank": 6,
            "classification": "proven validation limitation",
            "title": "Model selection and evaluation are too narrow for a stable claim",
            "evidence": {
                "seed_count": training["seed_count"],
                "fold_count": training["fold_count"],
                "selection_days": 2,
                "development_forward_days": economic["days"],
                "best_epoch": training["best_epoch"],
            },
            "implication": "One seed, one fold, two selection days, and four forward days cannot establish regime robustness.",
        },
        {
            "rank": 7,
            "classification": "proven wasted objective capacity",
            "title": "Presence labels are degenerate on liquid BTC minutes",
            "evidence": {
                "forward_h1_buy_positive_fraction": forward["presence"][0][0][
                    "positive_fraction"
                ],
                "forward_h1_sell_positive_fraction": forward["presence"][0][1][
                    "positive_fraction"
                ],
                "has_both_classes": forward["presence"][0][0]["has_both_classes"],
            },
            "implication": "The auxiliary classifier learns an almost constant one and adds no useful selectivity.",
        },
        {
            "rank": 8,
            "classification": "proven output pathology; causal impact not isolated",
            "title": "Delta quantile bounds saturate",
            "evidence": h5_delta["bounded_output"],
            "implication": "Tanh after ordered-width construction pins the lower tail at -1 and pushes the upper tail near +1, producing overwide, weakly informative delta intervals.",
        },
        {
            "rank": 9,
            "classification": "plausible optimization contributor; not isolated",
            "title": "All recorded updates operated above the gradient clip threshold",
            "evidence": {
                "recorded_update_count": training[
                    "all_recorded_preclip_gradient_norms"
                ]["count"],
                "median_preclip_gradient_norm": training[
                    "all_recorded_preclip_gradient_norms"
                ]["median"],
                "fraction_above_clip": training[
                    "all_recorded_preclip_gradient_norms"
                ]["fraction_above_clip"],
                "configured_clip_norm": training["configured_gradient_clip_norm"],
                "auxiliary_decay_started": training["auxiliary_decay_started"],
            },
            "implication": "Persistent clipping and an auxiliary schedule that never decayed may distort multi-target learning, but controlled ablations are needed before calling them causal.",
        },
        {
            "rank": 10,
            "classification": "proven representation loss; metric impact not isolated",
            "title": "Sparse L1 features collapse at the robust-scaler clip boundary",
            "evidence": {
                "l1_minimum_scale_feature_indices": contract["input_scaler"][
                    "l1_minimum_scale_feature_indices"
                ],
                "minimum_scale": contract["input_scaler"]["minimum_scale"],
                "clip_bounds": contract["input_scaler"]["clip_bounds"],
            },
            "implication": "For zero-MAD discrete changes, any nonzero value is mapped to a boundary, preserving mostly sign rather than magnitude.",
        },
        {
            "rank": 11,
            "classification": "proven benchmark limitation",
            "title": "The transformer was not tested against strong simple models",
            "evidence": {
                "registered_baselines": ["empirical_marginal", "zero_residual"],
                "missing_baseline_classes": [
                    "seasonal_or_ewma",
                    "ridge_or_logistic",
                    "boosted_tree",
                    "tcn_or_deeplob_like",
                ],
            },
            "implication": "A small gain over a marginal distribution does not establish that a 2.09M-parameter transformer is necessary.",
        },
        {
            "rank": 12,
            "classification": "proven research-fidelity limitation",
            "title": "The replay cannot establish live causal or executable performance",
            "evidence": {
                "execution_limitations": economic["execution_limitations"],
                "direct_executable_return_target": contract[
                    "contains_executable_bid_ask_return_target"
                ],
            },
            "implication": "Minute trade prices and fixed cost haircuts cannot validate queue, spread, impact, financing, or live receive-time causality.",
        },
    ]


def revision_blueprint() -> dict[str, Any]:
    return {
        "principle": "Treat CHMT-Q v1 as an auxiliary footprint encoder, not as a trading model.",
        "data": [
            "Acquire at least 3-6 months of pre-June BTC-USDT history with executable bid/ask, exchange sequence, checksum, and independent local receive timestamps.",
            "Use effective-dated instrument metadata and explicit outage/gap flags.",
            "Sample primary decisions every five minutes or use overlap-aware weighting; purge and embargo every validation boundary.",
            "Encode sparse discrete changes with an explicit zero indicator plus signed-log or nonzero-magnitude scaling; require under 1% feature clipping.",
        ],
        "targets": [
            "Primary: future executable long and short log return after entry latency and measured spread/fee/slippage.",
            "Predict return q10/q50/q90 plus realized volatility or downside risk; define flat as the default action.",
            "Keep footprint POC, volume, and delta only as lower-weight auxiliary representation targets.",
        ],
        "models": [
            "First beat cash, empirical, linear/ridge/logistic, gradient-boosted trees, and a compact TCN/DeepLOB-style baseline.",
            "Only then test a 0.2-0.5M parameter transformer with modality ablations for bars, L1, and L2.",
            "Replace bounded delta quantile construction with a non-saturating distributional parameterization or model transformed delta with calibrated inverse mapping.",
        ],
        "training": [
            "Use day-blocked walk-forward folds, at least three seeds, a calibration split, and early stopping on at least ten selection days.",
            "Optimize a return-distribution score plus a small auxiliary footprint loss; never optimize backtest PnL directly on the validation set.",
            "Calibrate return intervals out of sample and tune a single cost-aware abstention threshold before the final holdout.",
            "Re-budget learning-rate and auxiliary-loss schedules for the actual stopping horizon; target a post-warmup clip rate below 10% and ablate the auxiliary loss.",
        ],
        "decision_rule": "Long only when the lower net-return quantile exceeds zero; short only when the upper net-return quantile is below zero; otherwise flat. Cap size by forecast volatility and uncertainty.",
        "proposed_go_no_go_gates": [
            "Forecast score improves at least 5% over the strongest simple baseline across at least three folds and three seeds.",
            "Directional MCC exceeds 0.10 with stable sign in every development fold.",
            "Estimated gross edge is at least 3x frozen all-in round-trip cost before capital allocation is considered.",
            "Base- and stress-cost net return are positive and the day-block bootstrap lower confidence bound is above zero.",
            "No sealed outer holdout access until architecture, thresholds, costs, and all gates are frozen.",
            "Because Jun 25-30 was inadvertently read during this audit, replace it with a newly sealed later period for any final confirmation.",
        ],
    }


def distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size != right.size or left.size < 2:
        return None
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = math.sqrt(
        float(np.dot(left_centered, left_centered))
        * float(np.dot(right_centered, right_centered))
    )
    return float(np.dot(left_centered, right_centered) / denominator) if denominator else None


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    return safe_pearson(average_ranks(left), average_ranks(right))


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def _validate_holdout_boundary(
    results: dict[str, Any], protocol: dict[str, Any]
) -> None:
    if results.get("outer_holdout_accessed") is not False:
        raise ValueError("verified results do not preserve the sealed outer holdout")
    sealed = protocol.get("sealed_outer_holdout", {})
    if sealed.get("access_authorized") is not False or sealed.get("will_access") is not False:
        raise ValueError("study protocol does not seal the outer holdout")
    if set(results.get("row_counts", {})) != set(PARTITIONS):
        raise ValueError("results contain an unexpected evaluated partition")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
