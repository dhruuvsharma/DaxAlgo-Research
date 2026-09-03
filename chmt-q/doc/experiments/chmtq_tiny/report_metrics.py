"""Compute aggregate diagnostics for the CHMT-Q Tiny technical PDF.

No prediction tape is written. The output contains aggregate metrics only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from run import (
    HORIZON_WEIGHTS,
    QUANTILE_LEVELS,
    TARGET_WEIGHTS,
    ArrowCorpus,
    Partition,
    RobustScaler,
    make_partitions,
    model_forward,
    move_and_scale,
    sha256_file,
)
from model import ChmtQTiny, TinyConfig, ordered_quantiles, parameter_count
from scipy import stats
import torch


TARGET_NAMES = (
    "total_poc_offset",
    "buy_poc_offset",
    "sell_poc_offset",
    "log_volume_residual",
    "delta_fraction",
)
DATE_TIME_PATTERN = re.compile(r"(\d{8})T(\d{6})Z")


def parse_timestamp(example_id: str) -> datetime:
    match = DATE_TIME_PATTERN.search(example_id)
    if not match:
        raise ValueError(f"unrecognized example ID: {example_id}")
    return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").replace(
        tzinfo=timezone.utc
    )


def collect_targets(
    corpus: ArrowCorpus, partition: Partition, scaler: RobustScaler
) -> dict[str, Any]:
    ids: list[str] = []
    targets: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for batch in corpus.batches(
        partition, epoch=0, seed=17, include_features=False, shuffle=False
    ):
        ids.extend(batch["example_id"])
        y = batch["targets"]
        mask = batch["target_mask"]
        y_normalized = scaler.transform("targets", y, mask)
        targets.append(y.numpy())
        normalized.append(y_normalized.numpy())
        masks.append(mask.numpy().astype(bool))
    return {
        "ids": ids,
        "timestamps": [parse_timestamp(example_id) for example_id in ids],
        "physical": np.concatenate(targets),
        "normalized": np.concatenate(normalized),
        "mask": np.concatenate(masks),
    }


@torch.inference_mode()
def collect_model_predictions(
    corpus: ArrowCorpus,
    partition: Partition,
    scaler: RobustScaler,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ChmtQTiny(TinyConfig(**checkpoint["config"])).to(device).eval()
    model.load_state_dict(checkpoint["model_state"])
    ids: list[str] = []
    predictions: list[np.ndarray] = []
    raw_values: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    memory_counts: list[np.ndarray] = []
    first_input: dict[str, Any] | None = None
    for batch in corpus.batches(
        partition, epoch=0, seed=17, include_features=True, shuffle=False
    ):
        if first_input is None:
            first_input = {
                "input_shapes": {
                    "bars": list(batch["bars"].shape[1:]),
                    "l1": list(batch["l1"].shape[1:]),
                    "l2": list(batch["l2"].shape[1:]),
                    "quality": list(batch["quality"].shape[1:]),
                },
                "valid_input_counts": {
                    "bars": int(batch["bars_mask"][0].sum()),
                    "l1": int(batch["l1_mask"][0].sum()),
                    "l2_snapshots": int(batch["l2_snapshot_mask"][0].sum()),
                    "l2_levels": int(batch["l2_level_mask"][0].sum()),
                },
            }
        ids.extend(batch["example_id"])
        scaled = move_and_scale(batch, scaler, device)
        raw, _, diagnostics = model_forward(model, scaled)
        raw_values.append(raw.cpu().numpy())
        predictions.append(ordered_quantiles(raw).cpu().numpy())
        gates.append(diagnostics["gates"].cpu().numpy())
        memory_counts.append(diagnostics["memory_mask"].sum(dim=-1).cpu().numpy())
    return {
        "ids": ids,
        "prediction": np.concatenate(predictions),
        "raw": np.concatenate(raw_values),
        "gates": np.concatenate(gates),
        "memory_counts": np.concatenate(memory_counts),
        "parameters": parameter_count(model),
        "selected_epoch": int(checkpoint["epoch"]),
        "first_input": first_input,
    }


def fit_quantiles(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros((5, 5, 3), dtype=np.float32)
    for horizon in range(5):
        for target in range(5):
            values = y[:, horizon, target][mask[:, horizon, target]]
            if not values.size:
                raise ValueError(f"no valid values for horizon={horizon} target={target}")
            output[horizon, target] = np.quantile(values, QUANTILE_LEVELS)
    return output


def empirical_prediction(quantiles: np.ndarray, rows: int) -> np.ndarray:
    return np.broadcast_to(quantiles, (rows, 5, 5, 3)).copy()


def hour_of_day_prediction(
    train: dict[str, Any], test: dict[str, Any], fallback: np.ndarray
) -> np.ndarray:
    by_hour: dict[int, np.ndarray] = {}
    train_hours = np.array([timestamp.hour for timestamp in train["timestamps"]])
    for hour in range(24):
        selected = train_hours == hour
        by_hour[hour] = (
            fit_quantiles(train["normalized"][selected], train["mask"][selected])
            if selected.any()
            else fallback
        )
    return np.stack([by_hour[timestamp.hour] for timestamp in test["timestamps"]])


def trailing_complete_days_prediction(
    history: dict[str, Any], test: dict[str, Any], fallback: np.ndarray, days: int = 3
) -> np.ndarray:
    history_dates = np.array([timestamp.date() for timestamp in history["timestamps"]])
    by_day: dict[Any, np.ndarray] = {}
    for day in sorted({timestamp.date() for timestamp in test["timestamps"]}):
        permitted = {day - timedelta(days=offset) for offset in range(1, days + 1)}
        selected = np.array([value in permitted for value in history_dates])
        by_day[day] = (
            fit_quantiles(history["normalized"][selected], history["mask"][selected])
            if selected.any()
            else fallback
        )
    return np.stack([by_day[timestamp.date()] for timestamp in test["timestamps"]])


def residual_persistence_prediction(
    train: dict[str, Any], test: dict[str, Any], fallback: np.ndarray
) -> np.ndarray:
    residuals: list[np.ndarray] = []
    residual_masks: list[np.ndarray] = []
    for index in range(1, len(train["ids"])):
        current = train["timestamps"][index]
        previous = train["timestamps"][index - 1]
        if current - previous != timedelta(minutes=1) or current.date() != previous.date():
            continue
        anchor = train["normalized"][index - 1, 0, :]
        anchor_mask = train["mask"][index - 1, 0, :]
        residuals.append(train["normalized"][index] - anchor[None, :])
        residual_masks.append(train["mask"][index] & anchor_mask[None, :])
    residual_quantiles = fit_quantiles(
        np.stack(residuals), np.stack(residual_masks)
    )
    prediction = empirical_prediction(fallback, len(test["ids"]))
    for index in range(1, len(test["ids"])):
        current = test["timestamps"][index]
        previous = test["timestamps"][index - 1]
        if current - previous != timedelta(minutes=1) or current.date() != previous.date():
            continue
        anchor = test["normalized"][index - 1, 0, :]
        prediction[index] = anchor[None, :, None] + residual_quantiles
        prediction[index, :, 4, :] = np.clip(prediction[index, :, 4, :], -1.0, 1.0)
    return prediction


def safe_correlation(x: np.ndarray, y: np.ndarray, kind: str) -> float | None:
    if (
        x.size < 3
        or np.allclose(x, x[0], rtol=0, atol=1e-12)
        or np.allclose(y, y[0], rtol=0, atol=1e-12)
    ):
        return None
    result = stats.pearsonr(x, y) if kind == "pearson" else stats.spearmanr(x, y)
    value = float(result.statistic)
    return value if math.isfinite(value) else None


def binary_sign_metrics(prediction: np.ndarray, realized: np.ndarray) -> dict[str, float | None]:
    pred = prediction > 0
    truth = realized > 0
    tp = int(np.sum(pred & truth))
    tn = int(np.sum(~pred & ~truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    total = tp + tn + fp + fn
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (
        (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None
    )
    return {
        "accuracy": (tp + tn) / total if total else None,
        "balanced_accuracy": balanced,
        "mcc": (tp * tn - fp * fn) / denominator if denominator else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "zero_rule": "positive versus non-positive",
    }


def score_only(prediction: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    error = y[..., None] - prediction
    pinball = np.maximum(QUANTILE_LEVELS * error, (QUANTILE_LEVELS - 1.0) * error)
    weights = (
        mask[..., None]
        * HORIZON_WEIGHTS[None, :, None, None]
        * TARGET_WEIGHTS[None, None, :, None]
    )
    return float((pinball * weights).sum() / (3.0 * weights.sum()))


def prediction_metrics(
    prediction: np.ndarray,
    y_normalized: np.ndarray,
    y_physical: np.ndarray,
    mask: np.ndarray,
    scaler: RobustScaler,
) -> dict[str, Any]:
    error = y_normalized[..., None] - prediction
    pinball = np.maximum(QUANTILE_LEVELS * error, (QUANTILE_LEVELS - 1.0) * error)
    valid = mask.sum(axis=0)
    per_cell = np.divide(
        (pinball * mask[..., None]).sum(axis=(0, 3)),
        valid * 3,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )
    weighted = score_only(prediction, y_normalized, mask)
    macro = float(np.nanmean(per_cell))
    poc_weights = mask[:, :, 0, None] * HORIZON_WEIGHTS[None, :, None]
    poc_pinball = pinball[:, :, 0, :]
    poc_score = float((poc_pinball * poc_weights).sum() / (3.0 * poc_weights.sum()))

    prediction_physical = scaler.inverse_targets(prediction)
    median = prediction_physical[..., 1]
    median_error = median - y_physical
    mae = np.divide(
        (np.abs(median_error) * mask).sum(axis=0),
        valid,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )
    rmse = np.sqrt(
        np.divide(
            (np.square(median_error) * mask).sum(axis=0),
            valid,
            out=np.full((5, 5), np.nan),
            where=valid > 0,
        )
    )
    bias = np.divide(
        (median_error * mask).sum(axis=0),
        valid,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )

    lower = prediction_physical[..., 0]
    upper = prediction_physical[..., 2]
    covered = (y_physical >= lower) & (y_physical <= upper)
    coverage = np.divide(
        (covered * mask).sum(axis=0),
        valid,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )
    width_physical = upper - lower
    mean_width_physical = np.divide(
        (width_physical * mask).sum(axis=0),
        valid,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )
    width_normalized = prediction[..., 2] - prediction[..., 0]
    mean_width_normalized = np.divide(
        (width_normalized * mask).sum(axis=0),
        valid,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )
    interval_score = width_normalized.copy()
    interval_score += 10.0 * (prediction[..., 0] - y_normalized) * (
        y_normalized < prediction[..., 0]
    )
    interval_score += 10.0 * (y_normalized - prediction[..., 2]) * (
        y_normalized > prediction[..., 2]
    )
    mean_interval_score = np.divide(
        (interval_score * mask).sum(axis=0),
        valid,
        out=np.full((5, 5), np.nan),
        where=valid > 0,
    )

    hit_rates = np.zeros((5, 5, 3), dtype=np.float64)
    pearson = np.full((5, 5), np.nan)
    spearman = np.full((5, 5), np.nan)
    dispersion = np.full((5, 5), np.nan)
    sign_mcc = np.full((5, 5), np.nan)
    sign_balanced = np.full((5, 5), np.nan)
    for horizon in range(5):
        for target in range(5):
            selected = mask[:, horizon, target]
            actual = y_physical[selected, horizon, target]
            predicted = median[selected, horizon, target]
            for quantile in range(3):
                hit_rates[horizon, target, quantile] = np.mean(
                    actual <= prediction_physical[selected, horizon, target, quantile]
                )
            pearson[horizon, target] = safe_correlation(predicted, actual, "pearson") or 0.0
            spearman[horizon, target] = safe_correlation(predicted, actual, "spearman") or 0.0
            realized_std = float(np.std(actual))
            dispersion[horizon, target] = (
                float(np.std(predicted)) / realized_std if realized_std else np.nan
            )
            sign = binary_sign_metrics(predicted, actual)
            sign_mcc[horizon, target] = sign["mcc"] or 0.0
            sign_balanced[horizon, target] = sign["balanced_accuracy"] or 0.0

    h5_selected = mask[:, 4, 0]
    h5_actual = y_physical[h5_selected, 4, 0]
    h5_predicted = median[h5_selected, 4, 0]
    h5_sign = binary_sign_metrics(h5_predicted, h5_actual)
    return {
        "weighted_normalized_pinball": weighted,
        "macro_cell_pinball": macro,
        "weighted_total_poc_pinball": poc_score,
        "per_cell_pinball": per_cell.tolist(),
        "median_mae_physical": mae.tolist(),
        "median_rmse_physical": rmse.tolist(),
        "median_bias_physical": bias.tolist(),
        "coverage_by_cell": coverage.tolist(),
        "coverage_overall": float((covered * mask).sum() / mask.sum()),
        "coverage_error_overall": float((covered * mask).sum() / mask.sum() - 0.8),
        "mean_width_physical_by_cell": mean_width_physical.tolist(),
        "mean_width_normalized_by_cell": mean_width_normalized.tolist(),
        "interval_score_normalized_by_cell": mean_interval_score.tolist(),
        "quantile_hit_rate_by_cell": hit_rates.tolist(),
        "pearson_by_cell": pearson.tolist(),
        "spearman_by_cell": spearman.tolist(),
        "median_dispersion_ratio_by_cell": dispersion.tolist(),
        "sign_mcc_by_cell": sign_mcc.tolist(),
        "sign_balanced_accuracy_by_cell": sign_balanced.tolist(),
        "h5_total_poc": {
            "mae_ticks": float(np.mean(np.abs(h5_predicted - h5_actual))),
            "rmse_ticks": float(np.sqrt(np.mean((h5_predicted - h5_actual) ** 2))),
            "bias_ticks": float(np.mean(h5_predicted - h5_actual)),
            "pearson": safe_correlation(h5_predicted, h5_actual, "pearson"),
            "spearman": safe_correlation(h5_predicted, h5_actual, "spearman"),
            "dispersion_ratio": float(np.std(h5_predicted) / np.std(h5_actual)),
            "sign": h5_sign,
        },
        "valid_labels": int(mask.sum()),
    }


def daily_scores(
    prediction: np.ndarray,
    baseline: np.ndarray,
    test: dict[str, Any],
) -> list[dict[str, Any]]:
    dates = np.array([timestamp.date() for timestamp in test["timestamps"]])
    output: list[dict[str, Any]] = []
    for day in sorted(set(dates)):
        selected = dates == day
        model_score = score_only(
            prediction[selected], test["normalized"][selected], test["mask"][selected]
        )
        baseline_score = score_only(
            baseline[selected], test["normalized"][selected], test["mask"][selected]
        )
        output.append(
            {
                "date": day.isoformat(),
                "model": model_score,
                "baseline": baseline_score,
                "baseline_minus_model": baseline_score - model_score,
                "relative_improvement_percent": 100.0
                * (baseline_score - model_score)
                / baseline_score,
            }
        )
    return output


def phase_sensitivity(
    prediction: np.ndarray, baseline: np.ndarray, test: dict[str, Any]
) -> dict[str, Any]:
    minute = np.array(
        [timestamp.hour * 60 + timestamp.minute for timestamp in test["timestamps"]]
    )
    phases: list[dict[str, Any]] = []
    for phase in range(64):
        selected = minute % 64 == phase
        if not selected.any():
            continue
        model_score = score_only(
            prediction[selected], test["normalized"][selected], test["mask"][selected]
        )
        baseline_score = score_only(
            baseline[selected], test["normalized"][selected], test["mask"][selected]
        )
        phases.append(
            {
                "phase": phase,
                "rows": int(selected.sum()),
                "baseline_minus_model": baseline_score - model_score,
                "relative_improvement_percent": 100.0
                * (baseline_score - model_score)
                / baseline_score,
            }
        )
    differences = np.array([item["baseline_minus_model"] for item in phases])
    relative = np.array([item["relative_improvement_percent"] for item in phases])
    return {
        "definition": "one sample every 64 minutes for each of 64 clock phases",
        "favorable_phases": int(np.sum(differences > 0)),
        "phase_count": len(phases),
        "difference_min": float(differences.min()),
        "difference_median": float(np.median(differences)),
        "difference_max": float(differences.max()),
        "relative_improvement_min_percent": float(relative.min()),
        "relative_improvement_median_percent": float(np.median(relative)),
        "relative_improvement_max_percent": float(relative.max()),
        "phases": phases,
    }


def finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-v1"),
    )
    parser.add_argument(
        "--study",
        type=Path,
        default=Path(r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1"),
    )
    parser.add_argument(
        "--run",
        type=Path,
        default=Path(
            r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-tiny-retrospective-v2\seed-17"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the Tiny prediction tape")
    device = torch.device("cuda:0")
    split_path = args.study / "split.json"
    scaler_path = args.study / "fold-01-scaler.json"
    checkpoint_path = args.run / "best-model.pt"
    result_path = args.run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    train_partition, validation_partition, test_partition, split_hash, _ = make_partitions(
        split_path
    )
    history_partition = Partition(
        "history-through-2026-06-24",
        lambda example_id: "20260601" <= parse_timestamp(example_id).strftime("%Y%m%d") <= "20260624",
        "okx-btc-usdt-20260601T000000Z",
        "okx-btc-usdt-20260624T999999Z",
        34344,
    )
    target_corpus = ArrowCorpus(args.dataset, 256)
    scaler_cpu = RobustScaler(scaler_path, torch.device("cpu"))
    train = collect_targets(target_corpus, train_partition, scaler_cpu)
    validation = collect_targets(target_corpus, validation_partition, scaler_cpu)
    test = collect_targets(target_corpus, test_partition, scaler_cpu)
    history = collect_targets(target_corpus, history_partition, scaler_cpu)

    model_corpus = ArrowCorpus(args.dataset, 32)
    scaler_cuda = RobustScaler(scaler_path, device)
    model = collect_model_predictions(
        model_corpus, test_partition, scaler_cuda, checkpoint_path, device
    )
    if model["ids"] != test["ids"]:
        raise ValueError("model prediction IDs do not match target IDs")
    if not np.isfinite(model["prediction"]).all() or not np.isfinite(model["raw"]).all():
        raise ValueError("nonfinite model output")

    global_quantiles = fit_quantiles(train["normalized"], train["mask"])
    predictions = {
        "chmt_q_tiny": model["prediction"],
        "empirical_marginal_train_only": empirical_prediction(
            global_quantiles, len(test["ids"])
        ),
        "zero_degenerate": np.zeros_like(model["prediction"]),
        "utc_hour_train_only": hour_of_day_prediction(train, test, global_quantiles),
        "trailing_three_complete_days": trailing_complete_days_prediction(
            history, test, global_quantiles
        ),
        "causal_residual_persistence": residual_persistence_prediction(
            train, test, global_quantiles
        ),
    }
    benchmark_metrics = {
        name: prediction_metrics(
            prediction,
            test["normalized"],
            test["physical"],
            test["mask"],
            scaler_cpu,
        )
        for name, prediction in predictions.items()
    }

    primary_baseline = predictions["empirical_marginal_train_only"]
    model_prediction = predictions["chmt_q_tiny"]
    tiny_cells = np.array(benchmark_metrics["chmt_q_tiny"]["per_cell_pinball"])
    baseline_cells = np.array(
        benchmark_metrics["empirical_marginal_train_only"]["per_cell_pinball"]
    )
    cell_skill = 100.0 * (baseline_cells - tiny_cells) / baseline_cells
    daily = daily_scores(model_prediction, primary_baseline, test)
    phase = phase_sensitivity(model_prediction, primary_baseline, test)
    gates = model["gates"]

    lower_delta = model_prediction[:, :, 4, 0]
    upper_delta = model_prediction[:, :, 4, 2]
    report = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "post hoc retrospective technical diagnostics",
        "claim_boundary": (
            "Already studied June development evidence; no independent test, execution, "
            "profitability, deployment, or promotion claim."
        ),
        "partitions": {
            "train": {"examples": len(train["ids"]), "dates": "2026-06-01--18"},
            "validation": {
                "examples": len(validation["ids"]),
                "dates": "2026-06-19--20",
            },
            "development_forward": {
                "examples": len(test["ids"]),
                "dates": "2026-06-21--24",
            },
        },
        "model": {
            "name": "CHMT-Q Tiny",
            "seed": 17,
            "parameters": model["parameters"],
            "selected_epoch": model["selected_epoch"],
            "reference_parameters": 2_089_554,
            "parameter_reduction_percent": 100.0
            * (1.0 - model["parameters"] / 2_089_554),
            "raw_output_shape": [5, 17],
            "presence_head": "inactive and unvalidated",
        },
        "benchmark_definitions": {
            "empirical_marginal_train_only": "Registered train-only q10/q50/q90 for each horizon-target cell.",
            "zero_degenerate": "All normalized quantiles fixed at zero; a degenerate no-distribution location check.",
            "utc_hour_train_only": "Train-only empirical cell quantiles conditioned on decision UTC hour.",
            "trailing_three_complete_days": "For each forward UTC day, empirical quantiles from the three prior fully matured days.",
            "causal_residual_persistence": "Prior fully matured one-minute target plus train-fitted residual quantiles.",
            "status": "Only the empirical marginal was registered; all other context baselines are post hoc.",
        },
        "registered_result_reconciliation": {
            "validation_model": result["validation"]["model"][
                "weighted_normalized_pinball"
            ],
            "validation_baseline": result["validation"][
                "empirical_marginal_baseline"
            ]["weighted_normalized_pinball"],
            "development_model": result["development_forward_test"]["model"][
                "weighted_normalized_pinball"
            ],
            "development_baseline": result["development_forward_test"][
                "empirical_marginal_baseline"
            ]["weighted_normalized_pinball"],
            "development_improvement_percent": result["development_forward_test"][
                "relative_pinball_improvement_percent"
            ],
        },
        "benchmarks": benchmark_metrics,
        "cell_skill_vs_empirical_percent": cell_skill.tolist(),
        "cell_wins_vs_empirical": int(np.sum(tiny_cells < baseline_cells)),
        "target_mean_skill_percent": np.mean(cell_skill, axis=0).tolist(),
        "horizon_mean_skill_percent": np.mean(cell_skill, axis=1).tolist(),
        "daily_stability": daily,
        "overlap_sensitivity": phase,
        "model_diagnostics": {
            "quantile_crossings": int(
                np.sum(model_prediction[..., 0] > model_prediction[..., 1])
                + np.sum(model_prediction[..., 1] > model_prediction[..., 2])
            ),
            "nonfinite_outputs": int(np.sum(~np.isfinite(model_prediction))),
            "delta_lower_saturation_rate": float(np.mean(lower_delta <= -0.999)),
            "delta_upper_saturation_rate": float(np.mean(upper_delta >= 0.999)),
            "gate_mean": gates.mean(axis=0).tolist(),
            "gate_std": gates.std(axis=0).tolist(),
            "memory_token_count_min": int(model["memory_counts"].min()),
            "memory_token_count_median": float(np.median(model["memory_counts"])),
            "memory_token_count_max": int(model["memory_counts"].max()),
        },
        "example_trace": {
            "example_id": test["ids"][0],
            **(model["first_input"] or {}),
            "eligible_memory_tokens": int(model["memory_counts"][0]),
            "modality_gates": model["gates"][0].tolist(),
            "raw_output_shape": [5, 17],
            "horizon_1_forecast_q10_q50_q90": scaler_cpu.inverse_targets(
                model_prediction[:1]
            )[0, 0].tolist(),
            "horizon_1_realized": test["physical"][0, 0].tolist(),
            "status": "single execution trace; not representative performance",
        },
        "operational": {
            "batch_one_latency_ms_shared_gtx1650": result["batch_one_latency_ms"],
            "deterministic_repeat_max_abs_difference": result[
                "deterministic_rerun_max_abs_difference"
            ],
            "training_wall_seconds": result["wall_seconds"],
            "optimizer_updates": result["optimizer_updates"],
            "verified_shards": result["corpus_verification"]["verified_shards"],
            "verified_rows": result["corpus_verification"]["row_count"],
            "verified_bytes": result["corpus_verification"]["verified_bytes"],
            "training_history": result["history"],
        },
        "same_corpus_full_model_context": {
            "parameters": 2_089_554,
            "weighted_pinball": 0.2513909348275368,
            "empirical_baseline": 0.2571927662831899,
            "improvement_percent": 2.2558,
            "tiny_descriptive_improvement_vs_full_percent": 100.0
            * (0.2513909348275368 - benchmark_metrics["chmt_q_tiny"][
                "weighted_normalized_pinball"
            ])
            / 0.2513909348275368,
            "warning": "same accessed partition; not an architectural superiority test",
        },
        "historical_full_model_economic_context_not_tiny": {
            "registered_interval_policy_trades": 0,
            "post_hoc_median_sign_gross_return_percent": 2.16,
            "post_hoc_median_sign_net_return_percent_at_12bp_one_way": -68.07,
            "post_hoc_trades": 484,
            "gross_edge_bp_per_round_trip": 0.445,
            "modeled_cost_bp_per_round_trip": 24.0,
            "warning": "historical full-model diagnostic only; not a Tiny backtest",
        },
        "source_artifacts": {
            "dataset_id": target_corpus.manifest["dataset_id"],
            "dataset_manifest_sha256": sha256_file(args.dataset / "dataset.json"),
            "split_artifact_sha256": split_hash,
            "scaler_artifact_sha256": scaler_cpu.artifact["artifact_sha256"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "result_json_sha256": sha256_file(result_path),
            "model_source_sha256": result["runtime"]["model_source_sha256"],
            "runner_source_sha256": result["runtime"]["runner_source_sha256"],
            "analysis_source_sha256": sha256_file(Path(__file__)),
        },
        "method_notes": [
            "All added baselines and diagnostics are post hoc on an already accessed development partition.",
            "The trailing-three-day and persistence baselines use only labels fully matured before each forecast origin.",
            "Aggregate physical interval width is intentionally omitted because target units differ.",
            "Three marginal quantiles do not define a joint predictive distribution or executable strategy.",
            "Daily and 64-phase summaries are descriptive; four days do not support reliable inference.",
        ],
    }

    computed = report["benchmarks"]["chmt_q_tiny"]["weighted_normalized_pinball"]
    recorded = report["registered_result_reconciliation"]["development_model"]
    if not math.isclose(computed, recorded, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"recomputed model score {computed} != recorded {recorded}")
    computed_baseline = report["benchmarks"]["empirical_marginal_train_only"][
        "weighted_normalized_pinball"
    ]
    recorded_baseline = report["registered_result_reconciliation"][
        "development_baseline"
    ]
    if not math.isclose(computed_baseline, recorded_baseline, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(
            f"recomputed baseline {computed_baseline} != recorded {recorded_baseline}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_report = finite_json(report)
    args.output.write_text(json.dumps(safe_report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": sha256_file(args.output),
                "model_score": computed,
                "baseline_score": computed_baseline,
                "benchmarks": {
                    name: metrics["weighted_normalized_pinball"]
                    for name, metrics in benchmark_metrics.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
