"""Run the authorized CHMT-Q development forecast and economic study.

This harness deliberately lives outside ``tools/numeric-transformer`` so the
frozen training source-tree identity remains unchanged.  It is an offline
research evaluator only: it does not expose an order or broker integration.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

TOOL_ROOT = Path(__file__).resolve().parent
NUMERIC_ROOT = TOOL_ROOT.parent / "numeric-transformer"
if str(NUMERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(NUMERIC_ROOT))

from research_backtest import (  # noqa: E402
    CostScenario,
    TradeObservation,
    buy_and_hold_benchmark,
    cash_benchmark,
    correlation_metrics,
    day_block_bootstrap_ci,
    sign_classification_metrics,
    simulate,
)

from daxalgo_numeric_transformer.chmt_q.baselines import (  # noqa: E402
    ZeroResidualBaseline,
)
from daxalgo_numeric_transformer.chmt_q.contracts import ChmtQBatch  # noqa: E402
from daxalgo_numeric_transformer.chmt_q.data.dataset import ArrowDataset  # noqa: E402
from daxalgo_numeric_transformer.chmt_q.data.research_authorization import (  # noqa: E402
    LocalResearchAuthorization,
)
from daxalgo_numeric_transformer.chmt_q.data.scaler_binding import (  # noqa: E402
    FoldBoundScalerArtifact,
)
from daxalgo_numeric_transformer.chmt_q.dependencies import (  # noqa: E402
    verify_runtime_dependency_lock,
)
from daxalgo_numeric_transformer.chmt_q.experiment import (  # noqa: E402
    ExperimentConfig,
    source_tree_sha256,
)
from daxalgo_numeric_transformer.chmt_q.metrics import (  # noqa: E402
    ChmtQMetricsAccumulator,
)
from daxalgo_numeric_transformer.chmt_q.model import ChmtQTransformer  # noqa: E402
from daxalgo_numeric_transformer.chmt_q.output import ChmtQOutput  # noqa: E402
from daxalgo_numeric_transformer.chmt_q.training.checkpoint import (  # noqa: E402
    load_checkpoint,
)
from daxalgo_numeric_transformer.chmt_q.training.runner import (  # noqa: E402
    _validate_bindings,
    _verify_run_artifacts,
)
from daxalgo_numeric_transformer.chmt_q.training.trainer import (  # noqa: E402
    ChmtQTrainer,
)
from daxalgo_numeric_transformer.chmt_q.training.walk_forward import (  # noqa: E402
    WalkForwardManifest,
)
from daxalgo_numeric_transformer.okx_compile import (  # noqa: E402
    MINUTE_MS,
    OkxCompilerPolicy,
    load_source_contract,
    load_trade_minutes,
)

SCHEMA_VERSION = 1
TARGET_NAMES = ("total_poc_offset", "buy_poc_offset", "sell_poc_offset", "log_volume_residual", "delta_fraction")
PRESENCE_NAMES = ("buy_poc_present", "sell_poc_present")
PARTITION_ORDER = (
    "in_sample_train_replay",
    "selected_validation",
    "untouched_development_forward",
)
POLICIES = (
    "interval_long_flat",
    "median_long_flat",
    "base_cost_hurdle_long_flat",
    "interval_long_short",
    "median_long_short",
)
COSTS = (
    CostScenario("frictionless", 0.0),
    CostScenario("optimistic", 9.0),
    CostScenario("base", 12.0),
    CostScenario("stress", 20.0),
)
PRIMARY_POLICY = "interval_long_flat"
PRIMARY_COST = "base"
INITIAL_CAPITAL = 100_000.0
BOOTSTRAP_SEED = 20_260_726
BOOTSTRAP_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class Paths:
    dataset: Path
    authorization: Path
    split: Path
    scaler: Path
    experiment: Path
    dependency_lock: Path
    checkpoint: Path
    source_data: Path
    output: Path
    device: str
    batch_size: int


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = Paths(
        dataset=args.dataset.resolve(strict=True),
        authorization=args.authorization.resolve(strict=True),
        split=args.split.resolve(strict=True),
        scaler=args.scaler.resolve(strict=True),
        experiment=args.experiment.resolve(strict=True),
        dependency_lock=args.dependency_lock.resolve(strict=True),
        checkpoint=args.checkpoint.resolve(strict=True),
        source_data=args.source_data.resolve(strict=True),
        output=args.output.resolve(),
        device=args.device,
        batch_size=args.batch_size,
    )
    try:
        result = run(paths)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", flush=True)
        return 2
    print(_canonical_json({"status": "complete", **result}), flush=True)
    return 0


def run(paths: Paths) -> dict[str, Any]:
    if paths.output.exists():
        raise FileExistsError(f"output already exists: {paths.output}")
    if paths.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    print("phase preflight: verifying frozen model, data, split, and authorization", flush=True)
    experiment = ExperimentConfig.from_json(_read_text(paths.experiment))
    observed_source_hash = source_tree_sha256(NUMERIC_ROOT)
    if observed_source_hash != experiment.source_tree_sha256:
        raise ValueError("numeric-transformer source tree differs from the frozen experiment")
    observed_lock_hash = verify_runtime_dependency_lock(paths.dependency_lock)
    if observed_lock_hash != experiment.dependency_lock_sha256:
        raise ValueError("runtime dependency lock differs from the frozen experiment")

    authorization = LocalResearchAuthorization.read(paths.authorization)
    if authorization.outer_holdout_allowed:
        raise ValueError("this study requires the local-research outer holdout to remain forbidden")
    dataset = ArrowDataset.open(
        paths.dataset,
        expected_dataset_id=experiment.dataset_id,
        expected_instrument_id=experiment.instrument_id,
        local_research_authorization=authorization,
    )
    split = WalkForwardManifest.from_json(_read_text(paths.split))
    split.validate(dataset.dependencies())
    fold = split.fold(1)
    scaler = FoldBoundScalerArtifact.from_json(
        _read_text(paths.scaler),
        expected_dataset_id=experiment.dataset_id,
        expected_split_manifest_sha256=split.artifact_sha256,
        expected_fold_id=experiment.fold_id,
        ordered_train_example_ids=fold.train_example_ids,
    )
    _validate_bindings(dataset, split, scaler, experiment, 1)

    checkpoint = paths.checkpoint
    run_directory = checkpoint.parent.parent
    if checkpoint.parent != run_directory / "checkpoints":
        raise ValueError("checkpoint must be inside the verified run checkpoint directory")
    baseline, _stored_baseline_metrics = _verify_run_artifacts(
        run_directory,
        config=experiment,
        split=split,
        scaler=scaler,
        allow_completed=True,
        verify_environment=True,
    )
    zero_baseline = ZeroResidualBaseline.from_empirical(baseline)
    summary = json.loads(_read_text(run_directory / "summary.json"))
    if Path(summary["result"]["best_checkpoint"]).resolve() != checkpoint:
        raise ValueError("requested checkpoint is not the run's selected best checkpoint")

    categories = _partition_categories(split)
    allowed_ids = set().union(*categories.values())
    selected_ids = tuple(
        item.example_id for item in dataset.examples if item.example_id in allowed_ids
    )
    if len(selected_ids) != sum(len(value) for value in categories.values()):
        raise ValueError("authorized development example selection is incomplete")
    holdout_ids = frozenset(split.outer_holdout.example_ids)
    if holdout_ids & set(selected_ids):
        raise PermissionError("sealed outer-holdout example entered the study selection")
    id_to_partition = {
        example_id: partition
        for partition, example_ids in categories.items()
        for example_id in example_ids
    }

    paths.output.mkdir(parents=True)
    protocol = _protocol(
        paths,
        experiment,
        dataset,
        split,
        scaler,
        categories,
        observed_source_hash,
        observed_lock_hash,
    )
    protocol_path = paths.output / "study-protocol.json"
    _write_new(protocol_path, _canonical_json(protocol) + "\n")
    protocol_sha256 = _sha256(protocol_path)
    print(f"phase protocol: frozen {protocol_sha256}", flush=True)

    print("phase source: verifying official archives and aggregating genuine trade minutes", flush=True)
    source = load_source_contract(paths.source_data, OkxCompilerPolicy())
    if source.source_manifest_sha256 != dataset.manifest.source_manifest_sha256:
        raise ValueError("raw source manifest differs from the compiled dataset binding")
    trade_minutes, trade_stats = load_trade_minutes(
        source,
        start=date(2026, 6, 1),
        end=date(2026, 6, 24),
    )
    print(
        f"phase source: {trade_stats.row_count} trades -> {trade_stats.minute_count} minutes",
        flush=True,
    )

    trainer = _load_trainer(experiment, fold, checkpoint, scaler, split, baseline, paths.device)
    device = trainer.device
    model = trainer.model
    model.eval()

    accumulators = {
        partition: {
            owner: {
                "normalized": ChmtQMetricsAccumulator(experiment.model),
                "physical": ChmtQMetricsAccumulator(experiment.model),
            }
            for owner in ("model", "empirical_marginal", "zero_residual")
        }
        for partition in PARTITION_ORDER
    }
    calibration = {
        partition: {
            "counts": np.zeros((5, 5), dtype=np.int64),
            "leq": np.zeros((5, 5, 3), dtype=np.int64),
        }
        for partition in PARTITION_ORDER
    }
    poc_errors: dict[tuple[str, int], list[float]] = defaultdict(list)
    proxies: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pinball_effects: dict[str, list[tuple[date, float]]] = defaultdict(list)
    observations: dict[str, dict[str, list[TradeObservation]]] = {
        partition: {policy: [] for policy in POLICIES} for partition in PARTITION_ORDER
    }
    scheduled = {partition: 0 for partition in PARTITION_ORDER}
    execution_rejections = {partition: 0 for partition in PARTITION_ORDER}
    row_counts = {partition: 0 for partition in PARTITION_ORDER}

    predictions_path = paths.output / "predictions.jsonl.gz"
    predictions_staging = paths.output / ".predictions.jsonl.gz.tmp"
    if predictions_staging.exists():
        raise FileExistsError(predictions_staging)
    print(f"phase inference: {len(selected_ids)} authorized examples on {device}", flush=True)
    processed = 0
    with predictions_staging.open("xb") as raw_stream:
        with gzip.GzipFile(fileobj=raw_stream, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as output_stream:
                for materialized in dataset.iter_batches(
                    selected_ids,
                    batch_size=paths.batch_size,
                    partition="evaluation",
                ):
                    batch_partitions = tuple(
                        id_to_partition[example_id] for example_id in materialized.example_ids
                    )
                    if any(example_id in holdout_ids for example_id in materialized.example_ids):
                        raise PermissionError("sealed outer holdout reached inference")
                    normalized = scaler.normalize_batch(
                        materialized,
                        permitted_example_ids=selected_ids,
                        config=experiment.model,
                    )
                    device_batch = _move_batch(normalized.batch, device)
                    with torch.inference_mode():
                        model_normalized_device = model.forward_distribution(device_batch)
                    model_normalized = _cpu_output(model_normalized_device)
                    model_physical = scaler.scaler.denormalize_output(
                        model_normalized, experiment.model
                    )
                    empirical_normalized = baseline.predict(
                        materialized.batch.batch_size, config=experiment.model
                    )
                    zero_normalized = zero_baseline.predict(
                        materialized.batch.batch_size, config=experiment.model
                    )
                    empirical_physical = scaler.scaler.denormalize_output(
                        empirical_normalized, experiment.model
                    )
                    zero_physical = scaler.scaler.denormalize_output(
                        zero_normalized, experiment.model
                    )
                    _consume_batch(
                        materialized=materialized,
                        normalized_batch=normalized.batch,
                        batch_partitions=batch_partitions,
                        model_normalized=model_normalized,
                        model_physical=model_physical,
                        empirical_normalized=empirical_normalized,
                        empirical_physical=empirical_physical,
                        zero_normalized=zero_normalized,
                        zero_physical=zero_physical,
                        accumulators=accumulators,
                        calibration=calibration,
                        poc_errors=poc_errors,
                        proxies=proxies,
                        pinball_effects=pinball_effects,
                        observations=observations,
                        scheduled=scheduled,
                        execution_rejections=execution_rejections,
                        row_counts=row_counts,
                        trade_minutes=trade_minutes,
                        output_stream=output_stream,
                        config=experiment.model,
                    )
                    processed += materialized.batch.batch_size
                    if processed == len(selected_ids) or processed % (paths.batch_size * 25) == 0:
                        print(f"progress inference: {processed}/{len(selected_ids)}", flush=True)
    os.replace(predictions_staging, predictions_path)

    print("phase analysis: forecast, candle proxy, portfolio, and bootstrap metrics", flush=True)
    results = _finalize_results(
        experiment=experiment,
        dataset=dataset,
        split=split,
        scaler=scaler,
        source=source,
        trade_stats=trade_stats,
        accumulators=accumulators,
        calibration=calibration,
        poc_errors=poc_errors,
        proxies=proxies,
        pinball_effects=pinball_effects,
        observations=observations,
        scheduled=scheduled,
        execution_rejections=execution_rejections,
        row_counts=row_counts,
        protocol_sha256=protocol_sha256,
        checkpoint=checkpoint,
        predictions_path=predictions_path,
    )
    results_path = paths.output / "results.json"
    _write_new(results_path, _canonical_json(results) + "\n")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "study_id": protocol["study_id"],
        "outer_holdout_accessed": False,
        "promotable": False,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in (protocol_path, predictions_path, results_path)
        },
    }
    manifest_path = paths.output / "manifest.json"
    _write_new(manifest_path, _canonical_json(manifest) + "\n")
    return {
        "output": str(paths.output),
        "results": str(results_path),
        "manifest": str(manifest_path),
        "outer_holdout_accessed": False,
    }


def _consume_batch(
    *,
    materialized,
    normalized_batch: ChmtQBatch,
    batch_partitions: tuple[str, ...],
    model_normalized: ChmtQOutput,
    model_physical: ChmtQOutput,
    empirical_normalized: ChmtQOutput,
    empirical_physical: ChmtQOutput,
    zero_normalized: ChmtQOutput,
    zero_physical: ChmtQOutput,
    accumulators: dict[str, Any],
    calibration: dict[str, Any],
    poc_errors: dict[tuple[str, int], list[float]],
    proxies: dict[tuple[str, int], dict[str, list[float]]],
    pinball_effects: dict[str, list[tuple[date, float]]],
    observations: dict[str, dict[str, list[TradeObservation]]],
    scheduled: dict[str, int],
    execution_rejections: dict[str, int],
    row_counts: dict[str, int],
    trade_minutes: dict[int, Any],
    output_stream,
    config,
) -> None:
    assert normalized_batch.targets is not None and normalized_batch.target_mask is not None
    assert normalized_batch.presence_labels is not None and normalized_batch.presence_mask is not None
    assert materialized.batch.targets is not None and materialized.batch.target_mask is not None
    assert materialized.batch.presence_labels is not None
    assert materialized.batch.presence_mask is not None
    outputs = {
        "model": (model_normalized, model_physical),
        "empirical_marginal": (empirical_normalized, empirical_physical),
        "zero_residual": (zero_normalized, zero_physical),
    }
    for partition in set(batch_partitions):
        indices = torch.tensor(
            [index for index, value in enumerate(batch_partitions) if value == partition],
            dtype=torch.long,
        )
        normalized_targets = normalized_batch.targets.index_select(0, indices)
        target_mask = normalized_batch.target_mask.index_select(0, indices)
        presence_labels = normalized_batch.presence_labels.index_select(0, indices)
        presence_mask = normalized_batch.presence_mask.index_select(0, indices)
        physical_targets = materialized.batch.targets.index_select(0, indices)
        for owner, (normalized_output, physical_output) in outputs.items():
            selected_normalized = _select_output(normalized_output, indices)
            selected_physical = _select_output(physical_output, indices)
            accumulators[partition][owner]["normalized"].update(
                selected_normalized,
                targets=normalized_targets,
                target_mask=target_mask,
                presence_labels=presence_labels,
                presence_mask=presence_mask,
            )
            accumulators[partition][owner]["physical"].update(
                selected_physical,
                targets=physical_targets,
                target_mask=target_mask,
                presence_labels=presence_labels,
                presence_mask=presence_mask,
            )

    targets = materialized.batch.targets.numpy()
    target_mask_values = materialized.batch.target_mask.numpy()
    presence_labels_values = materialized.batch.presence_labels.numpy()
    presence_mask_values = materialized.batch.presence_mask.numpy()
    normalized_targets_values = normalized_batch.targets.numpy()
    normalized_mask_values = normalized_batch.target_mask.numpy()
    quantiles = model_physical.quantiles.numpy()
    normalized_quantiles = model_normalized.quantiles.numpy()
    presence_probabilities = torch.sigmoid(model_physical.presence_logits).numpy()
    empirical_quantiles = empirical_normalized.quantiles.numpy()
    levels = np.asarray(config.quantile_levels, dtype=np.float64)
    horizon_weights = np.asarray(config.horizon_weights, dtype=np.float64)[:, None]
    target_weights = np.asarray(config.target_weights, dtype=np.float64)[None, :]
    weights = horizon_weights * target_weights

    for index, partition in enumerate(batch_partitions):
        row_counts[partition] += 1
        cutoff_ms = materialized.feature_cutoff_ns[index] // 1_000_000
        day = datetime.fromtimestamp(cutoff_ms / 1000, tz=UTC).date()
        valid = normalized_mask_values[index]
        actual_norm = normalized_targets_values[index]
        model_error = actual_norm[..., None] - normalized_quantiles[index]
        empirical_error = actual_norm[..., None] - empirical_quantiles[index]
        model_pinball = np.maximum(levels * model_error, (levels - 1.0) * model_error)
        empirical_pinball = np.maximum(
            levels * empirical_error, (levels - 1.0) * empirical_error
        )
        denominator = float((valid * weights).sum() * len(levels))
        model_score = float((model_pinball * valid[..., None] * weights[..., None]).sum() / denominator)
        empirical_score = float(
            (empirical_pinball * valid[..., None] * weights[..., None]).sum() / denominator
        )
        pinball_effects[partition].append((day, empirical_score - model_score))

        calibration[partition]["counts"] += valid.astype(np.int64)
        calibration[partition]["leq"] += (
            actual_norm[..., None] <= normalized_quantiles[index]
        ).astype(np.int64) * valid[..., None]

        for horizon in range(5):
            if target_mask_values[index, horizon, 0]:
                poc_errors[(partition, horizon)].append(
                    abs(float(quantiles[index, horizon, 0, 1] - targets[index, horizon, 0]))
                )
            minute = trade_minutes.get(cutoff_ms + horizon * MINUTE_MS)
            if minute is None or not target_mask_values[index, horizon, 0]:
                continue
            proxy = proxies[(partition, horizon)]
            predicted_offset = float(quantiles[index, horizon, 0, 1])
            proxy["predicted_offset"].append(predicted_offset)
            proxy["predicted_q10"].append(float(quantiles[index, horizon, 0, 0]))
            proxy["predicted_q90"].append(float(quantiles[index, horizon, 0, 2]))
            proxy["actual_poc_offset"].append(float(targets[index, horizon, 0]))
            proxy["candle_body_ticks"].append(float(minute.close_tick - minute.open_tick))
            anchor_ticks = materialized.anchor_prices[index] / materialized.tick_sizes[index]
            proxy["anchor_close_ticks"].append(float(minute.close_tick - anchor_ticks))
            proxy["day_ordinal"].append(float(day.toordinal()))

        minute_of_day = (cutoff_ms % (24 * 60 * MINUTE_MS)) // MINUTE_MS
        if minute_of_day % 5 == 0:
            scheduled[partition] += 1
            entry_minute = trade_minutes.get(cutoff_ms + MINUTE_MS)
            exit_minute = trade_minutes.get(cutoff_ms + 4 * MINUTE_MS)
            if entry_minute is None or exit_minute is None:
                execution_rejections[partition] += 1
            else:
                entry_time = datetime.fromtimestamp((cutoff_ms + MINUTE_MS) / 1000, tz=UTC)
                exit_time = datetime.fromtimestamp((cutoff_ms + 5 * MINUTE_MS) / 1000, tz=UTC)
                q10 = float(quantiles[index, 4, 0, 0])
                q50 = float(quantiles[index, 4, 0, 1])
                q90 = float(quantiles[index, 4, 0, 2])
                hurdle_ticks = (
                    materialized.anchor_prices[index]
                    * (2.0 * next(item.one_way_bps for item in COSTS if item.name == PRIMARY_COST) / 10_000.0)
                    / materialized.tick_sizes[index]
                )
                directions = {
                    "interval_long_flat": int(q10 > 0.0),
                    "median_long_flat": int(q50 > 0.0),
                    "base_cost_hurdle_long_flat": int(q10 > hurdle_ticks),
                    "interval_long_short": 1 if q10 > 0.0 else -1 if q90 < 0.0 else 0,
                    "median_long_short": 1 if q50 > 0.0 else -1 if q50 < 0.0 else 0,
                }
                for policy, direction in directions.items():
                    observations[partition][policy].append(
                        TradeObservation(
                            timestamp=entry_time,
                            exit_timestamp=exit_time,
                            day=entry_time.date(),
                            direction=direction,
                            entry_price=entry_minute.open_tick * materialized.tick_sizes[index],
                            exit_price=exit_minute.close_tick * materialized.tick_sizes[index],
                        )
                    )

        row = {
            "schema_version": SCHEMA_VERSION,
            "example_id": materialized.example_ids[index],
            "session_id": materialized.session_ids[index],
            "evidence_class": partition,
            "feature_cutoff_ns": materialized.feature_cutoff_ns[index],
            "decision_ns": materialized.decision_ns[index],
            "anchor_price": materialized.anchor_prices[index],
            "tick_size": materialized.tick_sizes[index],
            "log_volume_reference": materialized.log_volume_references[index],
            "target_names": TARGET_NAMES,
            "quantile_levels": config.quantile_levels,
            "predicted_quantiles_physical": quantiles[index].tolist(),
            "presence_names": PRESENCE_NAMES,
            "predicted_presence_probabilities": presence_probabilities[index].tolist(),
            "actual_targets_physical": targets[index].tolist(),
            "actual_target_mask": target_mask_values[index].tolist(),
            "actual_presence_labels": presence_labels_values[index].tolist(),
            "actual_presence_mask": presence_mask_values[index].tolist(),
        }
        output_stream.write(_canonical_json(row) + "\n")


def _finalize_results(
    *,
    experiment,
    dataset,
    split,
    scaler,
    source,
    trade_stats,
    accumulators,
    calibration,
    poc_errors,
    proxies,
    pinball_effects,
    observations,
    scheduled,
    execution_rejections,
    row_counts,
    protocol_sha256,
    checkpoint,
    predictions_path,
) -> dict[str, Any]:
    forecast: dict[str, Any] = {}
    for partition in PARTITION_ORDER:
        forecast[partition] = {
            owner: {
                scale: accumulators[partition][owner][scale].finalize().to_dict()
                for scale in ("normalized", "physical")
            }
            for owner in ("model", "empirical_marginal", "zero_residual")
        }
        forecast[partition]["model_minus_empirical_primary"] = (
            forecast[partition]["model"]["normalized"]["weighted_primary_metric"]
            - forecast[partition]["empirical_marginal"]["normalized"]["weighted_primary_metric"]
        )
        forecast[partition]["relative_model_improvement_percent"] = 100.0 * (
            1.0
            - forecast[partition]["model"]["normalized"]["weighted_primary_metric"]
            / forecast[partition]["empirical_marginal"]["normalized"]["weighted_primary_metric"]
        )
        forecast[partition]["paired_daily_block_bootstrap_empirical_minus_model"] = asdict(
            day_block_bootstrap_ci(
                pinball_effects[partition],
                confidence=0.95,
                resamples=BOOTSTRAP_RESAMPLES,
                seed=BOOTSTRAP_SEED,
            )
        )
        counts = calibration[partition]["counts"]
        leq = calibration[partition]["leq"]
        forecast[partition]["model_quantile_empirical_cdf"] = np.divide(
            leq,
            counts[..., None],
            out=np.zeros_like(leq, dtype=np.float64),
            where=counts[..., None] > 0,
        ).tolist()
        forecast[partition]["total_poc_error_distribution_ticks"] = [
            _distribution(poc_errors[(partition, horizon)]) for horizon in range(5)
        ]

    proxy_results: dict[str, Any] = {}
    for partition in PARTITION_ORDER:
        proxy_results[partition] = []
        for horizon in range(5):
            values = proxies[(partition, horizon)]
            predicted = values["predicted_offset"]
            actual_poc = values["actual_poc_offset"]
            candle = values["candle_body_ticks"]
            anchor_close = values["anchor_close_ticks"]
            confidence_predictions = [
                1.0 if lower > 0 else -1.0 if upper < 0 else 0.0
                for lower, upper in zip(
                    values["predicted_q10"], values["predicted_q90"], strict=True
                )
            ]
            accepted = [index for index, value in enumerate(confidence_predictions) if value]
            accepted_correct = [
                float(math.copysign(1.0, anchor_close[index]) == confidence_predictions[index])
                for index in accepted
                if anchor_close[index] != 0.0
            ]
            days = [date.fromordinal(int(value)) for value in values["day_ordinal"]]
            accuracy_observations = [
                (day_value, float(_sign(predicted_value) == _sign(actual_value)))
                for day_value, predicted_value, actual_value in zip(
                    days, predicted, anchor_close, strict=True
                )
            ]
            proxy_results[partition].append(
                {
                    "horizon_minute": horizon + 1,
                    "sample_count": len(predicted),
                    "direct_total_poc_direction": asdict(
                        sign_classification_metrics(actual_poc, predicted, mode="ternary")
                    ),
                    "candle_body_direction_proxy": asdict(
                        sign_classification_metrics(candle, predicted, mode="ternary")
                    ),
                    "anchor_to_close_direction_proxy": asdict(
                        sign_classification_metrics(anchor_close, predicted, mode="ternary")
                    ),
                    "predicted_poc_vs_actual_poc_correlation": asdict(
                        correlation_metrics(actual_poc, predicted)
                    ),
                    "predicted_poc_vs_candle_body_correlation": asdict(
                        correlation_metrics(candle, predicted)
                    ),
                    "predicted_poc_vs_anchor_close_correlation": asdict(
                        correlation_metrics(anchor_close, predicted)
                    ),
                    "interval_direction_coverage": len(accepted) / len(predicted),
                    "interval_direction_accuracy_when_active": (
                        fmean(accepted_correct) if accepted_correct else None
                    ),
                    "anchor_close_accuracy_day_block_bootstrap": asdict(
                        day_block_bootstrap_ci(
                            accuracy_observations,
                            confidence=0.95,
                            resamples=BOOTSTRAP_RESAMPLES,
                            seed=BOOTSTRAP_SEED + horizon,
                        )
                    ),
                }
            )

    economics: dict[str, Any] = {}
    for partition in PARTITION_ORDER:
        economics[partition] = {
            "scheduled_decisions": scheduled[partition],
            "execution_rejections": execution_rejections[partition],
            "cash": asdict(cash_benchmark(initial_capital=INITIAL_CAPITAL)),
            "policies": {},
        }
        benchmark_observations = observations[partition][PRIMARY_POLICY]
        for costs in COSTS:
            economics[partition].setdefault("buy_and_hold", {})[costs.name] = asdict(
                buy_and_hold_benchmark(
                    benchmark_observations,
                    costs,
                    initial_capital=INITIAL_CAPITAL,
                )
            )
        for policy in POLICIES:
            economics[partition]["policies"][policy] = {}
            mode = "long_short" if policy.endswith("long_short") else "long_only"
            for costs in COSTS:
                result = simulate(
                    observations[partition][policy],
                    costs,
                    mode=mode,
                    initial_capital=INITIAL_CAPITAL,
                )
                daily = [(item.day, item.net_return) for item in result.daily_returns]
                active_count = result.metrics.trade_count
                available_count = len(observations[partition][policy])
                economics[partition]["policies"][policy][costs.name] = {
                    "metrics": asdict(result.metrics),
                    "daily_returns": [asdict(item) for item in result.daily_returns],
                    "active_decisions": active_count,
                    "abstentions": available_count - active_count,
                    "active_fraction": active_count / available_count,
                    "break_even_one_way_bps": (
                        result.metrics.gross_total_return / active_count * 5_000.0
                        if active_count
                        else None
                    ),
                    "mean_daily_net_return_block_bootstrap": asdict(
                        day_block_bootstrap_ci(
                            daily,
                            confidence=0.95,
                            resamples=BOOTSTRAP_RESAMPLES,
                            seed=BOOTSTRAP_SEED,
                        )
                    ),
                }

    checkpoint_hashes = json.loads(_read_text(checkpoint / "hashes.json"))["sha256"]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "study_kind": "local-non-promotable-development-research",
        "protocol_sha256": protocol_sha256,
        "outer_holdout_accessed": False,
        "outer_holdout_example_count": len(split.outer_holdout.example_ids),
        "evaluated_example_count": sum(row_counts.values()),
        "row_counts": row_counts,
        "artifact_identity": {
            "dataset_id": dataset.manifest.dataset_id,
            "split_manifest_sha256": split.artifact_sha256,
            "outer_holdout_sha256": split.outer_holdout.holdout_sha256,
            "scaler_artifact_sha256": scaler.artifact_sha256,
            "experiment_config_sha256": experiment.config_sha256,
            "checkpoint_step": 2010,
            "checkpoint_files_sha256": checkpoint_hashes,
            "prediction_tape_sha256": _sha256(predictions_path),
            "source_manifest_sha256": source.source_manifest_sha256,
            "source_audit_sha256": source.source_audit_sha256,
        },
        "trade_tape": {
            "archive_count": trade_stats.archive_count,
            "row_count": trade_stats.row_count,
            "minute_count": trade_stats.minute_count,
            "first_timestamp_ms": trade_stats.first_timestamp_ms,
            "last_timestamp_ms": trade_stats.last_timestamp_ms,
            "id_gap_count": trade_stats.id_gap_count,
            "out_of_order_count": trade_stats.out_of_order_count,
        },
        "forecast": forecast,
        "candle_proxy": proxy_results,
        "economics": economics,
        "decision": {
            "primary_partition": "untouched_development_forward",
            "primary_forecast_metric": "normalized weighted pinball vs empirical marginal",
            "primary_economic_policy": PRIMARY_POLICY,
            "primary_cost_scenario": PRIMARY_COST,
            "capital_scale_usdt": INITIAL_CAPITAL,
            "promotion_eligible": False,
        },
    }


def _load_trainer(experiment, fold, checkpoint, scaler, split, baseline, device):
    steps_per_epoch = len(fold.train_example_ids) // experiment.training.effective_batch_size
    trainer = ChmtQTrainer(
        ChmtQTransformer(experiment.model),
        total_optimizer_steps=steps_per_epoch * experiment.max_epochs,
        config=experiment.training,
        device=device,
    )
    restored = load_checkpoint(
        checkpoint,
        model=trainer.model,
        optimizer=trainer.optimizer,
        expected_identity=experiment.checkpoint_identity(),
        restore_rng=False,
    )
    if restored["step"] != 2010:
        raise ValueError("selected checkpoint step differs from the frozen study expectation")
    state = restored["trainer_state"]
    if not isinstance(state, dict) or set(state) != {"trainer", "experiment"}:
        raise ValueError("checkpoint runner state is invalid")
    experiment_state = state["experiment"]
    expected = {
        "scaler_artifact_sha256": scaler.artifact_sha256,
        "split_manifest_sha256": split.artifact_sha256,
        "fold_sha256": fold.fold_sha256,
        "baseline_artifact_sha256": baseline.artifact_sha256,
    }
    if any(experiment_state.get(name) != value for name, value in expected.items()):
        raise ValueError("checkpoint experiment bindings differ from verified artifacts")
    return trainer


def _partition_categories(split) -> dict[str, frozenset[str]]:
    fold1 = split.fold(1)
    fold2 = split.fold(2)
    fold3 = split.fold(3)
    categories = {
        "in_sample_train_replay": frozenset(fold1.train_example_ids),
        "selected_validation": frozenset(fold1.validation_example_ids),
        "untouched_development_forward": frozenset(
            (*fold2.validation_example_ids, *fold3.validation_example_ids)
        ),
    }
    sets = tuple(categories.values())
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise ValueError("evidence partitions overlap")
    if set().union(*sets) & set(split.outer_holdout.example_ids):
        raise ValueError("evidence partitions overlap the outer holdout")
    return categories


def _protocol(paths, experiment, dataset, split, scaler, categories, source_hash, lock_hash):
    content = {
        "schema_version": SCHEMA_VERSION,
        "frozen_before_inference": True,
        "created_at_utc": datetime.now(tz=UTC).isoformat().replace("+00:00", "Z"),
        "purpose": "offline forecast and simulated-trading development study",
        "model_output_contract": "five one-minute marginal quantile forecasts for footprint primitives; not OHLC or trade actions",
        "primary_endpoint": {
            "partition": "untouched_development_forward",
            "forecast": "normalized weighted pinball difference versus empirical-marginal baseline",
            "economic": {
                "policy": PRIMARY_POLICY,
                "cost_scenario": PRIMARY_COST,
                "measure": "net fixed-notional return and PnL per 100000 USDT",
            },
        },
        "partitions": {name: len(values) for name, values in categories.items()},
        "sealed_outer_holdout": {
            "example_count": len(split.outer_holdout.example_ids),
            "holdout_sha256": split.outer_holdout.holdout_sha256,
            "access_authorized": False,
            "will_access": False,
        },
        "forecast_baselines": ["empirical_marginal", "zero_residual"],
        "policies": {
            "interval_long_flat": "long iff five-minute total-POC q10 offset is above zero; otherwise flat",
            "median_long_flat": "long iff five-minute total-POC median offset is above zero; otherwise flat",
            "base_cost_hurdle_long_flat": "long iff q10 exceeds the base modeled round-trip cost expressed in anchor-price ticks",
            "interval_long_short": "long iff q10>0, short iff q90<0, otherwise flat; hypothetical margin proxy",
            "median_long_short": "sign of five-minute total-POC median; hypothetical margin proxy",
        },
        "execution_proxy": {
            "decision_frequency_minutes": 5,
            "signal_horizon": 5,
            "entry": "trade-minute open at cutoff plus one minute",
            "exit": "trade-minute close for minute beginning cutoff plus four minutes",
            "overlap": False,
            "position_size": "one fixed initial-capital notional; no leverage or pyramiding",
            "latency_assumption": "one minute conservative decision-to-entry delay",
            "limitations": [
                "minute trade prices are not executable bid/ask quotes",
                "spread, queue, partial fills, impact, financing, and capacity are not measured",
                "fixed bps scenarios model fee plus slippage assumptions",
            ],
        },
        "cost_scenarios_one_way_bps": {item.name: item.one_way_bps for item in COSTS},
        "bootstrap": {
            "block": "UTC session day",
            "confidence": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "identity": {
            "dataset_id": dataset.manifest.dataset_id,
            "split_manifest_sha256": split.artifact_sha256,
            "scaler_artifact_sha256": scaler.artifact_sha256,
            "experiment_config_sha256": experiment.config_sha256,
            "source_tree_sha256": source_hash,
            "dependency_lock_sha256": lock_hash,
            "authorization_sha256": dataset.local_research_authorization.authorization_sha256,
            "checkpoint": str(paths.checkpoint),
        },
    }
    identity_payload = dict(content)
    identity_payload.pop("created_at_utc")
    content["study_id"] = hashlib.sha256(
        _canonical_json(identity_payload).encode("utf-8")
    ).hexdigest()
    return content


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dependency-lock", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser


def _move_batch(batch: ChmtQBatch, device: torch.device) -> ChmtQBatch:
    def move(value):
        return None if value is None else value.to(device=device)

    return ChmtQBatch(
        bars=move(batch.bars),
        bars_mask=move(batch.bars_mask),
        l1=move(batch.l1),
        l1_mask=move(batch.l1_mask),
        l2=move(batch.l2),
        l2_level_mask=move(batch.l2_level_mask),
        l2_snapshot_mask=move(batch.l2_snapshot_mask),
        quality=move(batch.quality),
        targets=move(batch.targets),
        target_mask=move(batch.target_mask),
        presence_labels=move(batch.presence_labels),
        presence_mask=move(batch.presence_mask),
    )


def _cpu_output(output: ChmtQOutput) -> ChmtQOutput:
    return ChmtQOutput(
        output.quantiles.detach().to(device="cpu"),
        output.presence_logits.detach().to(device="cpu"),
    )


def _select_output(output: ChmtQOutput, indices: torch.Tensor) -> ChmtQOutput:
    return ChmtQOutput(
        output.quantiles.index_select(0, indices),
        output.presence_logits.index_select(0, indices),
    )


def _distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "median": median(values),
        "p90": _percentile(ordered, 0.90),
        "p99": _percentile(ordered, 0.99),
        "within_1_tick_fraction": sum(value <= 1.0 for value in values) / len(values),
        "within_2_tick_fraction": sum(value <= 2.0 for value in values) / len(values),
    }


def _percentile(ordered: list[float], quantile: float) -> float:
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _read_text(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {path}")
    return path.read_text(encoding="utf-8")


def _write_new(path: Path, content: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    staging = path.with_name(f".{path.name}.tmp")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    staging.write_text(content, encoding="utf-8", newline="\n")
    os.replace(staging, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


if __name__ == "__main__":
    raise SystemExit(main())
