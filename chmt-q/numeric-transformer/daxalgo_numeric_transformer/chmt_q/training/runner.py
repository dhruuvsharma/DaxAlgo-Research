"""Fail-closed epoch orchestration for one CHMT-Q development fold."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from ..baselines import EmpiricalBaseline, ZeroResidualBaseline
from ..contracts import ChmtQBatch
from ..data.dataset import ArrowDataset
from ..data.scaler_binding import (
    FoldBoundScalerArtifact,
    ordered_train_example_ids_sha256,
)
from ..metrics import ChmtQMetricReport, ChmtQMetricsAccumulator
from ..model import ChmtQTransformer
from .checkpoint import load_checkpoint, save_checkpoint
from .optim import seed_everything
from .trainer import ChmtQEvaluationResult, ChmtQTrainer, evaluate_model
from .walk_forward import WalkForwardManifest, iter_fold_batches

if TYPE_CHECKING:
    from ..experiment import ExperimentConfig

_RUN_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentRunResult:
    """Reviewable terminal state of one development-fold training run."""

    run_id: str
    run_directory: Path
    epochs_completed: int
    optimizer_steps: int
    best_epoch: int
    best_metric: float
    best_checkpoint: Path
    stopped_early: bool
    best_baseline_name: str
    baseline_primary_metric: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_directory": str(self.run_directory),
            "epochs_completed": self.epochs_completed,
            "optimizer_steps": self.optimizer_steps,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_metric,
            "best_checkpoint": str(self.best_checkpoint),
            "stopped_early": self.stopped_early,
            "best_baseline_name": self.best_baseline_name,
            "baseline_primary_metric": self.baseline_primary_metric,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentEvaluationResult:
    """Read-only validation result for one verified development checkpoint."""

    run_id: str
    checkpoint: Path
    checkpoint_step: int
    fold_id: str
    loss: float
    primary_metric_name: str
    primary_metric: float
    best_baseline_name: str
    baseline_primary_metric: float
    metrics: ChmtQMetricReport
    physical_metrics: ChmtQMetricReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "checkpoint": str(self.checkpoint),
            "checkpoint_step": self.checkpoint_step,
            "fold_id": self.fold_id,
            "loss": self.loss,
            "primary_metric_name": self.primary_metric_name,
            "primary_metric": self.primary_metric,
            "best_baseline_name": self.best_baseline_name,
            "baseline_primary_metric": self.baseline_primary_metric,
            "metrics": self.metrics.to_dict(),
            "physical_metrics": self.physical_metrics.to_dict(),
            "outer_holdout_accessed": False,
        }


@dataclass(frozen=True, slots=True)
class _BaselineMetrics:
    normalized: ChmtQMetricReport
    physical: ChmtQMetricReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized": self.normalized.to_dict(),
            "physical": self.physical.to_dict(),
        }


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None,
    *,
    phase: str,
    status: str,
    run_id: str,
    fold_id: str,
    **details: Any,
) -> None:
    if callback is None:
        return
    callback(
        {
            "phase": phase,
            "status": status,
            "run_id": run_id,
            "fold_id": fold_id,
            **details,
        }
    )


def run_development_experiment(
    dataset: ArrowDataset,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    config: ExperimentConfig,
    *,
    fold_number: int,
    output_root: str | Path,
    source_root: str | Path,
    dependency_lock: str | Path,
    device: torch.device | str | None = None,
    resume_checkpoint: str | Path | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ExperimentRunResult:
    """Train/validate one fold without exposing the sealed outer holdout."""

    from ..dependencies import verify_runtime_dependency_lock
    from ..experiment import source_tree_sha256

    _validate_bindings(dataset, split, scaler, config, fold_number)
    if source_tree_sha256(source_root) != config.source_tree_sha256:
        raise ValueError("current source tree differs from the frozen experiment hash")
    if verify_runtime_dependency_lock(dependency_lock) != config.dependency_lock_sha256:
        raise ValueError("dependency lock differs from the frozen experiment hash")
    if device is not None and str(torch.device(device)) != config.device:
        raise ValueError("requested device differs from the frozen experiment device")

    fold = split.fold(fold_number)
    steps_per_epoch = len(fold.train_example_ids) // config.training.effective_batch_size
    if steps_per_epoch <= 0:
        raise ValueError("training fold is smaller than one configured effective batch")
    total_steps = steps_per_epoch * config.max_epochs
    resolved_device = torch.device(config.device)

    seed_everything(config.seed, deterministic=config.training.deterministic)
    model = ChmtQTransformer(config.model)
    trainer = ChmtQTrainer(
        model,
        total_optimizer_steps=total_steps,
        config=config.training,
        device=resolved_device,
    )
    model.enable_attention_logit_telemetry(
        runaway_threshold=config.attention_logit_threshold
    )

    run_directory = _run_directory(output_root, config.run_id)
    _emit_progress(
        progress_callback,
        phase="baseline",
        status="start",
        run_id=config.run_id,
        fold_id=config.fold_id,
        max_epochs=config.max_epochs,
    )
    if resume_checkpoint is None:
        if run_directory.exists() or run_directory.is_symlink():
            baseline, baseline_metrics = _recover_precheckpoint_run(
                run_directory,
                config=config,
                split=split,
                scaler=scaler,
                expected_first_epoch_step=steps_per_epoch,
            )
        else:
            baseline, baseline_metrics = _fit_and_evaluate_baseline(
                dataset, split, scaler, config, fold_number
            )
            _initialize_run(
                run_directory,
                config=config,
                split=split,
                scaler=scaler,
                baseline=baseline,
                baseline_metrics=baseline_metrics,
                fold_number=fold_number,
                environment=_environment_payload(resolved_device, config),
            )
        start_epoch = 1
        best_epoch = 0
        best_metric: float | None = None
        best_tie_breaker: float | None = None
        best_checkpoint: Path | None = None
        validations_without_improvement = 0
        stopped_early = False
    else:
        baseline, baseline_metrics = _verify_run_artifacts(
            run_directory,
            config=config,
            split=split,
            scaler=scaler,
        )
        restored = _restore_run_checkpoint(
            resume_checkpoint,
            run_directory=run_directory,
            trainer=trainer,
            config=config,
            scaler=scaler,
            split=split,
            baseline=baseline,
        )
        start_epoch = restored["next_epoch"]
        best_epoch = restored["best_epoch"]
        best_metric = restored["best_metric"]
        best_tie_breaker = restored["best_tie_breaker"]
        best_checkpoint = run_directory / restored["best_checkpoint"]
        validations_without_improvement = restored[
            "validations_without_improvement"
        ]
        stopped_early = restored["stopped_early"]

    epochs_completed = start_epoch - 1
    final_epoch = start_epoch - 1 if stopped_early else config.max_epochs
    for epoch in range(start_epoch, final_epoch + 1):
        epoch_seed = _epoch_seed(config.seed, fold.fold_sha256, epoch)
        if resolved_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(resolved_device)
        model.reset_attention_logit_telemetry()
        _emit_progress(
            progress_callback,
            phase="epoch",
            status="start",
            run_id=config.run_id,
            fold_id=config.fold_id,
            epoch=epoch,
            max_epochs=config.max_epochs,
            optimizer_step=trainer.optimizer_step,
        )

        optimizer_window_callback: Callable[[dict[str, Any]], None] | None = None
        if progress_callback is not None:

            def optimizer_window_callback(details: dict[str, Any]) -> None:
                _emit_progress(
                    progress_callback,
                    phase="optimizer_window",
                    status="progress",
                    run_id=config.run_id,
                    fold_id=config.fold_id,
                    epoch=epoch,
                    max_epochs=config.max_epochs,
                    windows_per_epoch=steps_per_epoch,
                    **details,
                )

        started = time.perf_counter()
        window_records, trained_examples, mask_report = _train_epoch(
            trainer,
            dataset,
            split,
            scaler,
            fold_number=fold_number,
            epoch_seed=epoch_seed,
            optimizer_window_callback=optimizer_window_callback,
        )
        elapsed_seconds = time.perf_counter() - started

        first_validation_batch: ChmtQBatch | None = None

        def validation_batches() -> Iterator[ChmtQBatch]:
            nonlocal first_validation_batch
            for batch in _normalized_fold_batches(
                dataset,
                split,
                scaler,
                fold_number=fold_number,
                partition="validation",
                batch_size=config.training.micro_batch_size,
            ):
                if first_validation_batch is None:
                    first_validation_batch = batch
                yield batch

        validation = evaluate_model(
            model,
            validation_batches(),
            device=resolved_device,
            metric_scaler=scaler.scaler,
        )
        if validation.physical_metrics is None:  # pragma: no cover - scaler supplied
            raise RuntimeError("physical validation metrics were not produced")
        selection_metrics = _selection_metrics(validation, config.primary_metric)
        primary_metric = _primary_metric(selection_metrics, config.primary_metric)
        tie_breaker = _tie_breaker(validation.physical_metrics)
        improved = _is_improved(
            primary_metric,
            tie_breaker,
            best_metric,
            best_tie_breaker,
            config.minimum_improvement,
        )
        if improved:
            best_epoch = epoch
            best_metric = primary_metric
            best_tie_breaker = tie_breaker
            validations_without_improvement = 0
        else:
            validations_without_improvement += 1

        attention = _attention_payload(model.collect_attention_logit_telemetry(reset=True))
        if first_validation_batch is None:  # pragma: no cover - evaluate_model rejects this
            raise RuntimeError("validation partition produced no batches")
        gate_means = _gate_means(model, first_validation_batch, resolved_device)
        first_validation_batch = None
        stopped_early = (
            validations_without_improvement >= config.early_stop_patience
        )
        _emit_progress(
            progress_callback,
            phase="validation",
            status="complete",
            run_id=config.run_id,
            fold_id=config.fold_id,
            epoch=epoch,
            max_epochs=config.max_epochs,
            optimizer_step=trainer.optimizer_step,
            primary_metric_name=config.primary_metric,
            primary_metric=primary_metric,
            improved=improved,
        )
        epoch_payload = {
            "schema_version": _RUN_SCHEMA_VERSION,
            "run_id": config.run_id,
            "fold_id": config.fold_id,
            "epoch": epoch,
            "epoch_seed": epoch_seed,
            "optimizer_step": trainer.optimizer_step,
            "trained_examples": trained_examples,
            "dropped_train_examples": len(fold.train_example_ids) - trained_examples,
            "elapsed_seconds": elapsed_seconds,
            "examples_per_second": trained_examples / max(elapsed_seconds, 1.0e-12),
            "windows": window_records,
            "mask_fractions": mask_report,
            "validation": {
                "loss": validation.loss,
                "pinball": validation.pinball,
                "median_huber": validation.median_huber,
                "presence": validation.presence,
                "metrics": validation.metrics.to_dict(),
                "physical_metrics": validation.physical_metrics.to_dict(),
                "primary_metric_name": config.primary_metric,
                "primary_metric": primary_metric,
                "tie_breaker_median_mae": tie_breaker,
            },
            "selection": {
                "improved": improved,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "best_tie_breaker": best_tie_breaker,
                "validations_without_improvement": validations_without_improvement,
                "stopped_early": stopped_early,
            },
            "telemetry": {
                "attention_logits": attention,
                "modality_gate_means": gate_means,
                "cuda_peak_memory_bytes": (
                    torch.cuda.max_memory_allocated(resolved_device)
                    if resolved_device.type == "cuda"
                    else 0
                ),
            },
        }
        event_path, event_sha256 = _write_epoch_event(
            run_directory, epoch, epoch_payload
        )
        checkpoint_due = (
            improved
            or stopped_early
            or epoch == config.max_epochs
            or epoch % config.checkpoint_every_epochs == 0
        )
        if checkpoint_due:
            checkpoint = save_checkpoint(
                run_directory / "checkpoints",
                step=trainer.optimizer_step,
                model=trainer.model,
                optimizer=trainer.optimizer,
                run_identity=config.checkpoint_identity(),
                trainer_state={
                    "trainer": trainer.state_dict(),
                    "experiment": {
                        "schema_version": _RUN_SCHEMA_VERSION,
                        "completed_epoch": epoch,
                        "next_epoch": epoch + 1,
                        "best_epoch": best_epoch,
                        "best_metric": best_metric,
                        "best_tie_breaker": best_tie_breaker,
                        "best_checkpoint": (
                            "pending-current-checkpoint"
                            if improved
                            else _relative_path(best_checkpoint, run_directory)  # type: ignore[arg-type]
                        ),
                        "validations_without_improvement": (
                            validations_without_improvement
                        ),
                        "stopped_early": stopped_early,
                        "epoch_event": _relative_path(event_path, run_directory),
                        "epoch_event_sha256": event_sha256,
                        "scaler_artifact_sha256": scaler.artifact_sha256,
                        "split_manifest_sha256": split.artifact_sha256,
                        "fold_sha256": fold.fold_sha256,
                        "baseline_artifact_sha256": baseline.artifact_sha256,
                    },
                },
            )
            if improved:
                best_checkpoint = checkpoint
                _write_selection_record(
                    run_directory,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    metric=primary_metric,
                    tie_breaker=tie_breaker,
                )
                _patch_checkpoint_best_path(checkpoint, run_directory)
            _emit_progress(
                progress_callback,
                phase="checkpoint",
                status="committed",
                run_id=config.run_id,
                fold_id=config.fold_id,
                epoch=epoch,
                max_epochs=config.max_epochs,
                optimizer_step=trainer.optimizer_step,
                checkpoint=str(checkpoint),
                improved=improved,
            )
        epochs_completed = epoch
        _emit_progress(
            progress_callback,
            phase="epoch",
            status="committed",
            run_id=config.run_id,
            fold_id=config.fold_id,
            epoch=epoch,
            max_epochs=config.max_epochs,
            optimizer_step=trainer.optimizer_step,
            checkpoint_due=checkpoint_due,
            improved=improved,
            stopped_early=stopped_early,
        )
        if stopped_early:
            break

    if best_metric is None or best_checkpoint is None or best_epoch <= 0:
        raise RuntimeError("experiment ended without a selected checkpoint")
    best_baseline_name, best_baseline_report = _best_baseline_entry(
        baseline_metrics, config.primary_metric
    )
    result = ExperimentRunResult(
        run_id=config.run_id,
        run_directory=run_directory,
        epochs_completed=epochs_completed,
        optimizer_steps=trainer.optimizer_step,
        best_epoch=best_epoch,
        best_metric=best_metric,
        best_checkpoint=best_checkpoint,
        stopped_early=stopped_early,
        best_baseline_name=best_baseline_name,
        baseline_primary_metric=_primary_metric(
            best_baseline_report, config.primary_metric
        ),
    )
    _write_new_text(
        run_directory / "summary.json",
        _canonical_json(
            {
                "schema_version": _RUN_SCHEMA_VERSION,
                "result": result.to_dict(),
                "outer_holdout_accessed": False,
                "local_research_only": config.local_research_only,
                "local_research_authorization_sha256": (
                    config.local_research_authorization_sha256
                ),
                "promotable": not config.local_research_only,
            }
        )
        + "\n",
    )
    model.disable_attention_logit_telemetry(reset=True)
    return result


def evaluate_development_checkpoint(
    dataset: ArrowDataset,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    config: ExperimentConfig,
    *,
    fold_number: int,
    checkpoint: str | Path,
    source_root: str | Path,
    dependency_lock: str | Path,
    device: torch.device | str = "cpu",
) -> DevelopmentEvaluationResult:
    """Evaluate one verified checkpoint on validation data only."""

    from ..dependencies import verify_runtime_dependency_lock
    from ..experiment import source_tree_sha256

    _validate_bindings(dataset, split, scaler, config, fold_number)
    if source_tree_sha256(source_root) != config.source_tree_sha256:
        raise ValueError("current source tree differs from the frozen experiment hash")
    if verify_runtime_dependency_lock(dependency_lock) != config.dependency_lock_sha256:
        raise ValueError("dependency lock differs from the frozen experiment hash")

    checkpoint_path = Path(checkpoint).resolve()
    run_directory = checkpoint_path.parent.parent
    if checkpoint_path.parent != run_directory / "checkpoints":
        raise ValueError("checkpoint must be inside a run checkpoints directory")
    baseline, baseline_metrics = _verify_run_artifacts(
        run_directory,
        config=config,
        split=split,
        scaler=scaler,
        allow_completed=True,
        verify_environment=False,
    )
    fold = split.fold(fold_number)
    steps_per_epoch = len(fold.train_example_ids) // config.training.effective_batch_size
    if steps_per_epoch <= 0:
        raise ValueError("training fold is smaller than one configured effective batch")
    total_steps = steps_per_epoch * config.max_epochs
    seed_everything(config.seed, deterministic=config.training.deterministic)
    trainer = ChmtQTrainer(
        ChmtQTransformer(config.model),
        total_optimizer_steps=total_steps,
        config=config.training,
        device=device,
    )
    restored = load_checkpoint(
        checkpoint_path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        expected_identity=config.checkpoint_identity(),
        restore_rng=False,
    )
    state = restored["trainer_state"]
    if not isinstance(state, dict) or set(state) != {"trainer", "experiment"}:
        raise ValueError("checkpoint does not contain experiment-runner state")
    experiment = state["experiment"]
    if not isinstance(experiment, dict):
        raise TypeError("checkpoint experiment state must be an object")
    expected_bindings = {
        "scaler_artifact_sha256": scaler.artifact_sha256,
        "split_manifest_sha256": split.artifact_sha256,
        "fold_sha256": fold.fold_sha256,
        "baseline_artifact_sha256": baseline.artifact_sha256,
    }
    for name, expected in expected_bindings.items():
        if experiment.get(name) != expected:
            raise ValueError(f"checkpoint {name} differs from the evaluation artifacts")

    evaluation = evaluate_model(
        trainer.model,
        _normalized_fold_batches(
            dataset,
            split,
            scaler,
            fold_number=fold_number,
            partition="validation",
            batch_size=config.training.micro_batch_size,
        ),
        device=device,
        metric_scaler=scaler.scaler,
    )
    if evaluation.physical_metrics is None:  # pragma: no cover - scaler supplied
        raise RuntimeError("physical validation metrics were not produced")
    best_baseline_name, best_baseline_report = _best_baseline_entry(
        baseline_metrics, config.primary_metric
    )
    return DevelopmentEvaluationResult(
        run_id=config.run_id,
        checkpoint=checkpoint_path,
        checkpoint_step=_non_negative_int(restored["step"], "checkpoint step"),
        fold_id=config.fold_id,
        loss=evaluation.loss,
        primary_metric_name=config.primary_metric,
        primary_metric=_primary_metric(
            _selection_metrics(evaluation, config.primary_metric),
            config.primary_metric,
        ),
        best_baseline_name=best_baseline_name,
        baseline_primary_metric=_primary_metric(
            best_baseline_report, config.primary_metric
        ),
        metrics=evaluation.metrics,
        physical_metrics=evaluation.physical_metrics,
    )


def _validate_bindings(
    dataset: ArrowDataset,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    config: ExperimentConfig,
    fold_number: int,
) -> None:
    from ..experiment import ExperimentConfig

    if not isinstance(dataset, ArrowDataset):
        raise TypeError("dataset must be an ArrowDataset")
    if not isinstance(split, WalkForwardManifest):
        raise TypeError("split must be a WalkForwardManifest")
    if not isinstance(scaler, FoldBoundScalerArtifact):
        raise TypeError("scaler must be a FoldBoundScalerArtifact")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    dataset.require_development_access()
    observed_authorization_sha256 = (
        dataset.local_research_authorization.authorization_sha256
        if dataset.local_research_authorization
        else None
    )
    if (
        observed_authorization_sha256
        != config.local_research_authorization_sha256
    ):
        raise ValueError(
            "experiment local-research authorization differs from the opened dataset"
        )
    split.validate(dataset.dependencies())
    if dataset.manifest.dataset_id != config.dataset_id:
        raise ValueError("experiment dataset identity does not match the opened dataset")
    if dataset.manifest.instrument_id != config.instrument_id:
        raise ValueError("experiment instrument identity does not match the opened dataset")
    if split.source_dataset_hash != config.dataset_id:
        raise ValueError("walk-forward manifest is bound to a different dataset")
    if split.artifact_sha256 != config.split_manifest_sha256:
        raise ValueError("experiment split identity does not match the manifest")
    fold = split.fold(fold_number)
    expected_fold_id = f"fold-{fold_number:02d}"
    if config.fold_id != expected_fold_id:
        raise ValueError(f"experiment fold_id must be {expected_fold_id}")
    if scaler.dataset_id != config.dataset_id:
        raise ValueError("fold scaler is bound to a different dataset")
    if scaler.split_manifest_sha256 != split.artifact_sha256:
        raise ValueError("fold scaler is bound to a different split manifest")
    if scaler.fold_id != config.fold_id:
        raise ValueError("fold scaler is bound to a different fold")
    if scaler.artifact_sha256 != config.scaler_artifact_sha256:
        raise ValueError("fold scaler identity does not match the experiment")
    if scaler.train_example_ids != fold.train_example_ids:
        raise ValueError("fold scaler is bound to a different ordered train set")
    if scaler.ordered_train_example_ids_sha256 != ordered_train_example_ids_sha256(
        fold.train_example_ids
    ):
        raise ValueError("fold scaler train-set identity differs")


def _fit_and_evaluate_baseline(
    dataset: ArrowDataset,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    config: ExperimentConfig,
    fold_number: int,
) -> tuple[EmpiricalBaseline, dict[str, _BaselineMetrics]]:
    baseline = EmpiricalBaseline.fit(
        _normalized_fold_batches(
            dataset,
            split,
            scaler,
            fold_number=fold_number,
            partition="train",
            batch_size=config.training.micro_batch_size,
            seed=config.seed,
        ),
        source_example_set_sha256=scaler.ordered_train_example_ids_sha256,
        config=config.model,
    )
    baselines = {
        "empirical_marginal": baseline,
        "zero_residual": ZeroResidualBaseline.from_empirical(baseline),
    }
    normalized_accumulators = {
        name: ChmtQMetricsAccumulator(config.model) for name in baselines
    }
    physical_accumulators = {
        name: ChmtQMetricsAccumulator(config.model) for name in baselines
    }
    for batch in _normalized_fold_batches(
        dataset,
        split,
        scaler,
        fold_number=fold_number,
        partition="validation",
        batch_size=config.training.micro_batch_size,
    ):
        assert batch.targets is not None and batch.target_mask is not None
        assert batch.presence_labels is not None and batch.presence_mask is not None
        for name, registered in baselines.items():
            prediction = registered.predict(batch.batch_size, config=config.model)
            normalized_accumulators[name].update(
                prediction,
                targets=batch.targets,
                target_mask=batch.target_mask,
                presence_labels=batch.presence_labels,
                presence_mask=batch.presence_mask,
            )
            physical_accumulators[name].update(
                scaler.scaler.denormalize_output(prediction, config.model),
                targets=scaler.scaler.denormalize_targets(batch.targets),
                target_mask=batch.target_mask,
                presence_labels=batch.presence_labels,
                presence_mask=batch.presence_mask,
            )
    return baseline, {
        name: _BaselineMetrics(
            normalized=normalized_accumulators[name].finalize(),
            physical=physical_accumulators[name].finalize(),
        )
        for name in baselines
    }


def _normalized_fold_batches(
    dataset: ArrowDataset,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    *,
    fold_number: int,
    partition: str,
    batch_size: int,
    seed: int | None = None,
    drop_last: bool = False,
) -> Iterator[ChmtQBatch]:
    fold = split.fold(fold_number)
    permitted = (
        fold.train_example_ids if partition == "train" else fold.validation_example_ids
    )
    materialized = iter_fold_batches(
        dataset,
        split,
        fold_number=fold_number,
        partition=partition,  # type: ignore[arg-type]
        batch_size=batch_size,
        seed=seed,
        drop_last=drop_last,
    )
    for item in materialized:
        yield scaler.normalize_batch(item, permitted_example_ids=permitted).batch


def _train_epoch(
    trainer: ChmtQTrainer,
    dataset: ArrowDataset,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    *,
    fold_number: int,
    epoch_seed: int,
    optimizer_window_callback: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], int, dict[str, float]]:
    config = trainer.training_config
    iterator = _normalized_fold_batches(
        dataset,
        split,
        scaler,
        fold_number=fold_number,
        partition="train",
        batch_size=config.micro_batch_size,
        seed=epoch_seed,
        drop_last=True,
    )
    records: list[dict[str, Any]] = []
    window: list[ChmtQBatch] = []
    trained_examples = 0
    mask_totals = _empty_mask_totals()
    for batch in iterator:
        window.append(batch)
        if len(window) != config.accumulation_steps:
            continue
        for accumulated_batch in window:
            trained_examples += accumulated_batch.batch_size
            _add_mask_totals(mask_totals, accumulated_batch)
        result = trainer.train_accumulation_window(tuple(window))
        record = {
            "optimizer_step": result.optimizer_step,
            "learning_rate": result.learning_rate,
            "loss": result.loss,
            "pinball": result.pinball,
            "median_huber": result.median_huber,
            "presence": result.presence,
            "auxiliary_scale": result.auxiliary_scale,
            "gradient_norm": result.gradient_norm,
            "update_norm": result.update_norm,
        }
        records.append(record)
        if optimizer_window_callback is not None:
            optimizer_window_callback(
                {
                    "window": len(records),
                    "optimizer_step": result.optimizer_step,
                }
            )
        window.clear()
    # Drop a final partial accumulation window deterministically. The caller records
    # the exact number of omitted examples in the immutable epoch event.
    if not records:
        raise ValueError("training fold produced no complete accumulation windows")
    return records, trained_examples, _mask_fractions(mask_totals)


def _initialize_run(
    run_directory: Path,
    *,
    config: ExperimentConfig,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    baseline: EmpiricalBaseline,
    baseline_metrics: dict[str, _BaselineMetrics],
    fold_number: int,
    environment: dict[str, Any],
) -> None:
    run_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = run_directory.parent / f".{run_directory.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        (staging / "epochs").mkdir()
        (staging / "checkpoints").mkdir()
        (staging / "selections").mkdir()
        (staging / "config.json").write_text(
            config.to_json() + "\n", encoding="utf-8", newline="\n"
        )
        (staging / "walk-forward.json").write_text(
            split.to_json(), encoding="utf-8", newline="\n"
        )
        (staging / "scaler.json").write_text(
            scaler.to_json(), encoding="utf-8", newline="\n"
        )
        (staging / "baseline.json").write_text(
            baseline.to_json() + "\n", encoding="utf-8", newline="\n"
        )
        (staging / "baseline-metrics.json").write_text(
            _artifact_json(
                {
                    name: report.to_dict()
                    for name, report in sorted(baseline_metrics.items())
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "run.json").write_text(
            _artifact_json(
                {
                    "schema_version": _RUN_SCHEMA_VERSION,
                    "run_id": config.run_id,
                    "fold_number": fold_number,
                    "fold_sha256": split.fold(fold_number).fold_sha256,
                    "environment": environment,
                    "outer_holdout_accessed": False,
                    "local_research_only": config.local_research_only,
                    "local_research_authorization_sha256": (
                        config.local_research_authorization_sha256
                    ),
                    "promotable": not config.local_research_only,
                }
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, run_directory)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _verify_run_artifacts(
    run_directory: Path,
    *,
    config: ExperimentConfig,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    allow_completed: bool = False,
    verify_environment: bool = True,
) -> tuple[EmpiricalBaseline, dict[str, _BaselineMetrics]]:
    from ..experiment import ExperimentConfig

    if not run_directory.is_dir() or run_directory.is_symlink():
        raise ValueError("resume run directory is missing or unsafe")
    if (run_directory / "summary.json").exists() and not allow_completed:
        raise RuntimeError("a completed experiment cannot be resumed")
    if ExperimentConfig.from_json(
        (run_directory / "config.json").read_text(encoding="utf-8")
    ) != config:
        raise ValueError("run configuration differs from requested experiment")
    stored_split = WalkForwardManifest.from_json(
        (run_directory / "walk-forward.json").read_text(encoding="utf-8")
    )
    if stored_split != split:
        raise ValueError("stored walk-forward manifest differs")
    stored_scaler = FoldBoundScalerArtifact.from_json(
        (run_directory / "scaler.json").read_text(encoding="utf-8"),
        expected_dataset_id=config.dataset_id,
        expected_split_manifest_sha256=split.artifact_sha256,
        expected_fold_id=config.fold_id,
        ordered_train_example_ids=split.fold(
            int(config.fold_id.removeprefix("fold-"))
        ).train_example_ids,
    )
    if stored_scaler != scaler:
        raise ValueError("stored fold scaler differs")
    run_payload = _read_artifact_json(run_directory / "run.json")
    if not isinstance(run_payload, dict) or run_payload.get("run_id") != config.run_id:
        raise ValueError("stored run metadata differs from the experiment")
    expected_research_policy = {
        "local_research_only": config.local_research_only,
        "local_research_authorization_sha256": (
            config.local_research_authorization_sha256
        ),
        "promotable": not config.local_research_only,
    }
    if any(run_payload.get(name) != value for name, value in expected_research_policy.items()):
        raise ValueError("stored run research policy differs from the experiment")
    if verify_environment and run_payload.get("environment") != _environment_payload(
        torch.device(config.device), config
    ):
        raise ValueError("current runtime environment differs from the frozen run")
    baseline = EmpiricalBaseline.from_json(
        (run_directory / "baseline.json").read_text(encoding="utf-8")
    )
    metrics_payload = _read_artifact_json(run_directory / "baseline-metrics.json")
    metrics = _metric_suite_from_dict(metrics_payload)
    return baseline, metrics


def _recover_precheckpoint_run(
    run_directory: Path,
    *,
    config: ExperimentConfig,
    split: WalkForwardManifest,
    scaler: FoldBoundScalerArtifact,
    expected_first_epoch_step: int,
) -> tuple[EmpiricalBaseline, dict[str, _BaselineMetrics]]:
    """Replay epoch one only when a verified run has no published checkpoint."""

    baseline, metrics = _verify_run_artifacts(
        run_directory,
        config=config,
        split=split,
        scaler=scaler,
    )
    checkpoints_root = run_directory / "checkpoints"
    if checkpoints_root.is_symlink() or not checkpoints_root.is_dir():
        raise ValueError("run checkpoint directory is missing or unsafe")
    checkpoint_staging: list[Path] = []
    for item in checkpoints_root.iterdir():
        if item.name.startswith("step-"):
            raise FileExistsError(
                "existing run has a published checkpoint; pass the latest "
                "checkpoint via resume_checkpoint"
            )
        staging_step = _checkpoint_staging_step(item)
        if staging_step is None:
            raise ValueError("checkpoint directory contains an unknown artifact")
        if staging_step != expected_first_epoch_step:
            raise ValueError("unpublished checkpoint staging has an invalid step")
        checkpoint_staging.append(item)

    selections_root = run_directory / "selections"
    if selections_root.is_symlink() or not selections_root.is_dir():
        raise ValueError("run selection directory is missing or unsafe")
    if any(selections_root.iterdir()):
        raise ValueError("a run without checkpoints must not contain selections")

    epoch_events = _ordered_epoch_events(run_directory)
    if epoch_events and [epoch for epoch, _ in epoch_events] != [1]:
        raise ValueError("a run without checkpoints may contain only epoch-0001")
    if epoch_events:
        _validate_precheckpoint_epoch_event(
            epoch_events[0][1],
            config=config,
            expected_first_epoch_step=expected_first_epoch_step,
        )
    prior_recovery_evidence = _has_precheckpoint_recovery_evidence(
        run_directory,
        config=config,
        expected_first_epoch_step=expected_first_epoch_step,
    )
    if not epoch_events and not prior_recovery_evidence:
        raise FileExistsError(
            "run directory already exists without recoverable pre-checkpoint evidence"
        )

    _quarantine_unpublished_checkpoint_staging(
        run_directory, checkpoint_staging
    )
    _quarantine_uncommitted_epoch_events(
        run_directory, [path for _, path in epoch_events]
    )
    return baseline, metrics


def _restore_run_checkpoint(
    checkpoint: str | Path,
    *,
    run_directory: Path,
    trainer: ChmtQTrainer,
    config: ExperimentConfig,
    scaler: FoldBoundScalerArtifact,
    split: WalkForwardManifest,
    baseline: EmpiricalBaseline,
) -> dict[str, Any]:
    path = Path(checkpoint).resolve()
    checkpoints_root = (run_directory / "checkpoints").resolve()
    if path.parent != checkpoints_root:
        raise ValueError("resume checkpoint must belong to this run directory")
    available = sorted(
        item.resolve()
        for item in checkpoints_root.glob("step-*")
        if item.is_dir() and not item.is_symlink()
    )
    if not available or path != available[-1]:
        raise ValueError("resume requires the latest immutable checkpoint")
    restored = load_checkpoint(
        path,
        model=trainer.model,
        optimizer=trainer.optimizer,
        expected_identity=config.checkpoint_identity(),
    )
    state = restored["trainer_state"]
    if not isinstance(state, dict) or set(state) != {"trainer", "experiment"}:
        raise ValueError("checkpoint does not contain experiment-runner state")
    trainer_state = state["trainer"]
    experiment = state["experiment"]
    if not isinstance(trainer_state, dict) or not isinstance(experiment, dict):
        raise TypeError("checkpoint runner state must contain objects")
    trainer.load_state_dict(trainer_state)
    required = {
        "schema_version",
        "completed_epoch",
        "next_epoch",
        "best_epoch",
        "best_metric",
        "best_tie_breaker",
        "best_checkpoint",
        "validations_without_improvement",
        "stopped_early",
        "epoch_event",
        "epoch_event_sha256",
        "scaler_artifact_sha256",
        "split_manifest_sha256",
        "fold_sha256",
        "baseline_artifact_sha256",
    }
    if set(experiment) != required or experiment["schema_version"] != _RUN_SCHEMA_VERSION:
        raise ValueError("checkpoint experiment state field set is invalid")
    completed = _non_negative_int(experiment["completed_epoch"], "completed_epoch")
    next_epoch = _positive_int(experiment["next_epoch"], "next_epoch")
    if next_epoch != completed + 1 or next_epoch > config.max_epochs + 1:
        raise ValueError("checkpoint epoch cursor is invalid")
    if experiment["scaler_artifact_sha256"] != scaler.artifact_sha256:
        raise ValueError("checkpoint scaler identity differs")
    if experiment["split_manifest_sha256"] != split.artifact_sha256:
        raise ValueError("checkpoint split identity differs")
    if experiment["baseline_artifact_sha256"] != baseline.artifact_sha256:
        raise ValueError("checkpoint baseline identity differs")
    fold_number = int(config.fold_id.removeprefix("fold-"))
    if experiment["fold_sha256"] != split.fold(fold_number).fold_sha256:
        raise ValueError("checkpoint fold identity differs")
    event_path = run_directory / _safe_relative(experiment["epoch_event"])
    if _file_sha256(event_path) != experiment["epoch_event_sha256"]:
        raise ValueError("checkpoint epoch event hash differs")
    expected_event_path = (
        run_directory / "epochs" / f"epoch-{completed:04d}.json"
    ).resolve()
    if event_path.resolve() != expected_event_path:
        raise ValueError("checkpoint epoch event does not match its completed epoch")
    epoch_files = _ordered_epoch_events(run_directory)
    if event_path.resolve() not in {item.resolve() for _, item in epoch_files}:
        raise ValueError("checkpoint epoch event is missing from the run")
    event_payload = _read_artifact_json(event_path)
    if (
        not isinstance(event_payload, dict)
        or event_payload.get("epoch") != completed
        or event_payload.get("optimizer_step") != restored["step"]
    ):
        raise ValueError("checkpoint epoch event payload differs from checkpoint state")
    best_checkpoint = experiment["best_checkpoint"]
    pending_current_checkpoint = best_checkpoint == "pending-current-checkpoint"
    if pending_current_checkpoint:
        best_checkpoint = _relative_path(path, run_directory)
    result = dict(experiment)
    result["best_checkpoint"] = _safe_relative(best_checkpoint).as_posix()
    result["best_epoch"] = _positive_int(result["best_epoch"], "best_epoch")
    result["best_metric"] = _finite_float(result["best_metric"], "best_metric")
    result["best_tie_breaker"] = _finite_float(
        result["best_tie_breaker"], "best_tie_breaker"
    )
    result["validations_without_improvement"] = _non_negative_int(
        result["validations_without_improvement"],
        "validations_without_improvement",
    )
    if not isinstance(result["stopped_early"], bool):
        raise TypeError("stopped_early must be a boolean")
    result["next_epoch"] = next_epoch
    _quarantine_uncommitted_epoch_events(
        run_directory,
        [item for epoch, item in epoch_files if epoch > completed],
    )
    if pending_current_checkpoint:
        _ensure_selection_record(
            run_directory,
            epoch=result["best_epoch"],
            checkpoint=path,
            metric=result["best_metric"],
            tie_breaker=result["best_tie_breaker"],
        )
    return result


def _patch_checkpoint_best_path(checkpoint: Path, run_directory: Path) -> None:
    """Checkpoint state is immutable; current-best resolution is stored separately."""

    # The checkpoint intentionally retains the sentinel when the current epoch first wins.
    # Resume resolves that sentinel to the checkpoint directory itself after hash verification.
    if checkpoint.parent != run_directory / "checkpoints":
        raise ValueError("selected checkpoint is outside the run directory")


def _write_epoch_event(
    run_directory: Path, epoch: int, payload: dict[str, Any]
) -> tuple[Path, str]:
    path = run_directory / "epochs" / f"epoch-{epoch:04d}.json"
    content = _artifact_json(payload) + "\n"
    _write_new_text(path, content)
    return path, _file_sha256(path)


def _write_selection_record(
    run_directory: Path,
    *,
    epoch: int,
    checkpoint: Path,
    metric: float,
    tie_breaker: float,
) -> None:
    _ensure_selection_record(
        run_directory,
        epoch=epoch,
        checkpoint=checkpoint,
        metric=metric,
        tie_breaker=tie_breaker,
    )


def _ensure_selection_record(
    run_directory: Path,
    *,
    epoch: int,
    checkpoint: Path,
    metric: float,
    tie_breaker: float,
) -> None:
    path = run_directory / "selections" / f"best-at-epoch-{epoch:04d}.json"
    content = (
        _artifact_json(
            {
                "epoch": epoch,
                "checkpoint": _relative_path(checkpoint, run_directory),
                "metric": metric,
                "tie_breaker": tie_breaker,
            }
        )
        + "\n"
    )
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("selection record path is unsafe")
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("selection record differs from checkpoint state")
        return
    _write_new_text(path, content)


def _ordered_epoch_events(run_directory: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in (run_directory / "epochs").glob("epoch-*.json"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("epoch event path is unsafe")
        suffix = path.name.removeprefix("epoch-").removesuffix(".json")
        if len(suffix) != 4 or not suffix.isascii() or not suffix.isdigit():
            raise ValueError("epoch event filename is invalid")
        epoch = int(suffix)
        if epoch <= 0:
            raise ValueError("epoch event number must be positive")
        result.append((epoch, path))
    result.sort(key=lambda item: item[0])
    if len({epoch for epoch, _ in result}) != len(result):  # pragma: no cover - names unique
        raise ValueError("duplicate epoch event number")
    return result


def _validate_precheckpoint_epoch_event(
    path: Path,
    *,
    config: ExperimentConfig,
    expected_first_epoch_step: int,
) -> None:
    payload = _read_artifact_json(path)
    if not isinstance(payload, dict):
        raise TypeError("pre-checkpoint epoch event payload must be an object")
    expected = {
        "schema_version": _RUN_SCHEMA_VERSION,
        "run_id": config.run_id,
        "fold_id": config.fold_id,
        "epoch": 1,
        "optimizer_step": expected_first_epoch_step,
    }
    if any(payload.get(name) != value for name, value in expected.items()):
        raise ValueError("pre-checkpoint epoch event differs from the frozen run")


def _has_precheckpoint_recovery_evidence(
    run_directory: Path,
    *,
    config: ExperimentConfig,
    expected_first_epoch_step: int,
) -> bool:
    root = run_directory / "quarantine" / "uncommitted-epoch-events"
    if not root.exists() and not root.is_symlink():
        return False
    if root.is_symlink() or not root.is_dir():
        raise ValueError("epoch-event quarantine path is unsafe")
    evidence = False
    prefix = "epoch-0001-"
    suffix = ".json"
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file():
            raise ValueError("quarantined epoch event path is unsafe")
        name = path.name
        digest = name[len(prefix) : -len(suffix)]
        if (
            not name.startswith(prefix)
            or not name.endswith(suffix)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("pre-checkpoint quarantine contains an unknown artifact")
        if _file_sha256(path) != digest:
            raise ValueError("quarantined epoch event digest differs")
        _validate_precheckpoint_epoch_event(
            path,
            config=config,
            expected_first_epoch_step=expected_first_epoch_step,
        )
        evidence = True
    return evidence


def _checkpoint_staging_step(path: Path) -> int | None:
    if path.is_symlink() or not path.is_dir():
        return None
    prefix, separator, nonce = path.name.partition(".staging-")
    step = prefix.removeprefix(".step-")
    if (
        not separator
        or not prefix.startswith(".step-")
        or len(step) != 8
        or not step.isascii()
        or not step.isdigit()
        or len(nonce) != 32
        or any(character not in "0123456789abcdef" for character in nonce)
    ):
        return None
    for item in path.rglob("*"):
        if item.is_symlink() or not (item.is_file() or item.is_dir()):
            raise ValueError("unpublished checkpoint staging path is unsafe")
    return int(step)


def _quarantine_unpublished_checkpoint_staging(
    run_directory: Path, entries: list[Path]
) -> None:
    if not entries:
        return
    quarantine_parent = run_directory / "quarantine"
    quarantine_root = quarantine_parent / "unpublished-checkpoints"
    for directory in (quarantine_parent, quarantine_root):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise ValueError("checkpoint quarantine path is unsafe")
        directory.mkdir(exist_ok=True)
    checkpoints_root = (run_directory / "checkpoints").resolve()
    for source in entries:
        if source.parent.resolve() != checkpoints_root:
            raise ValueError("unpublished checkpoint staging is outside the run")
        if _checkpoint_staging_step(source) is None:
            raise ValueError("unpublished checkpoint staging path is unsafe")
        target = quarantine_root / source.name
        if target.exists() or target.is_symlink():
            raise FileExistsError("unpublished checkpoint quarantine target exists")
        os.replace(source, target)


def _quarantine_uncommitted_epoch_events(
    run_directory: Path, events: list[Path]
) -> None:
    if not events:
        return
    quarantine_parent = run_directory / "quarantine"
    quarantine_root = quarantine_parent / "uncommitted-epoch-events"
    for directory in (quarantine_parent, quarantine_root):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise ValueError("epoch-event quarantine path is unsafe")
        directory.mkdir(exist_ok=True)
    for source in events:
        if source.parent.resolve() != (run_directory / "epochs").resolve():
            raise ValueError("uncommitted epoch event is outside the run")
        if source.is_symlink() or not source.is_file():
            raise ValueError("uncommitted epoch event path is unsafe")
        digest = _file_sha256(source)
        target = quarantine_root / f"{source.stem}-{digest}.json"
        if target.exists() or target.is_symlink():
            if target.is_symlink() or not target.is_file():
                raise ValueError("quarantined epoch event path is unsafe")
            if _file_sha256(target) != digest:
                raise ValueError("quarantined epoch event content differs")
        else:
            try:
                os.link(source, target)
            except FileExistsError:
                if target.is_symlink() or _file_sha256(target) != digest:
                    raise ValueError("quarantined epoch event content differs")
        source.unlink()


def _gate_means(
    model: ChmtQTransformer, batch: ChmtQBatch, device: torch.device
) -> list[float]:
    moved = _move_batch(batch, device)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            gates = model.modality_gates(moved)
        return [float(value) for value in gates.float().mean(dim=0).cpu()]
    finally:
        model.train(was_training)


def _attention_payload(values: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "max_abs_by_head": list(item.max_abs_by_head),
            "observations": item.observations,
            "runaway_threshold": item.runaway_threshold,
        }
        for name, item in sorted(values.items())
    }


def _primary_metric(report: ChmtQMetricReport, name: str) -> float:
    if name == "weighted_pinball":
        value = report.weighted_primary_metric
    elif name == "one_step_poc_mae":
        value = float(report.median_mae[0, 0])
        if int(report.forecast_counts[0, 0]) <= 0:
            raise ValueError("one-step POC metric has no supervised examples")
    else:  # pragma: no cover - ExperimentConfig validates this
        raise ValueError(f"unsupported primary metric: {name}")
    return _finite_float(value, name)


def _selection_metrics(
    evaluation: ChmtQEvaluationResult, metric_name: str
) -> ChmtQMetricReport:
    if metric_name == "one_step_poc_mae":
        if evaluation.physical_metrics is None:
            raise ValueError("one-step POC MAE selection requires physical-unit metrics")
        return evaluation.physical_metrics
    return evaluation.metrics


def _tie_breaker(report: ChmtQMetricReport) -> float:
    counts = report.forecast_counts.to(torch.float64)
    denominator = float(counts.sum())
    if denominator <= 0.0:
        raise ValueError("validation metrics contain no forecast labels")
    value = float((report.median_mae * counts).sum() / denominator)
    return _finite_float(value, "validation median MAE")


def _is_improved(
    metric: float,
    tie_breaker: float,
    best_metric: float | None,
    best_tie_breaker: float | None,
    minimum_improvement: float,
) -> bool:
    if best_metric is None:
        return True
    if metric < best_metric - minimum_improvement:
        return True
    return (
        abs(metric - best_metric) <= minimum_improvement
        and best_tie_breaker is not None
        and tie_breaker < best_tie_breaker
    )


def _metric_report_from_dict(value: Any) -> ChmtQMetricReport:
    if not isinstance(value, dict):
        raise TypeError("metric artifact payload must be an object")
    expected = {
        "forecast_counts",
        "pinball",
        "median_mae",
        "median_rmse",
        "median_bias",
        "interval_coverage",
        "interval_width",
        "presence_counts",
        "presence_log_loss",
        "presence_brier",
        "weighted_primary_metric",
    }
    if set(value) != expected:
        raise ValueError("metric artifact field set is invalid")
    return ChmtQMetricReport(
        forecast_counts=torch.tensor(value["forecast_counts"], dtype=torch.int64),
        pinball=torch.tensor(value["pinball"], dtype=torch.float64),
        median_mae=torch.tensor(value["median_mae"], dtype=torch.float64),
        median_rmse=torch.tensor(value["median_rmse"], dtype=torch.float64),
        median_bias=torch.tensor(value["median_bias"], dtype=torch.float64),
        interval_coverage=torch.tensor(value["interval_coverage"], dtype=torch.float64),
        interval_width=torch.tensor(value["interval_width"], dtype=torch.float64),
        presence_counts=torch.tensor(value["presence_counts"], dtype=torch.int64),
        presence_log_loss=torch.tensor(
            value["presence_log_loss"], dtype=torch.float64
        ),
        presence_brier=torch.tensor(value["presence_brier"], dtype=torch.float64),
        weighted_primary_metric=_finite_float(
            value["weighted_primary_metric"], "weighted_primary_metric"
        ),
    )


def _metric_suite_from_dict(value: Any) -> dict[str, _BaselineMetrics]:
    if not isinstance(value, dict) or set(value) != {
        "empirical_marginal",
        "zero_residual",
    }:
        raise ValueError("baseline metric suite field set is invalid")
    result: dict[str, _BaselineMetrics] = {}
    for name, report in value.items():
        if not isinstance(report, dict) or set(report) != {"normalized", "physical"}:
            raise ValueError("baseline metric unit field set is invalid")
        result[name] = _BaselineMetrics(
            normalized=_metric_report_from_dict(report["normalized"]),
            physical=_metric_report_from_dict(report["physical"]),
        )
    return result


def _best_baseline_entry(
    reports: dict[str, _BaselineMetrics], metric_name: str
) -> tuple[str, ChmtQMetricReport]:
    if not reports:
        raise ValueError("baseline metric suite is empty")
    def selected(pair: _BaselineMetrics) -> ChmtQMetricReport:
        return pair.physical if metric_name == "one_step_poc_mae" else pair.normalized

    name, pair = min(
        reports.items(), key=lambda item: _primary_metric(selected(item[1]), metric_name)
    )
    return name, selected(pair)


def _empty_mask_totals() -> dict[str, list[int]]:
    return {
        "bars": [0, 0],
        "l1": [0, 0],
        "l2_levels": [0, 0],
        "l2_snapshots": [0, 0],
        "targets": [0, 0],
        "presence": [0, 0],
    }


def _add_mask_totals(values: dict[str, list[int]], batch: ChmtQBatch) -> None:
    masks = {
        "bars": batch.bars_mask,
        "l1": batch.l1_mask,
        "l2_levels": batch.l2_level_mask,
        "l2_snapshots": batch.l2_snapshot_mask,
        "targets": batch.target_mask,
        "presence": batch.presence_mask,
    }
    for name, mask in masks.items():
        if mask is None:
            continue
        values[name][0] += int(mask.sum())
        values[name][1] += mask.numel()


def _mask_fractions(values: dict[str, list[int]]) -> dict[str, float]:
    return {
        name: valid / total if total else 0.0
        for name, (valid, total) in values.items()
    }


def _move_batch(batch: ChmtQBatch, device: torch.device) -> ChmtQBatch:
    def move(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value.to(device=device)

    return ChmtQBatch(
        bars=move(batch.bars),  # type: ignore[arg-type]
        bars_mask=move(batch.bars_mask),  # type: ignore[arg-type]
        l1=move(batch.l1),  # type: ignore[arg-type]
        l1_mask=move(batch.l1_mask),  # type: ignore[arg-type]
        l2=move(batch.l2),  # type: ignore[arg-type]
        l2_level_mask=move(batch.l2_level_mask),  # type: ignore[arg-type]
        l2_snapshot_mask=move(batch.l2_snapshot_mask),  # type: ignore[arg-type]
        quality=move(batch.quality),  # type: ignore[arg-type]
        targets=move(batch.targets),
        target_mask=move(batch.target_mask),
        presence_labels=move(batch.presence_labels),
        presence_mask=move(batch.presence_mask),
    )


def _run_directory(output_root: str | Path, run_id: str) -> Path:
    root = Path(output_root).resolve()
    return root / run_id


def _environment_payload(
    device: torch.device, config: ExperimentConfig
) -> dict[str, Any]:
    import numpy as np

    cuda_device_name: str | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        cuda_device_name = torch.cuda.get_device_name(index)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_device_name": cuda_device_name,
        "device": str(device),
        "deterministic": config.training.deterministic,
        "mixed_precision": config.training.mixed_precision,
    }


def _epoch_seed(seed: int, fold_sha256: str, epoch: int) -> int:
    payload = f"{seed}:{fold_sha256}:{epoch}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _artifact_json(payload: Any) -> str:
    encoded = _canonical_json(payload)
    return _canonical_json(
        {
            "artifact_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "payload": payload,
        }
    )


def _read_artifact_json(path: Path) -> Any:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run artifact: {path.name}") from error
    if not isinstance(document, dict) or set(document) != {"artifact_sha256", "payload"}:
        raise ValueError(f"run artifact field set is invalid: {path.name}")
    encoded = _canonical_json(document["payload"])
    actual = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if document["artifact_sha256"] != actual:
        raise ValueError(f"run artifact SHA-256 differs: {path.name}")
    return document["payload"]


def _write_new_text(path: Path, content: str) -> None:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"artifact already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
    try:
        staging.write_text(content, encoding="utf-8", newline="\n")
        try:
            os.link(staging, target)
        except FileExistsError as error:
            raise FileExistsError(f"artifact already exists: {target}") from error
    finally:
        if staging.exists() or staging.is_symlink():
            staging.unlink()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _safe_relative(value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise TypeError("artifact path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("artifact path must be a safe relative path")
    return path


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
