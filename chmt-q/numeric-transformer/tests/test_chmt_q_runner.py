from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.data.dataset import (
    ArrowDataset,
    DatasetCapabilities,
    EngineeredExample,
    ExampleMetadata,
    write_engineered_fixture_dataset,
)
from daxalgo_numeric_transformer.chmt_q.data.scaler_binding import (
    fit_fold_bound_scaler,
)
from daxalgo_numeric_transformer.chmt_q.dependencies import (
    verify_runtime_dependency_lock,
    write_runtime_dependency_lock,
)
from daxalgo_numeric_transformer.chmt_q.experiment import (
    ExperimentConfig,
    source_tree_sha256,
)
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch
from daxalgo_numeric_transformer.chmt_q.training.config import TrainingConfig
from daxalgo_numeric_transformer.chmt_q.training.runner import (
    evaluate_development_checkpoint,
    run_development_experiment,
)
from daxalgo_numeric_transformer.chmt_q.training.walk_forward import (
    build_expanding_walk_forward,
)


def _ready_dataset(root):
    config = ChmtQConfig.tiny()
    examples = []
    for index in range(4):
        batch = make_chmt_q_synthetic_batch(
            config, batch_size=1, seed=700 + index
        )
        batch = replace(
            batch,
            target_mask=torch.ones_like(batch.target_mask),
            presence_mask=torch.ones_like(batch.presence_mask),
        )
        examples.append(
            EngineeredExample(
                metadata=ExampleMetadata(
                    example_id=f"e{index}",
                    session_id=f"s{index}",
                    feature_cutoff_ns=index * 1_000 + 90,
                    decision_ns=index * 1_000 + 100,
                    dependency_start_ns=index * 1_000,
                    dependency_end_ns=index * 1_000 + 200,
                    instrument_id="BTC-USD",
                    anchor_source="l1_mid",
                    anchor_price=50_000.0 + index,
                    tick_size=0.5,
                    log_volume_reference=1.0,
                    provenance_sha256=f"{index + 1:064x}",
                ),
                batch=batch,
            )
        )
    capabilities = DatasetCapabilities(
        training_ready=True,
        rights_confirmed=True,
        provenance_complete=True,
        readiness_evidence_sha256="a" * 64,
        rights_evidence_sha256="b" * 64,
        provenance_evidence_sha256="c" * 64,
    )
    manifest = write_engineered_fixture_dataset(
        root,
        examples,
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=capabilities,
        rows_per_shard=2,
    )
    return ArrowDataset.open(root, expected_dataset_id=manifest.dataset_id), config


def _ready_run(tmp_path, *, max_epochs=2):
    dataset, model_config = _ready_dataset(tmp_path / "dataset")
    split = build_expanding_walk_forward(
        dataset.dependencies(),
        source_dataset_hash=dataset.manifest.dataset_id,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    fold = split.fold(1)
    scaler = fit_fold_bound_scaler(
        dataset,
        dataset_id=dataset.manifest.dataset_id,
        split_manifest_sha256=split.artifact_sha256,
        fold_id="fold-01",
        ordered_train_example_ids=fold.train_example_ids,
        batch_size=2,
        config=model_config,
    )
    source_root = tmp_path / "source"
    package = source_root / "daxalgo_numeric_transformer"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    dependency_lock = source_root / "dependencies.lock"
    write_runtime_dependency_lock(dependency_lock)
    config = ExperimentConfig(
        dataset_id=dataset.manifest.dataset_id,
        instrument_id=dataset.manifest.instrument_id,
        split_manifest_sha256=split.artifact_sha256,
        scaler_artifact_sha256=scaler.artifact_sha256,
        source_tree_sha256=source_tree_sha256(source_root),
        dependency_lock_sha256=verify_runtime_dependency_lock(dependency_lock),
        fold_id="fold-01",
        seed=17,
        model=model_config,
        training=TrainingConfig(seed=17, micro_batch_size=1, accumulation_steps=1),
        max_epochs=max_epochs,
        early_stop_patience=max_epochs + 1,
    )
    return dataset, split, scaler, config, source_root, dependency_lock


def test_development_runner_resumes_at_epoch_boundary_and_keeps_holdout_sealed(
    tmp_path, monkeypatch
) -> None:
    dataset, model_config = _ready_dataset(tmp_path / "dataset")
    split = build_expanding_walk_forward(
        dataset.dependencies(),
        source_dataset_hash=dataset.manifest.dataset_id,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    fold = split.fold(1)
    scaler = fit_fold_bound_scaler(
        dataset,
        dataset_id=dataset.manifest.dataset_id,
        split_manifest_sha256=split.artifact_sha256,
        fold_id="fold-01",
        ordered_train_example_ids=fold.train_example_ids,
        batch_size=2,
        config=model_config,
    )
    source_root = tmp_path / "source"
    package = source_root / "daxalgo_numeric_transformer"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    dependency_lock = source_root / "dependencies.lock"
    write_runtime_dependency_lock(dependency_lock)
    config = ExperimentConfig(
        dataset_id=dataset.manifest.dataset_id,
        instrument_id=dataset.manifest.instrument_id,
        split_manifest_sha256=split.artifact_sha256,
        scaler_artifact_sha256=scaler.artifact_sha256,
        source_tree_sha256=source_tree_sha256(source_root),
        dependency_lock_sha256=verify_runtime_dependency_lock(dependency_lock),
        fold_id="fold-01",
        seed=17,
        model=model_config,
        training=TrainingConfig(
            seed=17,
            micro_batch_size=1,
            accumulation_steps=1,
        ),
        max_epochs=2,
        early_stop_patience=2,
    )
    substitute_scaler = replace(
        scaler,
        scaler=replace(
            scaler.scaler,
            minimum_scale=scaler.scaler.minimum_scale * 2.0,
        ),
    )

    with pytest.raises(ValueError, match="scaler identity"):
        run_development_experiment(
            dataset,
            split,
            substitute_scaler,
            config,
            fold_number=1,
            output_root=tmp_path / "substitute-runs",
            source_root=source_root,
            dependency_lock=dependency_lock,
        )

    from daxalgo_numeric_transformer.chmt_q.training import runner

    original = runner._train_epoch
    calls = 0

    def interrupt_second_epoch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_train_epoch", interrupt_second_epoch)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_development_experiment(
            dataset,
            split,
            scaler,
            config,
            fold_number=1,
            output_root=tmp_path / "runs",
            source_root=source_root,
            dependency_lock=dependency_lock,
        )

    monkeypatch.setattr(runner, "_train_epoch", original)
    run_directory = tmp_path / "runs" / config.run_id
    checkpoint = next((run_directory / "checkpoints").glob("step-*"))
    result = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
        resume_checkpoint=checkpoint,
    )

    assert result.epochs_completed == 2
    assert result.optimizer_steps == 4
    assert result.best_checkpoint.is_dir()
    assert (run_directory / "summary.json").is_file()
    assert not any(
        '"e3"' in path.read_text(encoding="utf-8")
        for path in (run_directory / "epochs").glob("*.json")
    )
    evaluation = evaluate_development_checkpoint(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        checkpoint=result.best_checkpoint,
        source_root=source_root,
        dependency_lock=dependency_lock,
    )
    assert evaluation.run_id == config.run_id
    assert evaluation.checkpoint_step in {2, 4}
    assert torch.isfinite(torch.tensor(evaluation.primary_metric))
    with pytest.raises(ValueError, match="scaler identity"):
        evaluate_development_checkpoint(
            dataset,
            split,
            substitute_scaler,
            config,
            fold_number=1,
            checkpoint=result.best_checkpoint,
            source_root=source_root,
            dependency_lock=dependency_lock,
        )


def test_resume_from_early_stop_checkpoint_does_not_train_another_epoch(
    tmp_path, monkeypatch
) -> None:
    dataset, model_config = _ready_dataset(tmp_path / "dataset")
    split = build_expanding_walk_forward(
        dataset.dependencies(),
        source_dataset_hash=dataset.manifest.dataset_id,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    fold = split.fold(1)
    scaler = fit_fold_bound_scaler(
        dataset,
        dataset_id=dataset.manifest.dataset_id,
        split_manifest_sha256=split.artifact_sha256,
        fold_id="fold-01",
        ordered_train_example_ids=fold.train_example_ids,
        batch_size=2,
        config=model_config,
    )
    source_root = tmp_path / "source"
    package = source_root / "daxalgo_numeric_transformer"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    dependency_lock = source_root / "dependencies.lock"
    write_runtime_dependency_lock(dependency_lock)
    config = ExperimentConfig(
        dataset_id=dataset.manifest.dataset_id,
        instrument_id=dataset.manifest.instrument_id,
        split_manifest_sha256=split.artifact_sha256,
        scaler_artifact_sha256=scaler.artifact_sha256,
        source_tree_sha256=source_tree_sha256(source_root),
        dependency_lock_sha256=verify_runtime_dependency_lock(dependency_lock),
        fold_id="fold-01",
        seed=17,
        model=model_config,
        training=TrainingConfig(seed=17, micro_batch_size=1, accumulation_steps=1),
        max_epochs=4,
        early_stop_patience=1,
    )

    from daxalgo_numeric_transformer.chmt_q.training import runner

    decisions = iter((True, False))
    monkeypatch.setattr(runner, "_is_improved", lambda *_args: next(decisions))
    original_write = runner._write_new_text

    def interrupt_summary(path, content):
        if path.name == "summary.json":
            raise RuntimeError("simulated post-checkpoint interruption")
        return original_write(path, content)

    monkeypatch.setattr(runner, "_write_new_text", interrupt_summary)
    with pytest.raises(RuntimeError, match="post-checkpoint"):
        run_development_experiment(
            dataset,
            split,
            scaler,
            config,
            fold_number=1,
            output_root=tmp_path / "runs",
            source_root=source_root,
            dependency_lock=dependency_lock,
        )

    run_directory = tmp_path / "runs" / config.run_id
    checkpoints = sorted((run_directory / "checkpoints").glob("step-*"))
    assert len(checkpoints) == 2
    monkeypatch.setattr(runner, "_write_new_text", original_write)

    def reject_extra_training(*_args, **_kwargs):
        raise AssertionError("early-stop resume attempted another training epoch")

    monkeypatch.setattr(runner, "_train_epoch", reject_extra_training)
    result = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
        resume_checkpoint=checkpoints[-1],
    )

    assert result.stopped_early
    assert result.epochs_completed == 2
    assert result.optimizer_steps == 4


def test_resume_quarantines_epoch_event_published_before_checkpoint(
    tmp_path, monkeypatch
) -> None:
    dataset, split, scaler, config, source_root, dependency_lock = _ready_run(
        tmp_path
    )
    from daxalgo_numeric_transformer.chmt_q.training import runner

    original_save = runner.save_checkpoint
    original_evaluate = runner.evaluate_model
    save_calls = 0
    streamed_validations = 0

    def interrupt_second_checkpoint(*args, **kwargs):
        nonlocal save_calls
        save_calls += 1
        if save_calls == 2:
            raise RuntimeError("simulated event-checkpoint gap")
        return original_save(*args, **kwargs)

    def require_streaming_validation(model, batches, **kwargs):
        nonlocal streamed_validations
        streamed_validations += 1
        assert not isinstance(batches, (list, tuple))
        return original_evaluate(model, batches, **kwargs)

    monkeypatch.setattr(runner, "save_checkpoint", interrupt_second_checkpoint)
    monkeypatch.setattr(runner, "evaluate_model", require_streaming_validation)
    with pytest.raises(RuntimeError, match="event-checkpoint gap"):
        run_development_experiment(
            dataset,
            split,
            scaler,
            config,
            fold_number=1,
            output_root=tmp_path / "runs",
            source_root=source_root,
            dependency_lock=dependency_lock,
        )

    run_directory = tmp_path / "runs" / config.run_id
    uncommitted_event = run_directory / "epochs" / "epoch-0002.json"
    uncommitted_content = uncommitted_event.read_bytes()
    checkpoints = sorted((run_directory / "checkpoints").glob("step-*"))
    assert len(checkpoints) == 1

    monkeypatch.setattr(runner, "save_checkpoint", original_save)
    result = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
        resume_checkpoint=checkpoints[0],
    )

    quarantined = list(
        (run_directory / "quarantine" / "uncommitted-epoch-events").glob(
            "epoch-0002-*.json"
        )
    )
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == uncommitted_content
    assert uncommitted_event.is_file()
    event_payload = runner._read_artifact_json(uncommitted_event)
    assert len(event_payload["telemetry"]["modality_gate_means"]) == 3
    assert streamed_validations == 3
    assert result.epochs_completed == 2
    assert result.optimizer_steps == 4


def test_resume_reconciles_checkpoint_before_selection_record(
    tmp_path, monkeypatch
) -> None:
    dataset, split, scaler, config, source_root, dependency_lock = _ready_run(
        tmp_path
    )
    from daxalgo_numeric_transformer.chmt_q.training import runner

    original_selection = runner._write_selection_record

    def interrupt_selection(*_args, **_kwargs):
        raise RuntimeError("simulated checkpoint-selection gap")

    monkeypatch.setattr(runner, "_write_selection_record", interrupt_selection)
    with pytest.raises(RuntimeError, match="checkpoint-selection gap"):
        run_development_experiment(
            dataset,
            split,
            scaler,
            config,
            fold_number=1,
            output_root=tmp_path / "runs",
            source_root=source_root,
            dependency_lock=dependency_lock,
        )

    run_directory = tmp_path / "runs" / config.run_id
    checkpoint = next((run_directory / "checkpoints").glob("step-*"))
    selection = run_directory / "selections" / "best-at-epoch-0001.json"
    assert not selection.exists()

    monkeypatch.setattr(runner, "_write_selection_record", original_selection)
    result = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
        resume_checkpoint=checkpoint,
    )

    selection_payload = runner._read_artifact_json(selection)
    assert selection_payload["epoch"] == 1
    assert selection_payload["checkpoint"] == f"checkpoints/{checkpoint.name}"
    assert result.epochs_completed == 2


def test_restart_recovers_first_epoch_event_without_a_checkpoint(
    tmp_path, monkeypatch
) -> None:
    dataset, split, scaler, config, source_root, dependency_lock = _ready_run(
        tmp_path
    )
    from daxalgo_numeric_transformer.chmt_q.training import runner

    original_save = runner.save_checkpoint

    def interrupt_first_checkpoint(root, *args, **kwargs):
        staging = root / (".step-00000002.staging-" + "f" * 32)
        staging.mkdir()
        (staging / "partial-state").write_bytes(b"governed partial checkpoint")
        raise RuntimeError("simulated first checkpoint publication failure")

    monkeypatch.setattr(runner, "save_checkpoint", interrupt_first_checkpoint)
    with pytest.raises(RuntimeError, match="first checkpoint publication"):
        run_development_experiment(
            dataset,
            split,
            scaler,
            config,
            fold_number=1,
            output_root=tmp_path / "runs",
            source_root=source_root,
            dependency_lock=dependency_lock,
        )

    run_directory = tmp_path / "runs" / config.run_id
    orphan = run_directory / "epochs" / "epoch-0001.json"
    orphan_content = orphan.read_bytes()
    assert not any((run_directory / "checkpoints").glob("step-*"))

    monkeypatch.setattr(runner, "save_checkpoint", original_save)
    recovered = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
    )
    uninterrupted = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "control-runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
    )

    quarantined_events = list(
        (run_directory / "quarantine" / "uncommitted-epoch-events").glob(
            "epoch-0001-*.json"
        )
    )
    assert len(quarantined_events) == 1
    assert quarantined_events[0].read_bytes() == orphan_content
    quarantined_staging = (
        run_directory
        / "quarantine"
        / "unpublished-checkpoints"
        / (".step-00000002.staging-" + "f" * 32)
        / "partial-state"
    )
    assert quarantined_staging.read_bytes() == b"governed partial checkpoint"
    assert recovered.epochs_completed == uninterrupted.epochs_completed == 2
    assert recovered.optimizer_steps == uninterrupted.optimizer_steps == 4
    assert recovered.best_epoch == uninterrupted.best_epoch
    assert recovered.best_metric == uninterrupted.best_metric

    with (
        np.load(recovered.best_checkpoint / "weights.npz", allow_pickle=False) as left,
        np.load(
            uninterrupted.best_checkpoint / "weights.npz", allow_pickle=False
        ) as right,
    ):
        assert left.files == right.files
        for name in left.files:
            np.testing.assert_array_equal(left[name], right[name])
