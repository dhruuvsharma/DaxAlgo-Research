from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.data.dataset import (
    DatasetCapabilities,
    EngineeredExample,
    ExampleMetadata,
    write_engineered_fixture_dataset,
)
from daxalgo_numeric_transformer.chmt_q.data.scaler_binding import (
    FoldBoundScalerArtifact,
)
from daxalgo_numeric_transformer.chmt_q.experiment import ExperimentConfig
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch
from daxalgo_numeric_transformer.chmt_q.training.walk_forward import (
    WalkForwardManifest,
)
from daxalgo_numeric_transformer.cli import main


def _dataset(root, *, ready: bool = True):
    config = ChmtQConfig.tiny()
    examples = []
    for index in range(5):
        batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=900 + index)
        examples.append(
            EngineeredExample(
                ExampleMetadata(
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
                replace(
                    batch,
                    target_mask=torch.ones_like(batch.target_mask),
                    presence_mask=torch.ones_like(batch.presence_mask),
                ),
            )
        )
    return write_engineered_fixture_dataset(
        root,
        examples,
        source_manifest_sha256="a" * 64,
        builder_sha256="b" * 64,
        capabilities=DatasetCapabilities(
            training_ready=ready,
            rights_confirmed=ready,
            provenance_complete=ready,
            readiness_evidence_sha256="c" * 64,
            rights_evidence_sha256="d" * 64,
            provenance_evidence_sha256="e" * 64,
        ),
    )


def _write_compiler_report(root: Path, manifest) -> None:
    report = {
        "schema_version": 1,
        "status": "compiled_non_promotable",
        "dataset_id": manifest.dataset_id,
        "dataset_rows": manifest.row_count,
        "instrument_id": manifest.instrument_id,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "capabilities": manifest.capabilities.to_payload(),
    }
    (root / "okx-compiler-report.json").write_text(
        json.dumps(report, sort_keys=True), encoding="utf-8"
    )


def test_cli_prepares_verified_reference_experiment_artifacts(tmp_path, capsys) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root)
    split_path = tmp_path / "split.json"
    scaler_path = tmp_path / "scaler.json"
    config_path = tmp_path / "experiment.json"
    source_root = tmp_path / "source"
    package = source_root / "daxalgo_numeric_transformer"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    dependency_lock = source_root / "dependencies.lock"

    assert main(["lock-environment", "--output", str(dependency_lock)]) == 0
    lock_summary = json.loads(capsys.readouterr().out)
    assert lock_summary["dependency_lock_sha256"]

    assert (
        main(
            [
                "verify-dataset",
                "--input",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--instrument-id",
                "BTC-USD",
            ]
        )
        == 0
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["chmt_q_training_ready"]

    assert (
        main(
            [
                "build-split",
                "--dataset",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--initial-train-sessions",
                "2",
                "--validation-sessions",
                "1",
                "--outer-holdout-sessions",
                "1",
                "--output",
                str(split_path),
            ]
        )
        == 0
    )
    split_summary = json.loads(capsys.readouterr().out)
    assert split_summary["fold_count"] == 2

    assert (
        main(
            [
                "fit-scaler",
                "--dataset",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--split",
                str(split_path),
                "--fold",
                "1",
                "--batch-size",
                "2",
                "--output",
                str(scaler_path),
            ]
        )
        == 0
    )
    scaler_summary = json.loads(capsys.readouterr().out)
    assert scaler_summary["fold_id"] == "fold-01"

    assert (
        main(
            [
                "init-experiment",
                "--dataset",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--instrument-id",
                "BTC-USD",
                "--split",
                str(split_path),
                "--scaler",
                str(scaler_path),
                "--fold",
                "1",
                "--source-root",
                str(source_root),
                "--dependency-lock",
                str(dependency_lock),
                "--output",
                str(config_path),
            ]
        )
        == 0
    )
    config_summary = json.loads(capsys.readouterr().out)
    config = ExperimentConfig.from_json(config_path.read_text(encoding="utf-8"))
    assert config_summary["run_id"] == config.run_id
    assert config.instrument_id == "BTC-USD"
    assert config.scaler_artifact_sha256 == scaler_summary["scaler_artifact_sha256"]
    assert config.model == ChmtQConfig.reference()
    assert config.training.effective_batch_size == 128

    split = WalkForwardManifest.from_json(split_path.read_text(encoding="utf-8"))
    fold = split.fold(1)
    scaler = FoldBoundScalerArtifact.from_json(
        scaler_path.read_text(encoding="utf-8"),
        expected_dataset_id=dataset.dataset_id,
        expected_split_manifest_sha256=split.artifact_sha256,
        expected_fold_id="fold-01",
        ordered_train_example_ids=fold.train_example_ids,
    )
    substitute = replace(
        scaler,
        scaler=replace(
            scaler.scaler,
            minimum_scale=scaler.scaler.minimum_scale * 2.0,
        ),
    )
    substitute_path = tmp_path / "substitute-scaler.json"
    substitute_path.write_text(substitute.to_json(), encoding="utf-8")
    assert (
        main(
            [
                "train",
                "--dataset",
                str(dataset_root),
                "--config",
                str(config_path),
                "--split",
                str(split_path),
                "--scaler",
                str(substitute_path),
                "--source-root",
                str(source_root),
                "--dependency-lock",
                str(dependency_lock),
                "--output-root",
                str(tmp_path / "runs"),
            ]
        )
        == 2
    )
    assert "configuration and scaler artifact differ" in capsys.readouterr().out


def test_cli_authorizes_only_local_research_and_keeps_strict_gates_false(tmp_path, capsys) -> None:
    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root, ready=False)
    _write_compiler_report(dataset_root, dataset)
    authorization_path = tmp_path / "research-authorization.json"
    split_path = tmp_path / "split.json"

    base_authorization = [
        "authorize-local-research",
        "--dataset",
        str(dataset_root),
        "--dataset-id",
        dataset.dataset_id,
        "--authorizer",
        "workspace-user",
        "--reason",
        "Explicit local non-promotable development training.",
        "--output",
        str(authorization_path),
    ]
    assert main(base_authorization) == 2
    assert "rights_confirmed" in capsys.readouterr().out
    assert not authorization_path.exists()

    assert (
        main(
            [
                *base_authorization,
                "--confirm-rights",
                "--acknowledge-incomplete-readiness",
                "--acknowledge-incomplete-provenance",
            ]
        )
        == 0
    )
    authorized = json.loads(capsys.readouterr().out)
    assert authorized["promotable"] is False
    assert authorized["outer_holdout_allowed"] is False

    assert (
        main(
            [
                "verify-dataset",
                "--input",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--instrument-id",
                "BTC-USD",
                "--local-research-authorization",
                str(authorization_path),
            ]
        )
        == 3
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["chmt_q_training_ready"] is False
    assert verified["local_research_authorized"] is True
    assert verified["promotable"] is False

    assert (
        main(
            [
                "build-split",
                "--dataset",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--initial-train-sessions",
                "2",
                "--validation-sessions",
                "1",
                "--outer-holdout-sessions",
                "1",
                "--local-research-authorization",
                str(authorization_path),
                "--output",
                str(split_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["fold_count"] == 2


def test_cli_trains_and_evaluates_verified_fold_without_holdout_access(
    tmp_path, capsys, monkeypatch
) -> None:
    tiny_model = ChmtQConfig.tiny()
    monkeypatch.setattr(
        ChmtQConfig,
        "reference",
        classmethod(lambda _cls: tiny_model),
    )

    dataset_root = tmp_path / "dataset"
    dataset = _dataset(dataset_root, ready=False)
    _write_compiler_report(dataset_root, dataset)
    authorization_path = tmp_path / "research-authorization.json"
    split_path = tmp_path / "split.json"
    scaler_path = tmp_path / "scaler.json"
    config_path = tmp_path / "experiment.json"
    source_root = tmp_path / "source"
    package = source_root / "daxalgo_numeric_transformer"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    (source_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    dependency_lock = source_root / "dependencies.lock"

    def invoke(arguments: list[str]) -> dict[str, object]:
        assert main(arguments) == 0
        captured = capsys.readouterr()
        assert captured.err == ""
        return json.loads(captured.out)

    authorization = invoke(
        [
            "authorize-local-research",
            "--dataset",
            str(dataset_root),
            "--dataset-id",
            dataset.dataset_id,
            "--authorizer",
            "workspace-user",
            "--reason",
            "Explicit local non-promotable development training.",
            "--confirm-rights",
            "--acknowledge-incomplete-readiness",
            "--acknowledge-incomplete-provenance",
            "--output",
            str(authorization_path),
        ]
    )
    assert authorization["promotable"] is False
    authorization_args = [
        "--local-research-authorization",
        str(authorization_path),
    ]

    lock_summary = invoke(["lock-environment", "--output", str(dependency_lock)])
    assert dependency_lock.is_file()
    assert lock_summary["dependency_lock_sha256"]

    assert (
        main(
            [
                "verify-dataset",
                "--input",
                str(dataset_root),
                "--dataset-id",
                dataset.dataset_id,
                "--instrument-id",
                "BTC-USD",
                *authorization_args,
            ]
        )
        == 3
    )
    verified = json.loads(capsys.readouterr().out)
    assert verified["dataset_id"] == dataset.dataset_id
    assert verified["row_count"] == 5
    assert verified["chmt_q_training_ready"] is False
    assert verified["local_research_authorized"] is True

    split_summary = invoke(
        [
            "build-split",
            "--dataset",
            str(dataset_root),
            "--dataset-id",
            dataset.dataset_id,
            "--initial-train-sessions",
            "2",
            "--validation-sessions",
            "1",
            "--outer-holdout-sessions",
            "1",
            *authorization_args,
            "--output",
            str(split_path),
        ]
    )
    split = WalkForwardManifest.from_json(split_path.read_text(encoding="utf-8"))
    fold = split.fold(2)
    assert split_summary["fold_count"] == 2
    assert fold.train_example_ids == ("e0", "e1", "e2")
    assert fold.validation_example_ids == ("e3",)
    assert split.outer_holdout.example_ids == ("e4",)

    scaler_summary = invoke(
        [
            "fit-scaler",
            "--dataset",
            str(dataset_root),
            "--dataset-id",
            dataset.dataset_id,
            "--split",
            str(split_path),
            "--fold",
            "2",
            "--batch-size",
            "2",
            *authorization_args,
            "--output",
            str(scaler_path),
        ]
    )
    scaler = FoldBoundScalerArtifact.from_json(
        scaler_path.read_text(encoding="utf-8"),
        expected_dataset_id=dataset.dataset_id,
        expected_split_manifest_sha256=split.artifact_sha256,
        expected_fold_id="fold-02",
        ordered_train_example_ids=fold.train_example_ids,
    )
    assert scaler_summary["train_example_count"] == 3
    assert scaler.dataset_id == dataset.dataset_id
    assert scaler.fold_id == "fold-02"
    assert scaler.train_example_ids == fold.train_example_ids

    config_summary = invoke(
        [
            "init-experiment",
            "--dataset",
            str(dataset_root),
            "--dataset-id",
            dataset.dataset_id,
            "--instrument-id",
            "BTC-USD",
            "--split",
            str(split_path),
            "--scaler",
            str(scaler_path),
            "--fold",
            "2",
            "--source-root",
            str(source_root),
            "--dependency-lock",
            str(dependency_lock),
            "--micro-batch-size",
            "1",
            "--accumulation-steps",
            "2",
            "--max-epochs",
            "1",
            "--early-stop-patience",
            "1",
            *authorization_args,
            "--output",
            str(config_path),
        ]
    )
    config = ExperimentConfig.from_json(config_path.read_text(encoding="utf-8"))
    assert config_summary["effective_batch_size"] == 2
    assert config.instrument_id == dataset.instrument_id == "BTC-USD"
    assert config.fold_id == "fold-02"
    assert config.scaler_artifact_sha256 == scaler.artifact_sha256
    assert config.model == tiny_model
    assert config.local_research_only

    train_summary = invoke(
        [
            "train",
            "--dataset",
            str(dataset_root),
            "--config",
            str(config_path),
            "--split",
            str(split_path),
            "--scaler",
            str(scaler_path),
            "--source-root",
            str(source_root),
            "--dependency-lock",
            str(dependency_lock),
            *authorization_args,
            "--output-root",
            str(tmp_path / "runs"),
        ]
    )
    run_directory = Path(str(train_summary["run_directory"]))
    checkpoint = Path(str(train_summary["best_checkpoint"]))
    assert train_summary["optimizer_steps"] == 1
    assert train_summary["outer_holdout_accessed"] is False
    assert train_summary["promotable"] is False
    assert checkpoint.is_dir()
    assert checkpoint.parent == run_directory / "checkpoints"

    epoch_document = json.loads(
        (run_directory / "epochs" / "epoch-0001.json").read_text(encoding="utf-8")
    )
    epoch = epoch_document["payload"]
    assert epoch["optimizer_step"] == 1
    assert epoch["trained_examples"] == 2
    assert epoch["dropped_train_examples"] == 1
    assert len(epoch["windows"]) == 1
    assert '"e4"' not in json.dumps(epoch, sort_keys=True)

    evaluation = invoke(
        [
            "evaluate",
            "--dataset",
            str(dataset_root),
            "--config",
            str(config_path),
            "--split",
            str(split_path),
            "--scaler",
            str(scaler_path),
            "--source-root",
            str(source_root),
            "--dependency-lock",
            str(dependency_lock),
            "--checkpoint",
            str(checkpoint),
            *authorization_args,
        ]
    )
    assert evaluation["run_id"] == config.run_id
    assert evaluation["fold_id"] == "fold-02"
    assert evaluation["checkpoint_step"] == 1
    assert evaluation["outer_holdout_accessed"] is False
    assert evaluation["promotable"] is False
    assert torch.isfinite(torch.tensor(evaluation["primary_metric"]))

    run_metadata = json.loads((run_directory / "run.json").read_text(encoding="utf-8"))["payload"]
    run_summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    assert run_metadata["outer_holdout_accessed"] is False
    assert run_metadata["local_research_only"] is True
    assert run_metadata["promotable"] is False
    assert run_summary["outer_holdout_accessed"] is False
    assert run_summary["local_research_only"] is True
    assert run_summary["promotable"] is False
    assert not list(run_directory.rglob("outer-holdout-consumption.json"))
