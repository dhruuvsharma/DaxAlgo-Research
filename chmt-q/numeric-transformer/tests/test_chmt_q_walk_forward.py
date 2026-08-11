from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.data.dataset import (
    ArrowDataset,
    DatasetCapabilities,
    EngineeredExample,
    ExampleMetadata,
    write_engineered_fixture_dataset,
)
from daxalgo_numeric_transformer.chmt_q.splits import ExampleDependency
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch
from daxalgo_numeric_transformer.chmt_q.training.walk_forward import (
    OuterHoldoutAccess,
    WalkForwardManifest,
    build_expanding_walk_forward,
    iter_fold_batches,
    iter_outer_holdout_batches,
    outer_holdout_consumption_ledger_path,
)


def _dependency(
    example_id: str,
    session_id: str,
    start: int,
    decision: int,
    end: int,
) -> ExampleDependency:
    return ExampleDependency(example_id, session_id, decision, start, end)


def _simple_dependencies() -> list[ExampleDependency]:
    return [
        _dependency(
            f"e{index}",
            f"s{index}",
            index * 1_000,
            index * 1_000 + 100,
            index * 1_000 + 200,
        )
        for index in range(5)
    ]


def _verified_dataset_and_split(tmp_path: Path) -> tuple[ArrowDataset, WalkForwardManifest]:
    capabilities = DatasetCapabilities(
        training_ready=True,
        rights_confirmed=True,
        provenance_complete=True,
        readiness_evidence_sha256="a" * 64,
        rights_evidence_sha256="b" * 64,
        provenance_evidence_sha256="c" * 64,
    )
    engineered = [
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
            batch=make_chmt_q_synthetic_batch(ChmtQConfig.tiny(), batch_size=1, seed=500 + index),
        )
        for index in range(5)
    ]
    dataset_root = tmp_path / "dataset"
    dataset_manifest = write_engineered_fixture_dataset(
        dataset_root,
        engineered,
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=capabilities,
        rows_per_shard=2,
    )
    dataset = ArrowDataset.open(dataset_root, expected_dataset_id=dataset_manifest.dataset_id)
    split = build_expanding_walk_forward(
        dataset.dependencies(),
        source_dataset_hash=dataset_manifest.dataset_id,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    return dataset, split


def _outer_access(tmp_path: Path, split: WalkForwardManifest) -> OuterHoldoutAccess:
    candidate_path = tmp_path / "frozen-candidate.pt"
    evidence_path = tmp_path / "freeze-evidence.json"
    candidate_bytes = b"reviewed frozen candidate\n"
    evidence_bytes = b'{"candidate_is_frozen":true}\n'
    candidate_path.write_bytes(candidate_bytes)
    evidence_path.write_bytes(evidence_bytes)
    return OuterHoldoutAccess(
        split_sha256=split.artifact_sha256,
        frozen_candidate_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        freeze_evidence_sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        frozen_candidate_path=candidate_path,
        freeze_evidence_path=evidence_path,
    )


def test_expanding_folds_are_deterministic_canonical_and_holdout_sealed() -> None:
    examples = _simple_dependencies()

    first = build_expanding_walk_forward(
        examples,
        source_dataset_hash="a" * 64,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    second = build_expanding_walk_forward(
        reversed(examples),
        source_dataset_hash="a" * 64,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    restored = WalkForwardManifest.from_json(first.to_json())

    assert first == second == restored
    assert first.to_json() == restored.to_json()
    assert len({fold.fold_sha256 for fold in first.folds}) == 2
    assert first.fold(1).train_session_ids == ("s0", "s1")
    assert first.fold(1).validation_session_ids == ("s2",)
    assert first.fold(2).train_session_ids == ("s0", "s1", "s2")
    assert first.fold(2).validation_session_ids == ("s3",)
    assert first.outer_holdout.session_ids == ("s4",)
    assert first.outer_holdout.example_ids == ("e4",)


def test_exact_overlap_and_embargo_purge_only_conflicting_earlier_examples() -> None:
    examples = [
        _dependency("far-safe", "s0", 0, 100, 200),
        _dependency("near", "s0", 50, 130, 240),
        _dependency("overlap", "s0", 100, 150, 270),
        _dependency("v1", "s1", 250, 300, 400),
        _dependency("v2", "s2", 1_000, 1_100, 1_200),
        _dependency("holdout", "s3", 2_000, 2_100, 2_200),
    ]

    overlap = build_expanding_walk_forward(
        examples,
        source_dataset_hash="b" * 64,
        initial_train_session_count=1,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    embargo = build_expanding_walk_forward(
        examples,
        source_dataset_hash="b" * 64,
        initial_train_session_count=1,
        validation_session_count=1,
        outer_holdout_session_count=1,
        embargo_ns=20,
    )

    first = overlap.fold(1)
    assert first.train_example_ids == ("far-safe", "near")
    assert first.train_purged_example_ids == ("overlap",)
    assert first.purges[0].against_example_id == "v1"
    assert first.purges[0].reason == "dependency_overlap"
    assert embargo.fold(1).train_example_ids == ("far-safe",)
    assert embargo.fold(1).train_purged_example_ids == ("near", "overlap")
    assert embargo.fold(1).purges[0].reason == "embargo"


def test_outer_holdout_requires_frozen_candidate_access_and_fold_binding(tmp_path) -> None:
    dataset, split = _verified_dataset_and_split(tmp_path)

    train = list(
        iter_fold_batches(
            dataset,
            split,
            fold_number=1,
            partition="train",
            batch_size=2,
            seed=17,
        )
    )
    validation = list(
        iter_fold_batches(
            dataset,
            split,
            fold_number=1,
            partition="validation",
            batch_size=2,
        )
    )
    assert {item for batch in train for item in batch.example_ids} == {"e0", "e1"}
    assert validation[0].example_ids == ("e2",)
    with pytest.raises(ValueError, match="outer holdout"):
        iter_fold_batches(
            dataset,
            split,
            fold_number=1,
            partition="outer_holdout",  # type: ignore[arg-type]
            batch_size=1,
        )

    access = _outer_access(tmp_path, split)
    ledger_path = outer_holdout_consumption_ledger_path(dataset, split)
    wrong = replace(access, split_sha256="f" * 64)
    with pytest.raises(PermissionError, match="different split"):
        iter_outer_holdout_batches(dataset, split, access=wrong, batch_size=1)

    with pytest.raises(ValueError, match="batch_size"):
        iter_outer_holdout_batches(dataset, split, access=access, batch_size=0)
    assert not ledger_path.exists()

    outer_iterator = iter_outer_holdout_batches(dataset, split, access=access, batch_size=1)
    assert ledger_path.is_file()
    ledger = {
        "schema_version": 1,
        "purpose": "frozen-candidate-final-evaluation",
        "split_sha256": split.artifact_sha256,
        "outer_holdout_sha256": split.outer_holdout.holdout_sha256,
        "frozen_candidate_sha256": access.frozen_candidate_sha256,
        "freeze_evidence_sha256": access.freeze_evidence_sha256,
    }
    assert ledger_path.read_text(encoding="ascii") == (
        json.dumps(ledger, sort_keys=True, separators=(",", ":")) + "\n"
    )
    outer = list(outer_iterator)
    assert outer[0].example_ids == ("e4",)
    with pytest.raises(PermissionError, match="already been consumed"):
        iter_outer_holdout_batches(
            dataset,
            split,
            access=_outer_access(tmp_path, split),
            batch_size=1,
        )


def test_outer_holdout_rejects_changed_and_non_regular_freeze_artifacts(tmp_path) -> None:
    dataset, split = _verified_dataset_and_split(tmp_path)
    access = _outer_access(tmp_path, split)
    ledger_path = outer_holdout_consumption_ledger_path(dataset, split)

    access.frozen_candidate_path.write_bytes(b"candidate changed after review\n")
    with pytest.raises(ValueError, match="frozen candidate artifact SHA-256"):
        iter_outer_holdout_batches(dataset, split, access=access, batch_size=1)
    assert not ledger_path.exists()

    access = _outer_access(tmp_path, split)
    non_regular = tmp_path / "not-an-evidence-file"
    non_regular.mkdir()
    access = replace(access, freeze_evidence_path=non_regular)
    with pytest.raises(ValueError, match="regular non-symlink file"):
        iter_outer_holdout_batches(dataset, split, access=access, batch_size=1)
    assert not ledger_path.exists()

    access = _outer_access(tmp_path, split)
    ledger_path.mkdir()
    with pytest.raises(ValueError, match="occupied by a non-regular file"):
        iter_outer_holdout_batches(dataset, split, access=access, batch_size=1)


def test_local_research_access_never_unlocks_outer_holdout(tmp_path) -> None:
    dataset, _strict_split = _verified_dataset_and_split(tmp_path)
    blocked_capabilities = DatasetCapabilities(
        training_ready=False,
        rights_confirmed=False,
        provenance_complete=False,
        readiness_evidence_sha256="a" * 64,
        rights_evidence_sha256="b" * 64,
        provenance_evidence_sha256="c" * 64,
    )
    dataset.manifest = replace(
        dataset.manifest,
        capabilities=blocked_capabilities,
    )
    dataset.local_research_authorization = object()
    research_split = build_expanding_walk_forward(
        dataset.dependencies(),
        source_dataset_hash=dataset.manifest.dataset_id,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )

    with pytest.raises(PermissionError, match="training_ready"):
        iter_outer_holdout_batches(
            dataset,
            research_split,
            access=_outer_access(tmp_path, research_split),
            batch_size=1,
        )


def test_outer_holdout_rejects_symlinked_artifact_and_canonical_ledger(tmp_path) -> None:
    dataset, split = _verified_dataset_and_split(tmp_path)
    access = _outer_access(tmp_path, split)
    ledger_path = outer_holdout_consumption_ledger_path(dataset, split)
    candidate_link = tmp_path / "candidate-link.pt"
    try:
        candidate_link.symlink_to(access.frozen_candidate_path)
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this platform")

    linked_candidate = replace(access, frozen_candidate_path=candidate_link)
    with pytest.raises(ValueError, match="symbolic links or reparse points"):
        iter_outer_holdout_batches(dataset, split, access=linked_candidate, batch_size=1)
    assert not ledger_path.exists()

    existing_ledger = tmp_path / "existing-ledger.json"
    existing_ledger.write_text("{}\n", encoding="ascii")
    try:
        ledger_path.symlink_to(existing_ledger)
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this platform")
    with pytest.raises(ValueError, match="symbolic links or reparse points"):
        iter_outer_holdout_batches(dataset, split, access=access, batch_size=1)
    assert existing_ledger.read_text(encoding="ascii") == "{}\n"


def test_manifest_validation_detects_dependency_change() -> None:
    examples = _simple_dependencies()
    manifest = build_expanding_walk_forward(
        examples,
        source_dataset_hash="c" * 64,
        initial_train_session_count=2,
        validation_session_count=1,
        outer_holdout_session_count=1,
    )
    mutated = [
        replace(item, dependency_end_ns=2_050) if item.example_id == "e1" else item
        for item in examples
    ]

    with pytest.raises(ValueError, match="differs"):
        manifest.validate(mutated)


def test_incomplete_fold_windows_and_interleaved_sessions_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly consumed"):
        build_expanding_walk_forward(
            _simple_dependencies() + [_dependency("e5", "s5", 5_000, 5_100, 5_200)],
            source_dataset_hash="d" * 64,
            initial_train_session_count=2,
            validation_session_count=2,
            outer_holdout_session_count=1,
        )

    interleaved = [
        _dependency("a0", "s0", 0, 100, 150),
        _dependency("a1", "s0", 300, 400, 450),
        _dependency("b", "s1", 150, 200, 250),
        _dependency("c", "s2", 500, 600, 650),
        _dependency("d", "s3", 700, 800, 850),
    ]
    with pytest.raises(ValueError, match="interleave"):
        build_expanding_walk_forward(
            interleaved,
            source_dataset_hash="e" * 64,
            initial_train_session_count=1,
            validation_session_count=1,
            outer_holdout_session_count=1,
        )
