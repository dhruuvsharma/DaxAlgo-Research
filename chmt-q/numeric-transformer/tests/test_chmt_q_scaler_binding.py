from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from daxalgo_numeric_transformer.chmt_q import ChmtQBatch
from daxalgo_numeric_transformer.chmt_q.data.dataset import (
    ArrowDataset,
    MaterializedBatch,
)
from daxalgo_numeric_transformer.chmt_q.data.scaler_binding import (
    FoldBoundScalerArtifact,
    fit_fold_bound_scaler,
    ordered_train_example_ids_sha256,
)

DATASET_ID = "a" * 64
OTHER_DATASET_ID = "b" * 64
SPLIT_ID = "c" * 64
OTHER_SPLIT_ID = "d" * 64
TRAIN_IDS = ("example-001", "example-002")


class _Capabilities:
    def __init__(self) -> None:
        self.require_count = 0

    def require_training(self) -> None:
        self.require_count += 1


class _Dataset(ArrowDataset):
    def __init__(
        self,
        *,
        example_ids: tuple[str, ...],
        yielded: tuple[MaterializedBatch, ...],
        dataset_id: str = DATASET_ID,
    ) -> None:
        self._example_ids = example_ids
        self._yielded = yielded
        self.capabilities = _Capabilities()
        self.manifest = SimpleNamespace(
            dataset_id=dataset_id,
            capabilities=self.capabilities,
        )
        self.local_research_authorization = None
        self.calls: list[dict[str, Any]] = []

    @property
    def example_ids(self) -> tuple[str, ...]:
        return self._example_ids

    def iter_batches(
        self,
        example_ids: Iterable[str],
        *,
        batch_size: int,
        partition: str,
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[MaterializedBatch]:
        self.calls.append(
            {
                "example_ids": tuple(example_ids),
                "batch_size": batch_size,
                "partition": partition,
                "seed": seed,
                "drop_last": drop_last,
            }
        )
        yield from self._yielded


def _batch(*, offset: float, batch_size: int) -> ChmtQBatch:
    row = torch.arange(batch_size, dtype=torch.float32)
    bars = torch.zeros(batch_size, 64, 16)
    l1 = torch.zeros(batch_size, 240, 16)
    l2 = torch.zeros(batch_size, 240, 2, 10, 12)
    targets = torch.zeros(batch_size, 5, 5)
    for feature in range(16):
        bars[..., feature] = offset + feature + row[:, None]
        l1[..., feature] = offset + 2 * feature + row[:, None]
    for feature in range(12):
        l2[..., feature] = offset + 3 * feature + row[:, None, None, None]
    for horizon in range(5):
        for target in range(5):
            if target == 4:
                targets[:, horizon, target] = torch.linspace(-0.25, 0.25, batch_size)
            else:
                targets[:, horizon, target] = offset + 10 * horizon + target + row
    l2_level_mask = torch.ones(batch_size, 240, 2, 10, dtype=torch.bool)
    return ChmtQBatch(
        bars=bars,
        bars_mask=torch.ones(batch_size, 64, dtype=torch.bool),
        l1=l1,
        l1_mask=torch.ones(batch_size, 240, dtype=torch.bool),
        l2=l2,
        l2_level_mask=l2_level_mask,
        l2_snapshot_mask=l2_level_mask.any(dim=(-1, -2)),
        quality=torch.zeros(batch_size, 16),
        targets=targets,
        target_mask=torch.ones(batch_size, 5, 5, dtype=torch.bool),
        presence_labels=torch.zeros(batch_size, 5, 2),
        presence_mask=torch.ones(batch_size, 5, 2, dtype=torch.bool),
    )


def _materialized(
    example_ids: tuple[str, ...],
    *,
    offset: float = 0.0,
) -> MaterializedBatch:
    count = len(example_ids)
    return MaterializedBatch(
        example_ids=example_ids,
        session_ids=tuple("session" for _ in example_ids),
        feature_cutoff_ns=tuple(range(10, 10 + count)),
        decision_ns=tuple(range(20, 20 + count)),
        dependency_start_ns=tuple(range(count)),
        dependency_end_ns=tuple(range(30, 30 + count)),
        instrument_ids=tuple("BTC-USD" for _ in example_ids),
        anchor_sources=tuple("l1_mid" for _ in example_ids),
        anchor_prices=tuple(50_000.0 + index for index in range(count)),
        tick_sizes=tuple(0.5 for _ in example_ids),
        log_volume_references=tuple(1.0 for _ in example_ids),
        provenance_sha256=tuple(f"{index + 1:064x}" for index in range(count)),
        batch=_batch(offset=offset, batch_size=count),
    )


def _fit(dataset: _Dataset) -> FoldBoundScalerArtifact:
    return fit_fold_bound_scaler(
        dataset,
        dataset_id=DATASET_ID,
        split_manifest_sha256=SPLIT_ID,
        fold_id="fold-01",
        ordered_train_example_ids=TRAIN_IDS,
        batch_size=2,
    )


def test_fit_binds_exact_ordered_ids_without_exposing_role_selection() -> None:
    dataset = _Dataset(
        example_ids=(*TRAIN_IDS, "outer-holdout-001"),
        yielded=(_materialized(TRAIN_IDS),),
    )

    artifact = _fit(dataset)

    assert artifact.dataset_id == DATASET_ID
    assert artifact.split_manifest_sha256 == SPLIT_ID
    assert artifact.fold_id == "fold-01"
    assert artifact.train_example_count == 2
    assert artifact.ordered_train_example_ids_sha256 == (
        ordered_train_example_ids_sha256(TRAIN_IDS)
    )
    assert artifact.scaler.source_batch_set_hash == (artifact.ordered_train_example_ids_sha256)
    assert dataset.capabilities.require_count == 1
    assert dataset.calls == [
        {
            "example_ids": TRAIN_IDS,
            "batch_size": 2,
            "partition": "evaluation",
            "seed": None,
            "drop_last": False,
        }
    ]
    parameters = inspect.signature(fit_fold_bound_scaler).parameters
    assert "partition" not in parameters
    assert "role" not in parameters


def test_fit_is_deterministic_and_id_order_changes_the_digest() -> None:
    first = _fit(_Dataset(example_ids=TRAIN_IDS, yielded=(_materialized(TRAIN_IDS),)))
    second = _fit(_Dataset(example_ids=TRAIN_IDS, yielded=(_materialized(TRAIN_IDS),)))

    assert first == second
    assert first.artifact_sha256 == second.artifact_sha256
    assert ordered_train_example_ids_sha256(TRAIN_IDS) != (
        ordered_train_example_ids_sha256(tuple(reversed(TRAIN_IDS)))
    )


def test_fold_fit_uses_bounded_disk_spooling_without_identity_drift(tmp_path: Any) -> None:
    dataset = _Dataset(
        example_ids=TRAIN_IDS,
        yielded=(_materialized(TRAIN_IDS),),
    )

    artifact = fit_fold_bound_scaler(
        dataset,
        dataset_id=DATASET_ID,
        split_manifest_sha256=SPLIT_ID,
        fold_id="fold-01",
        ordered_train_example_ids=TRAIN_IDS,
        batch_size=2,
        spool_directory=tmp_path,
        max_chunk_values=5,
    )

    assert artifact.scaler.fit_method == "exact-disk-spooled-radix-median-mad-v1"
    assert artifact.scaler.source_batch_set_hash == ordered_train_example_ids_sha256(TRAIN_IDS)
    assert list(tmp_path.iterdir()) == []


def test_fit_rejects_binding_mismatch_unknown_ids_and_noncanonical_order() -> None:
    dataset = _Dataset(example_ids=TRAIN_IDS, yielded=(_materialized(TRAIN_IDS),))

    with pytest.raises(ValueError, match="identity differs"):
        fit_fold_bound_scaler(
            dataset,
            dataset_id=OTHER_DATASET_ID,
            split_manifest_sha256=SPLIT_ID,
            fold_id="fold-01",
            ordered_train_example_ids=TRAIN_IDS,
            batch_size=2,
        )
    with pytest.raises(KeyError, match="absent"):
        fit_fold_bound_scaler(
            dataset,
            dataset_id=DATASET_ID,
            split_manifest_sha256=SPLIT_ID,
            fold_id="fold-01",
            ordered_train_example_ids=(TRAIN_IDS[0], "unknown"),
            batch_size=2,
        )
    with pytest.raises(ValueError, match="chronological order"):
        fit_fold_bound_scaler(
            dataset,
            dataset_id=DATASET_ID,
            split_manifest_sha256=SPLIT_ID,
            fold_id="fold-01",
            ordered_train_example_ids=tuple(reversed(TRAIN_IDS)),
            batch_size=2,
        )


def test_fit_rejects_non_train_or_incomplete_materialized_batches() -> None:
    non_train = _Dataset(
        example_ids=(*TRAIN_IDS, "outer-holdout-001"),
        yielded=(_materialized((TRAIN_IDS[0], "outer-holdout-001")),),
    )
    with pytest.raises(PermissionError, match="non-train"):
        _fit(non_train)

    incomplete = _Dataset(
        example_ids=TRAIN_IDS,
        yielded=(_materialized((TRAIN_IDS[0],)),),
    )
    with pytest.raises(ValueError, match="cover every train"):
        _fit(incomplete)


def test_canonical_round_trip_requires_all_external_bindings() -> None:
    artifact = _fit(_Dataset(example_ids=TRAIN_IDS, yielded=(_materialized(TRAIN_IDS),)))
    serialized = artifact.to_json()

    restored = FoldBoundScalerArtifact.from_json(
        serialized,
        expected_dataset_id=DATASET_ID,
        expected_split_manifest_sha256=SPLIT_ID,
        expected_fold_id="fold-01",
        ordered_train_example_ids=TRAIN_IDS,
    )

    assert restored == artifact
    assert restored.to_json() == serialized
    with pytest.raises(ValueError, match="expected_dataset_id"):
        FoldBoundScalerArtifact.from_json(
            serialized,
            expected_dataset_id=OTHER_DATASET_ID,
            expected_split_manifest_sha256=SPLIT_ID,
            expected_fold_id="fold-01",
            ordered_train_example_ids=TRAIN_IDS,
        )
    with pytest.raises(ValueError, match="expected_split_manifest"):
        FoldBoundScalerArtifact.from_json(
            serialized,
            expected_dataset_id=DATASET_ID,
            expected_split_manifest_sha256=OTHER_SPLIT_ID,
            expected_fold_id="fold-01",
            ordered_train_example_ids=TRAIN_IDS,
        )
    with pytest.raises(ValueError, match="ordered train example-ID"):
        FoldBoundScalerArtifact.from_json(
            serialized,
            expected_dataset_id=DATASET_ID,
            expected_split_manifest_sha256=SPLIT_ID,
            expected_fold_id="fold-01",
            ordered_train_example_ids=(TRAIN_IDS[0], "different"),
        )


def test_serialized_tamper_and_noncanonical_json_fail_closed() -> None:
    artifact = _fit(_Dataset(example_ids=TRAIN_IDS, yielded=(_materialized(TRAIN_IDS),)))
    serialized = artifact.to_json()
    tampered = serialized.replace('"fold_id":"fold-01"', '"fold_id":"fold-02"')

    with pytest.raises(ValueError, match="SHA-256"):
        FoldBoundScalerArtifact.from_json(
            tampered,
            expected_dataset_id=DATASET_ID,
            expected_split_manifest_sha256=SPLIT_ID,
            expected_fold_id="fold-01",
            ordered_train_example_ids=TRAIN_IDS,
        )
    with pytest.raises(ValueError, match="canonical"):
        FoldBoundScalerArtifact.from_json(
            serialized + "\n",
            expected_dataset_id=DATASET_ID,
            expected_split_manifest_sha256=SPLIT_ID,
            expected_fold_id="fold-01",
            ordered_train_example_ids=TRAIN_IDS,
        )


def test_normalization_requires_explicit_permitted_ids_and_preserves_metadata() -> None:
    artifact = _fit(_Dataset(example_ids=TRAIN_IDS, yielded=(_materialized(TRAIN_IDS),)))
    validation = _materialized(("validation-001",), offset=100.0)

    normalized = artifact.normalize_batch(
        validation,
        permitted_example_ids=("validation-001",),
    )

    assert normalized.example_ids == validation.example_ids
    assert normalized.session_ids == validation.session_ids
    assert normalized.decision_ns == validation.decision_ns
    assert normalized.batch is not validation.batch
    assert not torch.equal(normalized.batch.bars, validation.batch.bars)
    with pytest.raises(PermissionError, match="non-permitted"):
        artifact.normalize_batch(
            validation,
            permitted_example_ids=("different",),
        )
    with pytest.raises(PermissionError, match="non-permitted"):
        artifact.normalize_train_batch(validation)
    with pytest.raises(ValueError, match="must not be empty"):
        artifact.normalize_batch(validation, permitted_example_ids=())


def test_train_normalization_accepts_only_bound_train_ids() -> None:
    training = _materialized(TRAIN_IDS, offset=0.25)
    artifact = _fit(_Dataset(example_ids=TRAIN_IDS, yielded=(training,)))

    normalized = artifact.normalize_train_batch(training)

    assert normalized.example_ids == TRAIN_IDS
    assert normalized.batch.bars.abs().max() <= 8.0
