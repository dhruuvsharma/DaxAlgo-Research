"""Expanding session walk-forward folds with sealed outer-holdout access."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..data.dataset import ArrowDataset, MaterializedBatch
from ..splits import ExampleDependency

WALK_FORWARD_SCHEMA_VERSION = 1
WALK_FORWARD_METHOD = "expanding-session-exact-dependency-purge-v1"
INTERVAL_SEMANTICS = "half-open [dependency_start_ns, dependency_end_ns)"
OUTER_HOLDOUT_PURPOSE = "frozen-candidate-final-evaluation"
OUTER_HOLDOUT_LEDGER_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class FoldPurge:
    """Exact example and reason for one earlier-side fold exclusion."""

    example_id: str
    assigned_partition: str
    against_partition: str
    against_example_id: str
    reason: str

    def __post_init__(self) -> None:
        for name in ("example_id", "against_example_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.assigned_partition not in {"train", "validation"}:
            raise ValueError("assigned_partition must be train or validation")
        allowed_against = (
            {"validation", "outer_holdout"}
            if self.assigned_partition == "train"
            else {"outer_holdout"}
        )
        if self.against_partition not in allowed_against:
            raise ValueError("purge partition ordering is invalid")
        if self.reason not in {"dependency_overlap", "embargo"}:
            raise ValueError("purge reason must be dependency_overlap or embargo")

    def to_payload(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "assigned_partition": self.assigned_partition,
            "against_partition": self.against_partition,
            "against_example_id": self.against_example_id,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> FoldPurge:
        expected = {
            "example_id",
            "assigned_partition",
            "against_partition",
            "against_example_id",
            "reason",
        }
        value = _require_object(payload, "purge", expected)
        return cls(**{name: _require_string(value[name], name) for name in expected})


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    """One expanding training window and its next chronological validation window."""

    fold_number: int
    train_session_ids: tuple[str, ...]
    validation_session_ids: tuple[str, ...]
    train_example_ids: tuple[str, ...]
    validation_example_ids: tuple[str, ...]
    train_purged_example_ids: tuple[str, ...]
    validation_purged_example_ids: tuple[str, ...]
    purges: tuple[FoldPurge, ...]

    def __post_init__(self) -> None:
        if _require_integer(self.fold_number, "fold_number") <= 0:
            raise ValueError("fold_number must be positive")
        _validate_unique_nonempty(self.train_session_ids, "train_session_ids")
        _validate_unique_nonempty(self.validation_session_ids, "validation_session_ids")
        if set(self.train_session_ids) & set(self.validation_session_ids):
            raise ValueError("a fold cannot train and validate on the same session")
        _validate_unique_nonempty(self.train_example_ids, "train_example_ids")
        _validate_unique_nonempty(self.validation_example_ids, "validation_example_ids")
        _validate_unique(self.train_purged_example_ids, "train_purged_example_ids")
        _validate_unique(self.validation_purged_example_ids, "validation_purged_example_ids")
        retained = set(self.train_example_ids) | set(self.validation_example_ids)
        purged = set(self.train_purged_example_ids) | set(self.validation_purged_example_ids)
        if len(retained) != len(self.train_example_ids) + len(self.validation_example_ids):
            raise ValueError("an example is retained in both fold partitions")
        if retained & purged:
            raise ValueError("a fold both retains and purges an example")
        if len(purged) != len(self.train_purged_example_ids) + len(
            self.validation_purged_example_ids
        ):
            raise ValueError("an example is purged from both fold partitions")
        expected_purges = (*self.train_purged_example_ids, *self.validation_purged_example_ids)
        if tuple(item.example_id for item in self.purges) != expected_purges:
            raise ValueError("purge records must match fold purged example IDs in order")
        if any(
            item.assigned_partition != "train"
            for item in self.purges[: len(self.train_purged_example_ids)]
        ) or any(
            item.assigned_partition != "validation"
            for item in self.purges[len(self.train_purged_example_ids) :]
        ):
            raise ValueError("purge records do not match their assigned fold partition")

    def payload(self) -> dict[str, Any]:
        return {
            "fold_number": self.fold_number,
            "train_session_ids": list(self.train_session_ids),
            "validation_session_ids": list(self.validation_session_ids),
            "train_example_ids": list(self.train_example_ids),
            "validation_example_ids": list(self.validation_example_ids),
            "train_purged_example_ids": list(self.train_purged_example_ids),
            "validation_purged_example_ids": list(self.validation_purged_example_ids),
            "purges": [item.to_payload() for item in self.purges],
        }

    @property
    def fold_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("ascii")).hexdigest()

    def record(self) -> dict[str, Any]:
        return {"fold_sha256": self.fold_sha256, "payload": self.payload()}

    @classmethod
    def from_record(cls, record: Any) -> WalkForwardFold:
        value = _require_object(record, "fold record", {"fold_sha256", "payload"})
        expected_hash = _require_string(value["fold_sha256"], "fold_sha256")
        _validate_sha256(expected_hash, "fold_sha256")
        payload = _require_object(
            value["payload"],
            "fold payload",
            {
                "fold_number",
                "train_session_ids",
                "validation_session_ids",
                "train_example_ids",
                "validation_example_ids",
                "train_purged_example_ids",
                "validation_purged_example_ids",
                "purges",
            },
        )
        fold = cls(
            fold_number=_require_integer(payload["fold_number"], "fold_number"),
            train_session_ids=_string_tuple(payload["train_session_ids"], "train_session_ids"),
            validation_session_ids=_string_tuple(
                payload["validation_session_ids"], "validation_session_ids"
            ),
            train_example_ids=_string_tuple(payload["train_example_ids"], "train_example_ids"),
            validation_example_ids=_string_tuple(
                payload["validation_example_ids"], "validation_example_ids"
            ),
            train_purged_example_ids=_string_tuple(
                payload["train_purged_example_ids"], "train_purged_example_ids"
            ),
            validation_purged_example_ids=_string_tuple(
                payload["validation_purged_example_ids"],
                "validation_purged_example_ids",
            ),
            purges=tuple(
                FoldPurge.from_payload(item) for item in _require_list(payload["purges"], "purges")
            ),
        )
        if fold.fold_sha256 != expected_hash:
            raise ValueError("fold SHA-256 does not match its canonical payload")
        return fold


@dataclass(frozen=True, slots=True)
class OuterHoldout:
    """The most recent complete sessions, sealed outside development folds."""

    session_ids: tuple[str, ...]
    example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_unique_nonempty(self.session_ids, "outer holdout session_ids")
        _validate_unique_nonempty(self.example_ids, "outer holdout example_ids")

    def payload(self) -> dict[str, Any]:
        return {"session_ids": list(self.session_ids), "example_ids": list(self.example_ids)}

    @property
    def holdout_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("ascii")).hexdigest()

    def record(self) -> dict[str, Any]:
        return {"holdout_sha256": self.holdout_sha256, "payload": self.payload()}

    @classmethod
    def from_record(cls, record: Any) -> OuterHoldout:
        value = _require_object(record, "outer holdout record", {"holdout_sha256", "payload"})
        expected_hash = _require_string(value["holdout_sha256"], "holdout_sha256")
        _validate_sha256(expected_hash, "holdout_sha256")
        payload = _require_object(
            value["payload"], "outer holdout payload", {"session_ids", "example_ids"}
        )
        holdout = cls(
            session_ids=_string_tuple(payload["session_ids"], "session_ids"),
            example_ids=_string_tuple(payload["example_ids"], "example_ids"),
        )
        if holdout.holdout_sha256 != expected_hash:
            raise ValueError("outer holdout SHA-256 does not match its canonical payload")
        return holdout


@dataclass(frozen=True, slots=True)
class WalkForwardManifest:
    """Content-addressed folds and the independently sealed outer holdout."""

    source_dataset_hash: str
    initial_train_session_count: int
    validation_session_count: int
    outer_holdout_session_count: int
    embargo_ns: int
    folds: tuple[WalkForwardFold, ...]
    outer_holdout: OuterHoldout
    schema_version: int = WALK_FORWARD_SCHEMA_VERSION
    method: str = WALK_FORWARD_METHOD
    interval_semantics: str = INTERVAL_SEMANTICS

    def __post_init__(self) -> None:
        _validate_sha256(self.source_dataset_hash, "source_dataset_hash")
        for name in (
            "initial_train_session_count",
            "validation_session_count",
            "outer_holdout_session_count",
        ):
            if _require_integer(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} must be positive")
        if _require_integer(self.embargo_ns, "embargo_ns") < 0:
            raise ValueError("embargo_ns must be non-negative")
        if (
            self.schema_version != WALK_FORWARD_SCHEMA_VERSION
            or self.method != WALK_FORWARD_METHOD
            or self.interval_semantics != INTERVAL_SEMANTICS
        ):
            raise ValueError("unsupported walk-forward schema, method, or interval semantics")
        if not self.folds or any(not isinstance(item, WalkForwardFold) for item in self.folds):
            raise ValueError("walk-forward manifest must contain at least one fold")
        if not isinstance(self.outer_holdout, OuterHoldout):
            raise TypeError("outer_holdout must be OuterHoldout")
        if len(self.outer_holdout.session_ids) != self.outer_holdout_session_count:
            raise ValueError("outer holdout session count differs from its policy")
        if tuple(item.fold_number for item in self.folds) != tuple(range(1, len(self.folds) + 1)):
            raise ValueError("fold numbers must be contiguous and one-based")
        self._validate_expansion()

    def _validate_expansion(self) -> None:
        holdout_sessions = set(self.outer_holdout.session_ids)
        holdout_examples = set(self.outer_holdout.example_ids)
        previous: WalkForwardFold | None = None
        for fold in self.folds:
            if len(fold.validation_session_ids) != self.validation_session_count:
                raise ValueError("a fold validation session count differs from policy")
            if holdout_sessions & (set(fold.train_session_ids) | set(fold.validation_session_ids)):
                raise ValueError("outer holdout session leaked into a development fold")
            fold_examples = (
                set(fold.train_example_ids)
                | set(fold.validation_example_ids)
                | set(fold.train_purged_example_ids)
                | set(fold.validation_purged_example_ids)
            )
            if holdout_examples & fold_examples:
                raise ValueError("outer holdout example leaked into a development fold")
            if previous is None:
                if len(fold.train_session_ids) != self.initial_train_session_count:
                    raise ValueError("first fold train session count differs from policy")
            else:
                expected = (*previous.train_session_ids, *previous.validation_session_ids)
                if fold.train_session_ids != expected:
                    raise ValueError("training sessions do not expand by prior validation sessions")
            previous = fold

    def fold(self, fold_number: int) -> WalkForwardFold:
        if isinstance(fold_number, bool) or not isinstance(fold_number, int):
            raise TypeError("fold_number must be an integer")
        if fold_number <= 0 or fold_number > len(self.folds):
            raise KeyError(fold_number)
        return self.folds[fold_number - 1]

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "interval_semantics": self.interval_semantics,
            "source_dataset_hash": self.source_dataset_hash,
            "initial_train_session_count": self.initial_train_session_count,
            "validation_session_count": self.validation_session_count,
            "outer_holdout_session_count": self.outer_holdout_session_count,
            "embargo_ns": self.embargo_ns,
            "folds": [item.record() for item in self.folds],
            "outer_holdout": self.outer_holdout.record(),
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("ascii")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json({"artifact_sha256": self.artifact_sha256, "payload": self.payload()})

    @classmethod
    def from_json(cls, value: str) -> WalkForwardManifest:
        parsed = _parse_strict_json(value)
        root = _require_object(parsed, "walk-forward manifest", {"artifact_sha256", "payload"})
        expected_hash = _require_string(root["artifact_sha256"], "artifact_sha256")
        _validate_sha256(expected_hash, "artifact_sha256")
        payload = _require_object(
            root["payload"],
            "walk-forward payload",
            {
                "schema_version",
                "method",
                "interval_semantics",
                "source_dataset_hash",
                "initial_train_session_count",
                "validation_session_count",
                "outer_holdout_session_count",
                "embargo_ns",
                "folds",
                "outer_holdout",
            },
        )
        manifest = cls(
            schema_version=_require_integer(payload["schema_version"], "schema_version"),
            method=_require_string(payload["method"], "method"),
            interval_semantics=_require_string(payload["interval_semantics"], "interval_semantics"),
            source_dataset_hash=_require_string(
                payload["source_dataset_hash"], "source_dataset_hash"
            ),
            initial_train_session_count=_require_integer(
                payload["initial_train_session_count"], "initial_train_session_count"
            ),
            validation_session_count=_require_integer(
                payload["validation_session_count"], "validation_session_count"
            ),
            outer_holdout_session_count=_require_integer(
                payload["outer_holdout_session_count"], "outer_holdout_session_count"
            ),
            embargo_ns=_require_integer(payload["embargo_ns"], "embargo_ns"),
            folds=tuple(
                WalkForwardFold.from_record(item)
                for item in _require_list(payload["folds"], "folds")
            ),
            outer_holdout=OuterHoldout.from_record(payload["outer_holdout"]),
        )
        if manifest.artifact_sha256 != expected_hash:
            raise ValueError("walk-forward artifact SHA-256 does not match its payload")
        if value != manifest.to_json():
            raise ValueError("walk-forward manifest JSON is not in canonical form")
        return manifest

    def validate(self, examples: Iterable[ExampleDependency]) -> WalkForwardManifest:
        """Rebuild every assignment and exact purge decision from dependency intervals."""

        rebuilt = build_expanding_walk_forward(
            examples,
            source_dataset_hash=self.source_dataset_hash,
            initial_train_session_count=self.initial_train_session_count,
            validation_session_count=self.validation_session_count,
            outer_holdout_session_count=self.outer_holdout_session_count,
            embargo_ns=self.embargo_ns,
        )
        if rebuilt.to_json() != self.to_json():
            raise ValueError("walk-forward manifest differs from supplied dependencies")
        return self


@dataclass(frozen=True, slots=True)
class OuterHoldoutAccess:
    """Paths and identities required for one verified final-holdout consumption."""

    split_sha256: str
    frozen_candidate_sha256: str
    freeze_evidence_sha256: str
    frozen_candidate_path: Path
    freeze_evidence_path: Path
    purpose: str = OUTER_HOLDOUT_PURPOSE

    def __post_init__(self) -> None:
        for name in (
            "split_sha256",
            "frozen_candidate_sha256",
            "freeze_evidence_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        for name in (
            "frozen_candidate_path",
            "freeze_evidence_path",
        ):
            path = getattr(self, name)
            if not isinstance(path, Path):
                raise TypeError(f"{name} must be pathlib.Path")
            _validate_absolute_normalized_path(path, name)
        if self.purpose != OUTER_HOLDOUT_PURPOSE:
            raise ValueError(
                "outer holdout access is restricted to final frozen-candidate evaluation"
            )


def build_expanding_walk_forward(
    examples: Iterable[ExampleDependency],
    *,
    source_dataset_hash: str,
    initial_train_session_count: int,
    validation_session_count: int,
    outer_holdout_session_count: int,
    embargo_ns: int = 0,
) -> WalkForwardManifest:
    """Build all complete expanding folds ending immediately before the outer holdout."""

    _validate_sha256(source_dataset_hash, "source_dataset_hash")
    counts: dict[str, int] = {}
    for name, value in (
        ("initial_train_session_count", initial_train_session_count),
        ("validation_session_count", validation_session_count),
        ("outer_holdout_session_count", outer_holdout_session_count),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        counts[name] = value
    if isinstance(embargo_ns, bool) or not isinstance(embargo_ns, int) or embargo_ns < 0:
        raise ValueError("embargo_ns must be a non-negative integer")

    ordered, grouped, sessions = _order_examples(examples)
    development_count = len(sessions) - counts["outer_holdout_session_count"]
    remaining = development_count - counts["initial_train_session_count"]
    if remaining < counts["validation_session_count"]:
        raise ValueError("not enough development sessions for one complete validation fold")
    if remaining % counts["validation_session_count"]:
        raise ValueError(
            "development sessions must be exactly consumed by complete validation windows"
        )
    fold_count = remaining // counts["validation_session_count"]
    outer_sessions = tuple(sessions[development_count:])
    outer_examples = _examples_for_sessions(outer_sessions, grouped)
    outer_holdout = OuterHoldout(
        session_ids=outer_sessions,
        example_ids=tuple(item.example_id for item in outer_examples),
    )

    folds: list[WalkForwardFold] = []
    for fold_index in range(fold_count):
        train_end = counts["initial_train_session_count"] + (
            fold_index * counts["validation_session_count"]
        )
        validation_end = train_end + counts["validation_session_count"]
        train_sessions = tuple(sessions[:train_end])
        validation_sessions = tuple(sessions[train_end:validation_end])
        raw_train = _examples_for_sessions(train_sessions, grouped)
        raw_validation = _examples_for_sessions(validation_sessions, grouped)

        retained_validation, validation_purges = _purge_earlier(
            raw_validation,
            (("outer_holdout", outer_examples),),
            assigned_partition="validation",
            embargo_ns=embargo_ns,
        )
        retained_train, train_purges = _purge_earlier(
            raw_train,
            (
                ("validation", retained_validation),
                ("outer_holdout", outer_examples),
            ),
            assigned_partition="train",
            embargo_ns=embargo_ns,
        )
        if not retained_train or not retained_validation:
            raise ValueError(
                f"fold {fold_index + 1} has no retained train or validation examples after purge"
            )
        folds.append(
            WalkForwardFold(
                fold_number=fold_index + 1,
                train_session_ids=train_sessions,
                validation_session_ids=validation_sessions,
                train_example_ids=tuple(item.example_id for item in retained_train),
                validation_example_ids=tuple(item.example_id for item in retained_validation),
                train_purged_example_ids=tuple(item.example_id for item, _ in train_purges),
                validation_purged_example_ids=tuple(
                    item.example_id for item, _ in validation_purges
                ),
                purges=tuple(record for _, record in (*train_purges, *validation_purges)),
            )
        )

    manifest = WalkForwardManifest(
        source_dataset_hash=source_dataset_hash,
        initial_train_session_count=counts["initial_train_session_count"],
        validation_session_count=counts["validation_session_count"],
        outer_holdout_session_count=counts["outer_holdout_session_count"],
        embargo_ns=embargo_ns,
        folds=tuple(folds),
        outer_holdout=outer_holdout,
    )
    if set(item.example_id for item in ordered) != (
        set(outer_holdout.example_ids)
        | {
            example_id
            for fold in manifest.folds
            for example_id in (
                *fold.train_example_ids,
                *fold.validation_example_ids,
                *fold.train_purged_example_ids,
                *fold.validation_purged_example_ids,
            )
        }
    ):
        raise RuntimeError("walk-forward construction lost dataset membership")
    return manifest


def iter_fold_batches(
    dataset: ArrowDataset,
    manifest: WalkForwardManifest,
    *,
    fold_number: int,
    partition: Literal["train", "validation"],
    batch_size: int,
    seed: int | None = None,
    drop_last: bool = False,
) -> Iterator[MaterializedBatch]:
    """Materialize a fold without exposing the outer-holdout partition."""

    _validate_dataset_binding(dataset, manifest, allow_local_research=True)
    fold = manifest.fold(fold_number)
    if partition == "train":
        return dataset.iter_batches(
            fold.train_example_ids,
            batch_size=batch_size,
            partition="train",
            seed=seed,
            drop_last=drop_last,
        )
    if partition == "validation":
        return dataset.iter_batches(
            fold.validation_example_ids,
            batch_size=batch_size,
            partition="evaluation",
            seed=seed,
            drop_last=drop_last,
        )
    raise ValueError("partition must be train or validation; outer holdout is separately sealed")


def iter_outer_holdout_batches(
    dataset: ArrowDataset,
    manifest: WalkForwardManifest,
    *,
    access: OuterHoldoutAccess,
    batch_size: int,
) -> Iterator[MaterializedBatch]:
    """Claim and materialize the holdout once for a verified frozen candidate."""

    _validate_dataset_binding(dataset, manifest, allow_local_research=False)
    if not isinstance(access, OuterHoldoutAccess):
        raise TypeError("access must be OuterHoldoutAccess")
    if access.split_sha256 != manifest.artifact_sha256:
        raise PermissionError("outer holdout access is bound to a different split manifest")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    candidate_path = _verify_frozen_artifact(
        access.frozen_candidate_path,
        access.frozen_candidate_sha256,
        "frozen candidate artifact",
    )
    evidence_path = _verify_frozen_artifact(
        access.freeze_evidence_path,
        access.freeze_evidence_sha256,
        "freeze evidence artifact",
    )
    if candidate_path == evidence_path or os.path.samefile(candidate_path, evidence_path):
        raise ValueError("frozen candidate and freeze evidence must be distinct artifacts")
    _claim_outer_holdout_consumption(
        access,
        manifest,
        outer_holdout_consumption_ledger_path(dataset, manifest),
    )
    return dataset.iter_batches(
        manifest.outer_holdout.example_ids,
        batch_size=batch_size,
        partition="evaluation",
    )


def _validate_dataset_binding(
    dataset: ArrowDataset,
    manifest: WalkForwardManifest,
    *,
    allow_local_research: bool,
) -> None:
    if not isinstance(dataset, ArrowDataset):
        raise TypeError("dataset must be a verified ArrowDataset")
    if not isinstance(manifest, WalkForwardManifest):
        raise TypeError("manifest must be WalkForwardManifest")
    if dataset.manifest.dataset_id != manifest.source_dataset_hash:
        raise ValueError("walk-forward source_dataset_hash differs from the opened dataset")
    if allow_local_research:
        dataset.require_development_access()
    else:
        dataset.manifest.capabilities.require_training()


def _verify_frozen_artifact(path: Path, expected_hash: str, name: str) -> Path:
    _validate_absolute_normalized_path(path, name)
    before = _validate_non_symlink_path(path, name, final_kind="file")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} could not be opened as a regular non-symlink file") from error
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise ValueError(f"{name} must be a regular non-symlink file")
        if not os.path.samestat(before, opened_before):
            raise ValueError(f"{name} changed while it was being verified")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
        opened_after = os.fstat(descriptor)
        if _file_state(opened_before) != _file_state(opened_after):
            raise ValueError(f"{name} changed while it was being verified")
    finally:
        os.close(descriptor)
    after = _validate_non_symlink_path(path, name, final_kind="file")
    if not os.path.samestat(opened_after, after):
        raise ValueError(f"{name} changed while it was being verified")
    if digest.hexdigest() != expected_hash:
        raise ValueError(f"{name} SHA-256 does not match the declared hash")
    return path.resolve(strict=True)


def outer_holdout_consumption_ledger_path(
    dataset: ArrowDataset, manifest: WalkForwardManifest
) -> Path:
    """Return the one canonical consumption ledger for this dataset and split."""

    _validate_dataset_binding(dataset, manifest, allow_local_research=False)
    return dataset.root / (f".chmtq-outer-holdout-{manifest.artifact_sha256}.consumed.json")


def _claim_outer_holdout_consumption(
    access: OuterHoldoutAccess, manifest: WalkForwardManifest, path: Path
) -> None:
    _validate_absolute_normalized_path(path, "consumption_ledger_path")
    _validate_non_symlink_path(
        path.parent,
        "outer holdout consumption ledger parent",
        final_kind="directory",
    )
    try:
        existing = path.lstat()
    except FileNotFoundError:
        pass
    except OSError as error:
        raise ValueError("outer holdout consumption ledger path cannot be inspected") from error
    else:
        if _is_link_or_reparse(existing):
            raise ValueError(
                "outer holdout consumption ledger path must not be a symbolic link or reparse point"
            )
        if not stat.S_ISREG(existing.st_mode):
            raise ValueError(
                "outer holdout consumption ledger path is occupied by a non-regular file"
            )
        raise PermissionError("outer holdout has already been consumed for this ledger")

    record = {
        "schema_version": OUTER_HOLDOUT_LEDGER_SCHEMA_VERSION,
        "purpose": access.purpose,
        "split_sha256": access.split_sha256,
        "outer_holdout_sha256": manifest.outer_holdout.holdout_sha256,
        "frozen_candidate_sha256": access.frozen_candidate_sha256,
        "freeze_evidence_sha256": access.freeze_evidence_sha256,
    }
    content = (_canonical_json(record) + "\n").encode("ascii")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as error:
        try:
            existing = path.lstat()
        except OSError:
            existing = None
        if existing is not None and _is_link_or_reparse(existing):
            raise ValueError(
                "outer holdout consumption ledger path must not be a symbolic link or reparse point"
            ) from error
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError(
                "outer holdout consumption ledger path is occupied by a non-regular file"
            ) from error
        raise PermissionError("outer holdout has already been consumed for this ledger") from error
    except OSError as error:
        raise ValueError("outer holdout consumption ledger could not be created") from error
    try:
        offset = 0
        while offset < len(content):
            written = os.write(descriptor, content[offset:])
            if written <= 0:
                raise OSError("ledger write made no progress")
            offset += written
        os.fsync(descriptor)
        created = os.fstat(descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise RuntimeError("outer holdout consumption ledger is not a regular file")
    except Exception as error:
        raise RuntimeError(
            "outer holdout consumption ledger creation failed after its one-shot claim"
        ) from error
    finally:
        os.close(descriptor)

    on_disk = _validate_non_symlink_path(
        path,
        "outer holdout consumption ledger",
        final_kind="file",
    )
    if not os.path.samestat(created, on_disk):
        raise RuntimeError("outer holdout consumption ledger changed during creation")


def _validate_absolute_normalized_path(path: Path, name: str) -> None:
    if not path.is_absolute() or ".." in path.parts or str(path) != os.path.normpath(path):
        raise ValueError(f"{name} must be an absolute normalized path")


def _validate_non_symlink_path(
    path: Path, name: str, *, final_kind: Literal["file", "directory"]
) -> os.stat_result:
    chain: list[Path] = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    for index, component in enumerate(reversed(chain)):
        try:
            details = component.lstat()
        except FileNotFoundError as error:
            raise ValueError(f"{name} is missing") from error
        except OSError as error:
            raise ValueError(f"{name} path cannot be inspected") from error
        if _is_link_or_reparse(details):
            raise ValueError(f"{name} path must not use symbolic links or reparse points")
        is_final = index == len(chain) - 1
        if not is_final and not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"{name} parent path must contain only directories")
    if final_kind == "file" and not stat.S_ISREG(details.st_mode):
        raise ValueError(f"{name} must be a regular non-symlink file")
    if final_kind == "directory" and not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"{name} must be a directory")
    return details


def _is_link_or_reparse(details: os.stat_result) -> bool:
    file_attributes = getattr(details, "st_file_attributes", 0)
    return stat.S_ISLNK(details.st_mode) or bool(file_attributes & 0x400)


def _file_state(details: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_size,
        details.st_mtime_ns,
        details.st_ctime_ns,
    )


def _order_examples(
    examples: Iterable[ExampleDependency],
) -> tuple[
    tuple[ExampleDependency, ...],
    dict[str, tuple[ExampleDependency, ...]],
    tuple[str, ...],
]:
    materialized = tuple(examples)
    if not materialized:
        raise ValueError("at least one dependency example is required")
    if any(not isinstance(item, ExampleDependency) for item in materialized):
        raise TypeError("examples must be ExampleDependency instances")
    ids = tuple(item.example_id for item in materialized)
    if len(set(ids)) != len(ids):
        raise ValueError("example_id values must be unique")
    ordered = tuple(sorted(materialized, key=_example_key))
    grouped_mutable: dict[str, list[ExampleDependency]] = {}
    for item in ordered:
        grouped_mutable.setdefault(item.session_id, []).append(item)
    sessions = tuple(
        sorted(
            grouped_mutable,
            key=lambda session: (
                grouped_mutable[session][0].decision_ns,
                session,
            ),
        )
    )
    for left, right in zip(sessions, sessions[1:]):
        if grouped_mutable[left][-1].decision_ns >= grouped_mutable[right][0].decision_ns:
            raise ValueError("session decision ranges interleave or share a boundary")
    grouped = {session: tuple(grouped_mutable[session]) for session in sessions}
    return ordered, grouped, sessions


def _examples_for_sessions(
    sessions: tuple[str, ...], grouped: Mapping[str, tuple[ExampleDependency, ...]]
) -> tuple[ExampleDependency, ...]:
    return tuple(item for session in sessions for item in grouped[session])


def _purge_earlier(
    earlier_examples: tuple[ExampleDependency, ...],
    later_partitions: tuple[tuple[str, tuple[ExampleDependency, ...]], ...],
    *,
    assigned_partition: str,
    embargo_ns: int,
) -> tuple[
    tuple[ExampleDependency, ...],
    tuple[tuple[ExampleDependency, FoldPurge], ...],
]:
    boundaries = tuple(
        (
            partition,
            min(examples, key=lambda item: (item.dependency_start_ns, *_example_key(item))),
        )
        for partition, examples in later_partitions
        if examples
    )
    retained: list[ExampleDependency] = []
    purged: list[tuple[ExampleDependency, FoldPurge]] = []
    for earlier in earlier_examples:
        record: FoldPurge | None = None
        for partition, boundary in boundaries:
            reason = _conflict_reason(earlier, boundary, embargo_ns)
            if reason is not None:
                record = FoldPurge(
                    example_id=earlier.example_id,
                    assigned_partition=assigned_partition,
                    against_partition=partition,
                    against_example_id=boundary.example_id,
                    reason=reason,
                )
                break
        if record is None:
            retained.append(earlier)
        else:
            purged.append((earlier, record))
    return tuple(retained), tuple(purged)


def _conflict_reason(
    earlier: ExampleDependency, later: ExampleDependency, embargo_ns: int
) -> str | None:
    if earlier.decision_ns >= later.decision_ns:
        raise ValueError("purge comparison must place the earlier decision first")
    if earlier.overlaps(later):
        return "dependency_overlap"
    if (
        embargo_ns > 0
        and earlier.dependency_end_ns <= later.dependency_start_ns
        and earlier.dependency_end_ns > later.dependency_start_ns - embargo_ns
    ):
        return "embargo"
    return None


def _example_key(example: ExampleDependency) -> tuple[int, int, int, str]:
    return (
        example.decision_ns,
        example.dependency_start_ns,
        example.dependency_end_ns,
        example.example_id,
    )


def _validate_unique(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate values")


def _validate_unique_nonempty(values: tuple[str, ...], name: str) -> None:
    _validate_unique(values, name)
    if not values:
        raise ValueError(f"{name} must not be empty")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    return tuple(_require_string(item, name) for item in _require_list(value, name))


def _require_object(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _parse_strict_json(value: str) -> Any:
    if not isinstance(value, str):
        raise TypeError("manifest JSON must be a string")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    def reject_constant(constant: str) -> None:
        raise ValueError(f"non-finite JSON constant is forbidden: {constant}")

    try:
        return json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError("manifest is not valid JSON") from error
