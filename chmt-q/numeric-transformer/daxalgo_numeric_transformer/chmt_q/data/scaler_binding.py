"""Strict dataset/fold binding for train-only CHMT-Q robust-scaler fitting.

The binding layer deliberately accepts exact example IDs rather than a partition
name. This keeps split and outer-holdout selection outside the scaler while still
proving which ordered training examples produced its statistics.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import ChmtQConfig
from ..contracts import ChmtQBatch
from ..scaling import (
    DEFAULT_SCALER_CHUNK_VALUES,
    RobustScalerArtifact,
    fit_robust_scaler,
)
from .dataset import ArrowDataset, MaterializedBatch

SCALER_BINDING_SCHEMA_VERSION = 1
SCALER_BINDING_METHOD = "ordered-train-example-ids-sha256-v1"
MAX_SCALER_FIT_BATCH_SIZE = 1024

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FOLD_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


@dataclass(frozen=True, slots=True)
class FoldBoundScalerArtifact:
    """A robust scaler cryptographically bound to one dataset split and fold."""

    dataset_id: str
    split_manifest_sha256: str
    fold_id: str
    ordered_train_example_ids_sha256: str
    train_example_count: int
    scaler: RobustScalerArtifact
    train_example_ids: tuple[str, ...] = field(repr=False)
    schema_version: int = SCALER_BINDING_SCHEMA_VERSION
    binding_method: str = SCALER_BINDING_METHOD
    _train_example_id_set: frozenset[str] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_sha256(self.dataset_id, "dataset_id")
        _validate_sha256(self.split_manifest_sha256, "split_manifest_sha256")
        _validate_sha256(
            self.ordered_train_example_ids_sha256,
            "ordered_train_example_ids_sha256",
        )
        _validate_fold_id(self.fold_id)
        if self.schema_version != SCALER_BINDING_SCHEMA_VERSION:
            raise ValueError("unsupported scaler-binding schema_version")
        if self.binding_method != SCALER_BINDING_METHOD:
            raise ValueError("unsupported scaler-binding method")
        if isinstance(self.train_example_count, bool) or not isinstance(
            self.train_example_count, int
        ):
            raise TypeError("train_example_count must be an integer")
        if not isinstance(self.train_example_ids, tuple):
            raise TypeError("train_example_ids must be an immutable tuple")
        train_ids = _validate_example_ids(
            self.train_example_ids,
            name="train_example_ids",
        )
        if self.train_example_count != len(train_ids):
            raise ValueError("train_example_count differs from the supplied train IDs")
        observed_hash = ordered_train_example_ids_sha256(train_ids)
        if observed_hash != self.ordered_train_example_ids_sha256:
            raise ValueError("ordered train example-ID SHA-256 does not match")
        if not isinstance(self.scaler, RobustScalerArtifact):
            raise TypeError("scaler must be a RobustScalerArtifact")
        if self.scaler.source_batch_set_hash != observed_hash:
            raise ValueError("inner scaler source hash differs from the ordered train IDs")
        object.__setattr__(self, "_train_example_id_set", frozenset(train_ids))

    def payload(self) -> dict[str, Any]:
        """Return the canonical, bounded binding payload without duplicating split IDs."""

        return {
            "schema_version": self.schema_version,
            "binding_method": self.binding_method,
            "dataset_id": self.dataset_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "fold_id": self.fold_id,
            "ordered_train_example_ids_sha256": self.ordered_train_example_ids_sha256,
            "train_example_count": self.train_example_count,
            "inner_scaler": json.loads(self.scaler.to_json()),
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("ascii")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json({"artifact_sha256": self.artifact_sha256, "payload": self.payload()})

    @classmethod
    def from_json(
        cls,
        value: str,
        *,
        expected_dataset_id: str,
        expected_split_manifest_sha256: str,
        expected_fold_id: str,
        ordered_train_example_ids: Iterable[str],
    ) -> FoldBoundScalerArtifact:
        """Load only when serialized and caller-supplied bindings agree exactly."""

        _validate_sha256(expected_dataset_id, "expected_dataset_id")
        _validate_sha256(
            expected_split_manifest_sha256,
            "expected_split_manifest_sha256",
        )
        _validate_fold_id(expected_fold_id, name="expected_fold_id")
        train_ids = _validate_example_ids(
            ordered_train_example_ids,
            name="ordered_train_example_ids",
        )
        parsed = _parse_strict_json(value)
        root = _require_object(
            parsed,
            "scaler binding",
            {"artifact_sha256", "payload"},
        )
        expected_artifact_hash = _require_string(
            root["artifact_sha256"],
            "artifact_sha256",
        )
        _validate_sha256(expected_artifact_hash, "artifact_sha256")
        payload = _require_object(
            root["payload"],
            "scaler-binding payload",
            {
                "schema_version",
                "binding_method",
                "dataset_id",
                "split_manifest_sha256",
                "fold_id",
                "ordered_train_example_ids_sha256",
                "train_example_count",
                "inner_scaler",
            },
        )
        inner_scaler = RobustScalerArtifact.from_json(_canonical_json(payload["inner_scaler"]))
        artifact = cls(
            schema_version=_require_integer(payload["schema_version"], "schema_version"),
            binding_method=_require_string(payload["binding_method"], "binding_method"),
            dataset_id=_require_string(payload["dataset_id"], "dataset_id"),
            split_manifest_sha256=_require_string(
                payload["split_manifest_sha256"],
                "split_manifest_sha256",
            ),
            fold_id=_require_string(payload["fold_id"], "fold_id"),
            ordered_train_example_ids_sha256=_require_string(
                payload["ordered_train_example_ids_sha256"],
                "ordered_train_example_ids_sha256",
            ),
            train_example_count=_require_integer(
                payload["train_example_count"],
                "train_example_count",
            ),
            scaler=inner_scaler,
            train_example_ids=train_ids,
        )
        if artifact.artifact_sha256 != expected_artifact_hash:
            raise ValueError("scaler-binding artifact SHA-256 does not match its payload")
        if artifact.dataset_id != expected_dataset_id:
            raise ValueError("scaler binding differs from expected_dataset_id")
        if artifact.split_manifest_sha256 != expected_split_manifest_sha256:
            raise ValueError("scaler binding differs from expected_split_manifest_sha256")
        if artifact.fold_id != expected_fold_id:
            raise ValueError("scaler binding differs from expected_fold_id")
        if value != artifact.to_json():
            raise ValueError("scaler-binding JSON is not in canonical form")
        return artifact

    def normalize_batch(
        self,
        materialized: MaterializedBatch,
        *,
        permitted_example_ids: Iterable[str],
        config: ChmtQConfig | None = None,
    ) -> MaterializedBatch:
        """Normalize one batch after an explicit caller-owned ID authorization."""

        if not isinstance(materialized, MaterializedBatch):
            raise TypeError("materialized must be a MaterializedBatch")
        permitted = frozenset(
            _validate_example_ids(
                permitted_example_ids,
                name="permitted_example_ids",
            )
        )
        batch_ids = _validate_example_ids(
            materialized.example_ids,
            name="materialized.example_ids",
        )
        forbidden = tuple(example_id for example_id in batch_ids if example_id not in permitted)
        if forbidden:
            raise PermissionError(
                f"materialized batch contains non-permitted example IDs: {forbidden}"
            )
        return replace(
            materialized,
            batch=self.scaler.normalize_batch(materialized.batch, config),
        )

    def normalize_train_batch(
        self,
        materialized: MaterializedBatch,
        *,
        config: ChmtQConfig | None = None,
    ) -> MaterializedBatch:
        """Normalize only examples cryptographically bound as this fold's training set."""

        return self.normalize_batch(
            materialized,
            permitted_example_ids=self._train_example_id_set,
            config=config,
        )


def ordered_train_example_ids_sha256(example_ids: Iterable[str]) -> str:
    """Hash an exact ordered, unique, non-empty train example-ID sequence."""

    ordered = _validate_example_ids(example_ids, name="ordered_train_example_ids")
    payload = {
        "binding_method": SCALER_BINDING_METHOD,
        "ordered_train_example_ids": list(ordered),
    }
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def fit_fold_bound_scaler(
    dataset: ArrowDataset,
    *,
    dataset_id: str,
    split_manifest_sha256: str,
    fold_id: str,
    ordered_train_example_ids: Iterable[str],
    batch_size: int,
    minimum_scale: float = 1.0e-6,
    config: ChmtQConfig | None = None,
    identity_indices: dict[str, tuple[int, ...]] | None = None,
    spool_directory: str | Path | None = None,
    max_chunk_values: int = DEFAULT_SCALER_CHUNK_VALUES,
) -> FoldBoundScalerArtifact:
    """Fit only the exact caller-supplied train IDs from one verified dataset.

    There is intentionally no partition or role argument. The caller obtains the
    train IDs from its verified split manifest; this function checks their exact
    dataset order and uses a fixed chronological materialization strategy.
    """

    if not isinstance(dataset, ArrowDataset):
        raise TypeError("dataset must be a verified ArrowDataset")
    _validate_sha256(dataset_id, "dataset_id")
    _validate_sha256(split_manifest_sha256, "split_manifest_sha256")
    _validate_fold_id(fold_id)
    if dataset.manifest.dataset_id != dataset_id:
        raise ValueError("dataset identity differs from dataset_id")
    dataset.require_development_access()
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or batch_size > MAX_SCALER_FIT_BATCH_SIZE
    ):
        raise ValueError(f"batch_size must be between 1 and {MAX_SCALER_FIT_BATCH_SIZE}")

    train_ids = _validate_example_ids(
        ordered_train_example_ids,
        name="ordered_train_example_ids",
    )
    dataset_ids = _validate_example_ids(dataset.example_ids, name="dataset.example_ids")
    dataset_id_set = frozenset(dataset_ids)
    unknown = tuple(example_id for example_id in train_ids if example_id not in dataset_id_set)
    if unknown:
        raise KeyError(f"train example IDs are absent from the dataset: {unknown}")
    train_id_set = frozenset(train_ids)
    canonical_train_order = tuple(
        example_id for example_id in dataset_ids if example_id in train_id_set
    )
    if train_ids != canonical_train_order:
        raise ValueError("train example IDs must follow exact dataset chronological order")

    train_hash = ordered_train_example_ids_sha256(train_ids)
    scaler = fit_robust_scaler(
        _verified_train_batches(dataset, train_ids, batch_size=batch_size),
        source_batch_set_hash=train_hash,
        minimum_scale=minimum_scale,
        config=config,
        identity_indices=identity_indices,
        spool_directory=spool_directory,
        max_chunk_values=max_chunk_values,
    )
    return FoldBoundScalerArtifact(
        dataset_id=dataset_id,
        split_manifest_sha256=split_manifest_sha256,
        fold_id=fold_id,
        ordered_train_example_ids_sha256=train_hash,
        train_example_count=len(train_ids),
        scaler=scaler,
        train_example_ids=train_ids,
    )


def _verified_train_batches(
    dataset: ArrowDataset,
    train_ids: tuple[str, ...],
    *,
    batch_size: int,
) -> Iterator[ChmtQBatch]:
    allowed = frozenset(train_ids)
    offset = 0
    for materialized in dataset.iter_batches(
        train_ids,
        batch_size=batch_size,
        partition="evaluation",
        seed=None,
        drop_last=False,
    ):
        if not isinstance(materialized, MaterializedBatch):
            raise TypeError("dataset yielded a non-MaterializedBatch value")
        batch_ids = _validate_example_ids(
            materialized.example_ids,
            name="materialized.example_ids",
        )
        forbidden = tuple(example_id for example_id in batch_ids if example_id not in allowed)
        if forbidden:
            raise PermissionError(f"scaler fit received non-train example IDs: {forbidden}")
        expected = train_ids[offset : offset + len(batch_ids)]
        if batch_ids != expected:
            raise ValueError("scaler-fit batches do not exactly cover train IDs in order")
        offset += len(batch_ids)
        yield materialized.batch
    if offset != len(train_ids):
        raise ValueError("scaler-fit batches did not cover every train example ID exactly once")


def _validate_example_ids(values: Iterable[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of example IDs, not a string")
    try:
        materialized = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    if not materialized:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(value, str) or not value for value in materialized):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(materialized)) != len(materialized):
        raise ValueError(f"{name} contains duplicate example IDs")
    return materialized


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


def _validate_fold_id(value: Any, *, name: str = "fold_id") -> None:
    if not isinstance(value, str) or _FOLD_ID.fullmatch(value) is None:
        raise ValueError(f"{name} must be a short lowercase stable identifier")


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_object(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _parse_strict_json(value: str) -> Any:
    if not isinstance(value, str):
        raise TypeError("scaler-binding JSON must be a string")

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
        raise ValueError("scaler binding is not valid JSON") from error
