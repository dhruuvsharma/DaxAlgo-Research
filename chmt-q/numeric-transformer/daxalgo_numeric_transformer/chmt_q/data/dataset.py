"""Verified Arrow IPC datasets for already-engineered CHMT-Q examples.

This module deliberately does not build features from market events.  It verifies and
materializes the fixed CHMT-Q tensor interchange contract emitted by the Pro compiler.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

import torch

from ..contracts import ChmtQBatch
from ..splits import ExampleDependency

if TYPE_CHECKING:
    from .research_authorization import LocalResearchAuthorization

DATASET_MANIFEST_NAME = "dataset.json"
DATASET_SCHEMA_VERSION = 2
DATASET_FORMAT = "arrow-ipc-file-chmt-q-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}")
_EVIDENCE_PURPOSES = ("readiness", "rights", "provenance")
MAX_RECORD_BATCH_ROWS = 64


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


_FEATURE_CONTRACT = (
    {"name": "bars", "dtype": "float32", "shape": [64, 16]},
    {"name": "bars_mask", "dtype": "bool", "shape": [64]},
    {"name": "l1", "dtype": "float32", "shape": [240, 16]},
    {"name": "l1_mask", "dtype": "bool", "shape": [240]},
    {"name": "l2", "dtype": "float32", "shape": [240, 2, 10, 12]},
    {"name": "l2_level_mask", "dtype": "bool", "shape": [240, 2, 10]},
    {"name": "l2_snapshot_mask", "dtype": "bool", "shape": [240]},
    {"name": "quality", "dtype": "float32", "shape": [16]},
)
_TARGET_CONTRACT = (
    {"name": "targets", "dtype": "float32", "shape": [5, 5]},
    {"name": "target_mask", "dtype": "bool", "shape": [5, 5]},
    {"name": "presence_labels", "dtype": "float32", "shape": [5, 2]},
    {"name": "presence_mask", "dtype": "bool", "shape": [5, 2]},
)
FEATURE_SCHEMA_SHA256 = hashlib.sha256(
    _canonical_json(_FEATURE_CONTRACT).encode("ascii")
).hexdigest()
TARGET_SCHEMA_SHA256 = hashlib.sha256(_canonical_json(_TARGET_CONTRACT).encode("ascii")).hexdigest()
TENSOR_SCHEMA_SHA256 = hashlib.sha256(
    _canonical_json(
        {
            "contract": "chmt-q-v1",
            "features": _FEATURE_CONTRACT,
            "targets": _TARGET_CONTRACT,
        }
    ).encode("ascii")
).hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetEvidenceArtifact:
    """One reviewable, content-addressed evidence file stored beside the dataset."""

    purpose: Literal["readiness", "rights", "provenance"]
    path: str
    byte_size: int
    sha256: str

    def __post_init__(self) -> None:
        if self.purpose not in _EVIDENCE_PURPOSES:
            raise ValueError("evidence purpose must be readiness, rights, or provenance")
        _validate_relative_evidence_path(self.path)
        if _require_integer(self.byte_size, "byte_size") <= 0:
            raise ValueError("evidence byte_size must be positive")
        _validate_sha256(self.sha256, "evidence sha256")

    def to_payload(self) -> dict[str, Any]:
        return {
            "purpose": self.purpose,
            "path": self.path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> DatasetEvidenceArtifact:
        value = _require_object(
            payload,
            "evidence artifact",
            {"purpose", "path", "byte_size", "sha256"},
        )
        return cls(
            purpose=_require_string(value["purpose"], "purpose"),
            path=_require_string(value["path"], "path"),
            byte_size=_require_integer(value["byte_size"], "byte_size"),
            sha256=_require_string(value["sha256"], "sha256"),
        )


@dataclass(frozen=True, slots=True)
class DatasetCapabilities:
    """Fail-closed training gates and their local content-addressed evidence."""

    training_ready: bool
    rights_confirmed: bool
    provenance_complete: bool
    readiness_evidence_sha256: str
    rights_evidence_sha256: str
    provenance_evidence_sha256: str
    evidence_artifacts: tuple[DatasetEvidenceArtifact, ...] = ()

    def __post_init__(self) -> None:
        for name in ("training_ready", "rights_confirmed", "provenance_complete"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a boolean")
        for name in (
            "readiness_evidence_sha256",
            "rights_evidence_sha256",
            "provenance_evidence_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        if not isinstance(self.evidence_artifacts, tuple) or any(
            not isinstance(item, DatasetEvidenceArtifact) for item in self.evidence_artifacts
        ):
            raise TypeError("evidence_artifacts must be a tuple of DatasetEvidenceArtifact")
        if self.evidence_artifacts:
            purposes = tuple(item.purpose for item in self.evidence_artifacts)
            if purposes != _EVIDENCE_PURPOSES:
                raise ValueError(
                    "evidence_artifacts must contain readiness, rights, and provenance in order"
                )
            paths = tuple(item.path.casefold() for item in self.evidence_artifacts)
            if len(set(paths)) != len(paths):
                raise ValueError("evidence_artifacts contains duplicate paths")
            expected_hashes = (
                self.readiness_evidence_sha256,
                self.rights_evidence_sha256,
                self.provenance_evidence_sha256,
            )
            if tuple(item.sha256 for item in self.evidence_artifacts) != expected_hashes:
                raise ValueError(
                    "evidence artifact hashes do not match the capability evidence hashes"
                )

    def require_training(self) -> None:
        failed = tuple(
            name
            for name, satisfied in (
                ("training_ready", self.training_ready),
                ("rights_confirmed", self.rights_confirmed),
                ("provenance_complete", self.provenance_complete),
            )
            if not satisfied
        )
        if tuple(item.purpose for item in self.evidence_artifacts) != _EVIDENCE_PURPOSES:
            failed += ("evidence_artifacts",)
        if failed:
            raise PermissionError(
                "dataset is not authorized for training/evaluation; failed gates: "
                + ", ".join(failed)
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "training_ready": self.training_ready,
            "rights_confirmed": self.rights_confirmed,
            "provenance_complete": self.provenance_complete,
            "readiness_evidence_sha256": self.readiness_evidence_sha256,
            "rights_evidence_sha256": self.rights_evidence_sha256,
            "provenance_evidence_sha256": self.provenance_evidence_sha256,
            "evidence_artifacts": [item.to_payload() for item in self.evidence_artifacts],
        }

    @classmethod
    def from_payload(cls, payload: Any) -> DatasetCapabilities:
        expected = {
            "training_ready",
            "rights_confirmed",
            "provenance_complete",
            "readiness_evidence_sha256",
            "rights_evidence_sha256",
            "provenance_evidence_sha256",
            "evidence_artifacts",
        }
        value = _require_object(payload, "capabilities", expected)
        artifacts = _require_list(value["evidence_artifacts"], "evidence_artifacts")
        return cls(
            training_ready=_require_bool(value["training_ready"], "training_ready"),
            rights_confirmed=_require_bool(value["rights_confirmed"], "rights_confirmed"),
            provenance_complete=_require_bool(value["provenance_complete"], "provenance_complete"),
            readiness_evidence_sha256=_require_string(
                value["readiness_evidence_sha256"], "readiness_evidence_sha256"
            ),
            rights_evidence_sha256=_require_string(
                value["rights_evidence_sha256"], "rights_evidence_sha256"
            ),
            provenance_evidence_sha256=_require_string(
                value["provenance_evidence_sha256"], "provenance_evidence_sha256"
            ),
            evidence_artifacts=tuple(
                DatasetEvidenceArtifact.from_payload(item) for item in artifacts
            ),
        )


@dataclass(frozen=True, slots=True)
class ExampleMetadata:
    """Identity and full feature-through-label dependency interval for one row."""

    example_id: str
    session_id: str
    feature_cutoff_ns: int
    decision_ns: int
    dependency_start_ns: int
    dependency_end_ns: int
    instrument_id: str
    anchor_source: str
    anchor_price: float
    tick_size: float
    log_volume_reference: float
    provenance_sha256: str

    def __post_init__(self) -> None:
        for name in ("example_id", "session_id", "anchor_source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        _validate_instrument_id(self.instrument_id, "instrument_id")
        for name in (
            "feature_cutoff_ns",
            "decision_ns",
            "dependency_start_ns",
            "dependency_end_ns",
        ):
            _require_integer(getattr(self, name), name)
        if not (
            self.dependency_start_ns
            <= self.feature_cutoff_ns
            <= self.decision_ns
            < self.dependency_end_ns
        ):
            raise ValueError(
                "timestamps must satisfy dependency_start <= feature_cutoff <= "
                "decision < dependency_end"
            )
        for name in ("anchor_price", "tick_size"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.log_volume_reference, bool)
            or not isinstance(self.log_volume_reference, (int, float))
            or not math.isfinite(float(self.log_volume_reference))
            or self.log_volume_reference < 0.0
        ):
            raise ValueError("log_volume_reference must be finite and non-negative")
        _validate_sha256(self.provenance_sha256, "provenance_sha256")

    @property
    def chronology_key(self) -> tuple[int, int, str]:
        return self.decision_ns, self.feature_cutoff_ns, self.example_id

    def dependency(self) -> ExampleDependency:
        return ExampleDependency(
            example_id=self.example_id,
            session_id=self.session_id,
            decision_ns=self.decision_ns,
            dependency_start_ns=self.dependency_start_ns,
            dependency_end_ns=self.dependency_end_ns,
        )


@dataclass(frozen=True, slots=True)
class EngineeredExample:
    """One already-engineered row accepted by the verified Arrow writer."""

    metadata: ExampleMetadata
    batch: ChmtQBatch

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ExampleMetadata):
            raise TypeError("metadata must be ExampleMetadata")
        if not isinstance(self.batch, ChmtQBatch):
            raise TypeError("batch must be ChmtQBatch")
        if self.batch.batch_size != 1:
            raise ValueError("an engineered example batch must contain exactly one row")


@dataclass(frozen=True, slots=True)
class DatasetShard:
    """Content and schema identity of one immutable Arrow IPC shard."""

    path: str
    byte_size: int
    row_count: int
    sha256: str
    schema_sha256: str
    first_example_id: str
    last_example_id: str
    first_decision_ns: int
    last_decision_ns: int

    def __post_init__(self) -> None:
        _validate_relative_shard_path(self.path)
        for name in ("byte_size", "row_count"):
            value = _require_integer(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("sha256", "schema_sha256"):
            _validate_sha256(getattr(self, name), name)
        for name in ("first_example_id", "last_example_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        _require_integer(self.first_decision_ns, "first_decision_ns")
        _require_integer(self.last_decision_ns, "last_decision_ns")
        if self.first_decision_ns > self.last_decision_ns:
            raise ValueError("shard decision bounds are reversed")

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "byte_size": self.byte_size,
            "row_count": self.row_count,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "first_example_id": self.first_example_id,
            "last_example_id": self.last_example_id,
            "first_decision_ns": self.first_decision_ns,
            "last_decision_ns": self.last_decision_ns,
        }

    @classmethod
    def from_payload(cls, payload: Any) -> DatasetShard:
        expected = {
            "path",
            "byte_size",
            "row_count",
            "sha256",
            "schema_sha256",
            "first_example_id",
            "last_example_id",
            "first_decision_ns",
            "last_decision_ns",
        }
        value = _require_object(payload, "shard", expected)
        return cls(
            path=_require_string(value["path"], "path"),
            byte_size=_require_integer(value["byte_size"], "byte_size"),
            row_count=_require_integer(value["row_count"], "row_count"),
            sha256=_require_string(value["sha256"], "sha256"),
            schema_sha256=_require_string(value["schema_sha256"], "schema_sha256"),
            first_example_id=_require_string(value["first_example_id"], "first_example_id"),
            last_example_id=_require_string(value["last_example_id"], "last_example_id"),
            first_decision_ns=_require_integer(value["first_decision_ns"], "first_decision_ns"),
            last_decision_ns=_require_integer(value["last_decision_ns"], "last_decision_ns"),
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Canonical semantic manifest whose payload hash is the dataset identity."""

    instrument_id: str
    source_manifest_sha256: str
    builder_sha256: str
    capabilities: DatasetCapabilities
    row_count: int
    shards: tuple[DatasetShard, ...]
    arrow_schema_sha256: str
    feature_schema_sha256: str = FEATURE_SCHEMA_SHA256
    target_schema_sha256: str = TARGET_SCHEMA_SHA256
    tensor_schema_sha256: str = TENSOR_SCHEMA_SHA256
    schema_version: int = DATASET_SCHEMA_VERSION
    format: str = DATASET_FORMAT

    def __post_init__(self) -> None:
        _validate_instrument_id(self.instrument_id, "instrument_id")
        _validate_sha256(self.source_manifest_sha256, "source_manifest_sha256")
        _validate_sha256(self.builder_sha256, "builder_sha256")
        _validate_sha256(self.arrow_schema_sha256, "arrow_schema_sha256")
        if not isinstance(self.capabilities, DatasetCapabilities):
            raise TypeError("capabilities must be DatasetCapabilities")
        if _require_integer(self.row_count, "row_count") <= 0:
            raise ValueError("row_count must be positive")
        if not self.shards or any(not isinstance(item, DatasetShard) for item in self.shards):
            raise ValueError("shards must contain at least one DatasetShard")
        if sum(item.row_count for item in self.shards) != self.row_count:
            raise ValueError("manifest row_count does not equal the shard row total")
        paths = tuple(item.path for item in self.shards)
        if len(set(paths)) != len(paths):
            raise ValueError("manifest contains duplicate shard paths")
        evidence_paths = tuple(
            item.path.casefold() for item in self.capabilities.evidence_artifacts
        )
        if set(item.casefold() for item in paths) & set(evidence_paths):
            raise ValueError("evidence artifact paths must not overlap shard paths")
        if any(item.schema_sha256 != self.arrow_schema_sha256 for item in self.shards):
            raise ValueError("every shard must use the manifest Arrow schema")
        if self.feature_schema_sha256 != FEATURE_SCHEMA_SHA256:
            raise ValueError("feature schema hash does not match fixed CHMT-Q v1")
        if self.target_schema_sha256 != TARGET_SCHEMA_SHA256:
            raise ValueError("target schema hash does not match fixed CHMT-Q v1")
        if self.tensor_schema_sha256 != TENSOR_SCHEMA_SHA256:
            raise ValueError("tensor schema hash does not match fixed CHMT-Q v1")
        if self.schema_version != DATASET_SCHEMA_VERSION or self.format != DATASET_FORMAT:
            raise ValueError("unsupported dataset schema version or format")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "format": self.format,
            "instrument_id": self.instrument_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "builder_sha256": self.builder_sha256,
            "capabilities": self.capabilities.to_payload(),
            "feature_schema_sha256": self.feature_schema_sha256,
            "target_schema_sha256": self.target_schema_sha256,
            "tensor_schema_sha256": self.tensor_schema_sha256,
            "arrow_schema_sha256": self.arrow_schema_sha256,
            "row_count": self.row_count,
            "shards": [item.to_payload() for item in self.shards],
        }

    @property
    def dataset_id(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("ascii")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json({"dataset_id": self.dataset_id, "payload": self.payload()})

    @classmethod
    def from_json(cls, value: str) -> DatasetManifest:
        parsed = _parse_strict_json(value)
        root = _require_object(parsed, "dataset manifest", {"dataset_id", "payload"})
        dataset_id = _require_string(root["dataset_id"], "dataset_id")
        _validate_sha256(dataset_id, "dataset_id")
        payload = _require_object(
            root["payload"],
            "dataset payload",
            {
                "schema_version",
                "format",
                "instrument_id",
                "source_manifest_sha256",
                "builder_sha256",
                "capabilities",
                "feature_schema_sha256",
                "target_schema_sha256",
                "tensor_schema_sha256",
                "arrow_schema_sha256",
                "row_count",
                "shards",
            },
        )
        shards = _require_list(payload["shards"], "shards")
        manifest = cls(
            schema_version=_require_integer(payload["schema_version"], "schema_version"),
            format=_require_string(payload["format"], "format"),
            instrument_id=_require_string(payload["instrument_id"], "instrument_id"),
            source_manifest_sha256=_require_string(
                payload["source_manifest_sha256"], "source_manifest_sha256"
            ),
            builder_sha256=_require_string(payload["builder_sha256"], "builder_sha256"),
            capabilities=DatasetCapabilities.from_payload(payload["capabilities"]),
            feature_schema_sha256=_require_string(
                payload["feature_schema_sha256"], "feature_schema_sha256"
            ),
            target_schema_sha256=_require_string(
                payload["target_schema_sha256"], "target_schema_sha256"
            ),
            tensor_schema_sha256=_require_string(
                payload["tensor_schema_sha256"], "tensor_schema_sha256"
            ),
            arrow_schema_sha256=_require_string(
                payload["arrow_schema_sha256"], "arrow_schema_sha256"
            ),
            row_count=_require_integer(payload["row_count"], "row_count"),
            shards=tuple(DatasetShard.from_payload(item) for item in shards),
        )
        if manifest.dataset_id != dataset_id:
            raise ValueError("dataset_id does not match the canonical manifest payload")
        if value != manifest.to_json():
            raise ValueError("dataset manifest JSON is not in canonical form")
        return manifest


@dataclass(frozen=True, slots=True)
class MaterializedBatch:
    """A model batch accompanied by immutable example and dependency identities."""

    example_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    feature_cutoff_ns: tuple[int, ...]
    decision_ns: tuple[int, ...]
    dependency_start_ns: tuple[int, ...]
    dependency_end_ns: tuple[int, ...]
    instrument_ids: tuple[str, ...]
    anchor_sources: tuple[str, ...]
    anchor_prices: tuple[float, ...]
    tick_sizes: tuple[float, ...]
    log_volume_references: tuple[float, ...]
    provenance_sha256: tuple[str, ...]
    batch: ChmtQBatch

    def __post_init__(self) -> None:
        count = len(self.example_ids)
        values = (
            self.session_ids,
            self.feature_cutoff_ns,
            self.decision_ns,
            self.dependency_start_ns,
            self.dependency_end_ns,
            self.instrument_ids,
            self.anchor_sources,
            self.anchor_prices,
            self.tick_sizes,
            self.log_volume_references,
            self.provenance_sha256,
        )
        if count == 0 or any(len(value) != count for value in values):
            raise ValueError("materialized batch metadata lengths must match and be non-zero")
        if self.batch.batch_size != count:
            raise ValueError("materialized metadata does not match tensor batch size")


@dataclass(frozen=True, slots=True)
class _ExampleLocation:
    metadata: ExampleMetadata
    shard_index: int
    record_batch_index: int
    row_index: int


class ArrowDataset:
    """A fully verified immutable dataset with deterministic batch materialization."""

    def __init__(
        self,
        root: Path,
        manifest: DatasetManifest,
        locations: tuple[_ExampleLocation, ...],
        shard_record_batch_rows: tuple[tuple[int, ...], ...],
        local_research_authorization: LocalResearchAuthorization | None = None,
    ) -> None:
        self.root = root
        self.manifest = manifest
        self.examples = tuple(item.metadata for item in locations)
        self._locations = {item.metadata.example_id: item for item in locations}
        self._shard_record_batch_rows = shard_record_batch_rows
        self.local_research_authorization = local_research_authorization

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        expected_dataset_id: str,
        expected_instrument_id: str | None = None,
        require_training_capabilities: bool = True,
        local_research_authorization: LocalResearchAuthorization | None = None,
    ) -> ArrowDataset:
        """Open only after identity, gates, shards, schema, values, and chronology verify."""

        _validate_sha256(expected_dataset_id, "expected_dataset_id")
        if expected_instrument_id is not None:
            _validate_instrument_id(expected_instrument_id, "expected_instrument_id")
        resolved_root = Path(root).resolve(strict=True)
        if not resolved_root.is_dir():
            raise NotADirectoryError(resolved_root)
        manifest_path = resolved_root / DATASET_MANIFEST_NAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ValueError("dataset.json must be a regular non-symlink file")
        manifest = DatasetManifest.from_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.dataset_id != expected_dataset_id:
            raise ValueError("opened dataset identity differs from expected_dataset_id")
        if expected_instrument_id is not None and manifest.instrument_id != expected_instrument_id:
            raise ValueError("opened dataset instrument_id differs from expected_instrument_id")
        evidence_paths = _verify_evidence_artifacts(resolved_root, manifest.capabilities)
        locations, shard_record_batch_rows = _verify_dataset(resolved_root, manifest)
        _validate_evidence_bindings(
            evidence_paths,
            manifest,
            tuple(item.metadata for item in locations),
        )
        if local_research_authorization is not None:
            from .research_authorization import LocalResearchAuthorization

            if not isinstance(local_research_authorization, LocalResearchAuthorization):
                raise TypeError("local_research_authorization must be a LocalResearchAuthorization")
            local_research_authorization.validate_against(manifest, resolved_root)
        if require_training_capabilities:
            if local_research_authorization is None:
                manifest.capabilities.require_training()
        return cls(
            resolved_root,
            manifest,
            locations,
            shard_record_batch_rows,
            local_research_authorization,
        )

    @property
    def local_research_only(self) -> bool:
        return self.local_research_authorization is not None

    def require_development_access(self) -> None:
        """Permit strict data or an explicit local train/validation authorization."""

        if self.local_research_authorization is None:
            self.manifest.capabilities.require_training()
            return
        self.local_research_authorization.validate_against(self.manifest, self.root)

    @property
    def example_ids(self) -> tuple[str, ...]:
        return tuple(item.example_id for item in self.examples)

    def dependencies(self) -> tuple[ExampleDependency, ...]:
        return tuple(item.dependency() for item in self.examples)

    def iter_batches(
        self,
        example_ids: Iterable[str],
        *,
        batch_size: int,
        partition: Literal["train", "evaluation"],
        seed: int | None = None,
        drop_last: bool = False,
    ) -> Iterator[MaterializedBatch]:
        """Materialize seeded train order or deterministic chronological evaluation order."""

        self.require_development_access()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(drop_last, bool):
            raise TypeError("drop_last must be a boolean")
        selected = tuple(example_ids)
        if not selected:
            raise ValueError("at least one example_id is required")
        if any(not isinstance(item, str) or not item for item in selected):
            raise ValueError("example_ids must be non-empty strings")
        if len(set(selected)) != len(selected):
            raise ValueError("batch selection contains duplicate example_ids")
        unknown = tuple(item for item in selected if item not in self._locations)
        if unknown:
            raise KeyError(f"unknown dataset example_ids: {unknown}")

        if partition == "train":
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ValueError("train materialization requires a non-negative integer seed")
        elif partition == "evaluation":
            if seed is not None:
                raise ValueError("evaluation order is chronological and does not accept a seed")
        else:
            raise ValueError("partition must be train or evaluation")

        locations = tuple(self._locations[item] for item in selected)
        groups = _ordered_location_groups(
            locations,
            partition=partition,
            seed=seed,
            dataset_id=self.manifest.dataset_id,
            shards=self.manifest.shards,
        )
        row_limit = len(selected)
        if drop_last:
            row_limit -= row_limit % batch_size

        pending_locations: list[_ExampleLocation] = []
        pending_batches: list[ChmtQBatch] = []
        pending_rows = 0
        for group_locations, group_batch in self._iter_selected_batches(
            groups, row_limit=row_limit
        ):
            group_offset = 0
            while group_offset < len(group_locations):
                take = min(batch_size - pending_rows, len(group_locations) - group_offset)
                pending_locations.extend(group_locations[group_offset : group_offset + take])
                pending_batches.append(_slice_batch_rows(group_batch, group_offset, take))
                pending_rows += take
                group_offset += take
                if pending_rows != batch_size:
                    continue
                materialized = _materialized_tensor_batch(
                    pending_locations,
                    _concatenate_batches(pending_batches),
                )
                pending_locations.clear()
                pending_batches.clear()
                pending_rows = 0
                yield materialized
                del materialized
            del group_batch
        if pending_rows:
            yield _materialized_tensor_batch(
                pending_locations,
                _concatenate_batches(pending_batches),
            )

    def _iter_selected_batches(
        self,
        groups: tuple[tuple[int, tuple[tuple[int, tuple[_ExampleLocation, ...]], ...]], ...],
        *,
        row_limit: int,
    ) -> Iterator[tuple[tuple[_ExampleLocation, ...], ChmtQBatch]]:
        pa = _require_pyarrow()
        emitted = 0
        for shard_index, record_batch_groups in groups:
            if emitted >= row_limit:
                return
            shard = self.manifest.shards[shard_index]
            expected_rows = self._shard_record_batch_rows[shard_index]
            with _open_verified_reader(
                self.root,
                shard,
                self.manifest,
                expected_record_batch_rows=expected_rows,
            ) as reader:
                for record_batch_index, locations in record_batch_groups:
                    if emitted >= row_limit:
                        return
                    record_batch = reader.get_batch(record_batch_index)
                    if record_batch.num_rows != expected_rows[record_batch_index]:
                        raise ValueError(f"shard {shard.path} record-batch layout changed")
                    selected_count = min(len(locations), row_limit - emitted)
                    selected_locations = locations[:selected_count]
                    selected_indices = pa.array(
                        [location.row_index for location in selected_locations],
                        type=pa.int64(),
                    )
                    selected_record_batch = record_batch.take(selected_indices)
                    selected_batch = _batch_from_record_batch(selected_record_batch)
                    emitted += selected_count
                    del selected_indices
                    del selected_record_batch
                    del record_batch
                    yield selected_locations, selected_batch
                    del selected_batch


def chmt_q_arrow_schema() -> Any:
    """Return the exact fixed Arrow schema, importing PyArrow only on demand."""

    pa = _require_pyarrow()
    fields = [
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("session_id", pa.string(), nullable=False),
        pa.field("feature_cutoff_ns", pa.int64(), nullable=False),
        pa.field("decision_ns", pa.int64(), nullable=False),
        pa.field("dependency_start_ns", pa.int64(), nullable=False),
        pa.field("dependency_end_ns", pa.int64(), nullable=False),
        pa.field("instrument_id", pa.string(), nullable=False),
        pa.field("anchor_source", pa.string(), nullable=False),
        pa.field("anchor_price", pa.float64(), nullable=False),
        pa.field("tick_size", pa.float64(), nullable=False),
        pa.field("log_volume_reference", pa.float64(), nullable=False),
        pa.field("provenance_sha256", pa.string(), nullable=False),
        pa.field("bars", pa.list_(pa.float32(), 64 * 16), nullable=False),
        pa.field("bars_mask", pa.list_(pa.bool_(), 64), nullable=False),
        pa.field("l1", pa.list_(pa.float32(), 240 * 16), nullable=False),
        pa.field("l1_mask", pa.list_(pa.bool_(), 240), nullable=False),
        pa.field("l2", pa.list_(pa.float32(), 240 * 2 * 10 * 12), nullable=False),
        pa.field("l2_level_mask", pa.list_(pa.bool_(), 240 * 2 * 10), nullable=False),
        pa.field("l2_snapshot_mask", pa.list_(pa.bool_(), 240), nullable=False),
        pa.field("quality", pa.list_(pa.float32(), 16), nullable=False),
        pa.field("targets", pa.list_(pa.float32(), 5 * 5), nullable=False),
        pa.field("target_mask", pa.list_(pa.bool_(), 5 * 5), nullable=False),
        pa.field("presence_labels", pa.list_(pa.float32(), 5 * 2), nullable=False),
        pa.field("presence_mask", pa.list_(pa.bool_(), 5 * 2), nullable=False),
        pa.field("feature_schema_sha256", pa.string(), nullable=False),
        pa.field("target_schema_sha256", pa.string(), nullable=False),
    ]
    metadata = {
        b"chmt_q.contract": b"chmt-q-v1",
        b"chmt_q.feature_schema_sha256": FEATURE_SCHEMA_SHA256.encode("ascii"),
        b"chmt_q.target_schema_sha256": TARGET_SCHEMA_SHA256.encode("ascii"),
        b"chmt_q.tensor_schema_sha256": TENSOR_SCHEMA_SHA256.encode("ascii"),
        b"chmt_q.timestamp_unit": b"utc-epoch-nanoseconds",
    }
    return pa.schema(fields, metadata=metadata)


def arrow_schema_sha256() -> str:
    """Return the byte-level identity of the exact Arrow schema."""

    return _schema_sha256(chmt_q_arrow_schema())


def write_engineered_fixture_dataset(
    root: str | Path,
    examples: Iterable[EngineeredExample],
    *,
    source_manifest_sha256: str,
    builder_sha256: str,
    capabilities: DatasetCapabilities,
    rows_per_shard: int = 64,
    rows_per_record_batch: int = MAX_RECORD_BATCH_ROWS,
) -> DatasetManifest:
    """Write already-engineered generated fixtures; never derive features from raw ticks."""

    materialized = tuple(examples)
    if not materialized:
        raise ValueError("at least one engineered fixture example is required")
    return _write_engineered_dataset(
        root,
        materialized,
        source_manifest_sha256=source_manifest_sha256,
        builder_sha256=builder_sha256,
        capabilities=capabilities,
        rows_per_shard=rows_per_shard,
        rows_per_record_batch=rows_per_record_batch,
    )


def write_engineered_dataset(
    root: str | Path,
    examples: Iterable[EngineeredExample],
    *,
    source_manifest_sha256: str,
    builder_sha256: str,
    capabilities: DatasetCapabilities,
    rows_per_shard: int = 64,
    rows_per_record_batch: int = MAX_RECORD_BATCH_ROWS,
    compression: Literal["lz4", "zstd"] | None = None,
) -> DatasetManifest:
    """Stream already-engineered production rows into verified Arrow IPC shards."""

    if not isinstance(capabilities, DatasetCapabilities):
        raise TypeError("capabilities must be DatasetCapabilities")
    if (
        capabilities.training_ready
        or capabilities.rights_confirmed
        or capabilities.provenance_complete
    ):
        raise ValueError("the production writer cannot self-authorize training capability gates")
    return _write_engineered_dataset(
        root,
        examples,
        source_manifest_sha256=source_manifest_sha256,
        builder_sha256=builder_sha256,
        capabilities=capabilities,
        rows_per_shard=rows_per_shard,
        rows_per_record_batch=rows_per_record_batch,
        compression=compression,
    )


def _write_engineered_dataset(
    root: str | Path,
    examples: Iterable[EngineeredExample],
    *,
    source_manifest_sha256: str,
    builder_sha256: str,
    capabilities: DatasetCapabilities,
    rows_per_shard: int,
    rows_per_record_batch: int,
    compression: Literal["lz4", "zstd"] | None = None,
) -> DatasetManifest:
    """Shared implementation; fixture callers alone may assert positive test gates."""

    _validate_sha256(source_manifest_sha256, "source_manifest_sha256")
    _validate_sha256(builder_sha256, "builder_sha256")
    if not isinstance(capabilities, DatasetCapabilities):
        raise TypeError("capabilities must be DatasetCapabilities")
    if (
        isinstance(rows_per_shard, bool)
        or not isinstance(rows_per_shard, int)
        or rows_per_shard <= 0
    ):
        raise ValueError("rows_per_shard must be a positive integer")
    if (
        isinstance(rows_per_record_batch, bool)
        or not isinstance(rows_per_record_batch, int)
        or not 1 <= rows_per_record_batch <= MAX_RECORD_BATCH_ROWS
    ):
        raise ValueError(
            "rows_per_record_batch must be a positive integer no greater than "
            f"{MAX_RECORD_BATCH_ROWS}"
        )
    if compression not in {None, "lz4", "zstd"}:
        raise ValueError("compression must be None, lz4, or zstd")
    pa = _require_pyarrow()
    if compression is not None and not pa.Codec.is_available(compression):
        raise RuntimeError(f"PyArrow codec is unavailable: {compression}")
    schema = chmt_q_arrow_schema()
    schema_hash = _schema_sha256(schema)

    destination = Path(root)
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise FileExistsError("dataset destination must not exist or must be empty")
    else:
        destination.mkdir(parents=True)
    shard_dir = destination / "shards"
    shard_dir.mkdir()

    shards: list[DatasetShard] = []
    metadata: list[ExampleMetadata] = []
    buffered: list[EngineeredExample] = []
    instrument_id: str | None = None
    for item in examples:
        if not isinstance(item, EngineeredExample):
            raise TypeError("examples must be EngineeredExample instances")
        if instrument_id is None:
            instrument_id = item.metadata.instrument_id
        elif item.metadata.instrument_id != instrument_id:
            raise ValueError("dataset row instrument_id differs from the manifest instrument_id")
        _validate_engineered_batch(item.batch)
        metadata.append(item.metadata)
        buffered.append(item)
        if len(buffered) == rows_per_shard:
            _validate_metadata_sequence(metadata)
            shards.append(
                _write_engineered_shard(
                    destination,
                    buffered,
                    shard_index=len(shards),
                    rows_per_record_batch=rows_per_record_batch,
                    schema=schema,
                    schema_hash=schema_hash,
                    pa=pa,
                    compression=compression,
                )
            )
            buffered = []
    if not metadata or instrument_id is None:
        raise ValueError("at least one engineered example is required")
    _validate_metadata_sequence(metadata)
    if buffered:
        shards.append(
            _write_engineered_shard(
                destination,
                buffered,
                shard_index=len(shards),
                rows_per_record_batch=rows_per_record_batch,
                schema=schema,
                schema_hash=schema_hash,
                pa=pa,
                compression=compression,
            )
        )
    capabilities = _write_dataset_evidence(
        destination,
        capabilities,
        instrument_id=instrument_id,
        source_manifest_sha256=source_manifest_sha256,
        builder_sha256=builder_sha256,
        arrow_schema_sha256=schema_hash,
        metadata=metadata,
    )

    manifest = DatasetManifest(
        instrument_id=instrument_id,
        source_manifest_sha256=source_manifest_sha256,
        builder_sha256=builder_sha256,
        capabilities=capabilities,
        row_count=len(metadata),
        shards=tuple(shards),
        arrow_schema_sha256=schema_hash,
    )
    manifest_path = destination / DATASET_MANIFEST_NAME
    manifest_temporary = manifest_path.with_name(f"{manifest_path.name}.tmp")
    manifest_temporary.write_text(manifest.to_json(), encoding="utf-8")
    manifest_temporary.replace(manifest_path)
    return manifest


def _write_engineered_shard(
    destination: Path,
    selected: Sequence[EngineeredExample],
    *,
    shard_index: int,
    rows_per_record_batch: int,
    schema: Any,
    schema_hash: str,
    pa: Any,
    compression: Literal["lz4", "zstd"] | None,
) -> DatasetShard:
    relative = f"shards/part-{shard_index:05d}.arrow"
    shard_path = destination / PurePosixPath(relative)
    options = pa.ipc.IpcWriteOptions(compression=compression) if compression else None
    with pa.OSFile(str(shard_path), "wb") as sink:
        with pa.ipc.new_file(sink, schema, options=options) as writer:
            for batch_offset in range(0, len(selected), rows_per_record_batch):
                record_batch_examples = selected[
                    batch_offset : batch_offset + rows_per_record_batch
                ]
                table = _table_from_engineered(record_batch_examples, schema)
                writer.write_table(table, max_chunksize=rows_per_record_batch)
                del table
    first = selected[0].metadata
    last = selected[-1].metadata
    return DatasetShard(
        path=relative,
        byte_size=shard_path.stat().st_size,
        row_count=len(selected),
        sha256=_file_sha256(shard_path),
        schema_sha256=schema_hash,
        first_example_id=first.example_id,
        last_example_id=last.example_id,
        first_decision_ns=first.decision_ns,
        last_decision_ns=last.decision_ns,
    )


def _write_dataset_evidence(
    destination: Path,
    capabilities: DatasetCapabilities,
    *,
    instrument_id: str,
    source_manifest_sha256: str,
    builder_sha256: str,
    arrow_schema_sha256: str,
    metadata: Sequence[ExampleMetadata],
) -> DatasetCapabilities:
    if capabilities.evidence_artifacts:
        raise ValueError("the writer generates its own bound evidence_artifacts")
    evidence_dir = destination / "evidence"
    evidence_dir.mkdir()
    binding = _dataset_evidence_binding(
        instrument_id=instrument_id,
        source_manifest_sha256=source_manifest_sha256,
        builder_sha256=builder_sha256,
        arrow_schema_sha256=arrow_schema_sha256,
        row_count=len(metadata),
    )

    provenance_relative = "evidence/provenance.jsonl"
    provenance_path = destination.joinpath(*PurePosixPath(provenance_relative).parts)
    provenance_header = {
        "artifact_type": "chmt-q-ordered-provenance-binding",
        "binding": binding,
        "gate": "provenance_complete",
        "gate_satisfied": capabilities.provenance_complete,
        "schema_version": 1,
    }
    with provenance_path.open("wb") as stream:
        stream.write(_canonical_evidence_line(provenance_header))
        for item in metadata:
            stream.write(
                _canonical_evidence_line(
                    {
                        "example_id": item.example_id,
                        "provenance_sha256": item.provenance_sha256,
                    }
                )
            )
    provenance_artifact = _evidence_artifact("provenance", provenance_relative, provenance_path)

    rights_relative = "evidence/rights.json"
    rights_path = destination.joinpath(*PurePosixPath(rights_relative).parts)
    rights_path.write_bytes(
        _canonical_evidence_line(
            {
                "artifact_type": "chmt-q-rights-binding",
                "gate": "rights_confirmed",
                "gate_satisfied": capabilities.rights_confirmed,
                "instrument_id": instrument_id,
                "schema_version": 1,
                "source_manifest_sha256": source_manifest_sha256,
            }
        )
    )
    rights_artifact = _evidence_artifact("rights", rights_relative, rights_path)

    readiness_relative = "evidence/readiness.json"
    readiness_path = destination.joinpath(*PurePosixPath(readiness_relative).parts)
    readiness_path.write_bytes(
        _canonical_evidence_line(
            {
                "artifact_type": "chmt-q-readiness-binding",
                "binding": binding,
                "gate": "training_ready",
                "gate_satisfied": capabilities.training_ready,
                "provenance_inventory": provenance_artifact.to_payload(),
                "schema_version": 1,
            }
        )
    )
    readiness_artifact = _evidence_artifact("readiness", readiness_relative, readiness_path)
    artifacts = (readiness_artifact, rights_artifact, provenance_artifact)
    return DatasetCapabilities(
        training_ready=capabilities.training_ready,
        rights_confirmed=capabilities.rights_confirmed,
        provenance_complete=capabilities.provenance_complete,
        readiness_evidence_sha256=artifacts[0].sha256,
        rights_evidence_sha256=artifacts[1].sha256,
        provenance_evidence_sha256=artifacts[2].sha256,
        evidence_artifacts=artifacts,
    )


def _verify_evidence_artifacts(root: Path, capabilities: DatasetCapabilities) -> dict[str, Path]:
    artifacts = capabilities.evidence_artifacts
    if tuple(item.purpose for item in artifacts) != _EVIDENCE_PURPOSES:
        raise ValueError("dataset capability evidence inventory is incomplete or out of order")
    verified: dict[str, Path] = {}
    for artifact in artifacts:
        path = _resolve_evidence_path(root, artifact.path)
        if path.stat(follow_symlinks=False).st_size != artifact.byte_size:
            raise ValueError(
                f"evidence artifact {artifact.path} byte size does not match its manifest"
            )
        if _file_sha256(path) != artifact.sha256:
            raise ValueError(
                f"evidence artifact {artifact.path} SHA-256 does not match its manifest"
            )
        verified[artifact.purpose] = path
    return verified


def _validate_evidence_bindings(
    evidence_paths: dict[str, Path],
    manifest: DatasetManifest,
    metadata: Sequence[ExampleMetadata],
) -> None:
    artifacts = {item.purpose: item for item in manifest.capabilities.evidence_artifacts}
    binding = _dataset_evidence_binding(
        instrument_id=manifest.instrument_id,
        source_manifest_sha256=manifest.source_manifest_sha256,
        builder_sha256=manifest.builder_sha256,
        arrow_schema_sha256=manifest.arrow_schema_sha256,
        row_count=manifest.row_count,
    )
    readiness_expected = _canonical_evidence_line(
        {
            "artifact_type": "chmt-q-readiness-binding",
            "binding": binding,
            "gate": "training_ready",
            "gate_satisfied": manifest.capabilities.training_ready,
            "provenance_inventory": artifacts["provenance"].to_payload(),
            "schema_version": 1,
        }
    )
    if evidence_paths["readiness"].read_bytes() != readiness_expected:
        raise ValueError("readiness evidence does not canonically bind the dataset manifest")

    rights_expected = _canonical_evidence_line(
        {
            "artifact_type": "chmt-q-rights-binding",
            "gate": "rights_confirmed",
            "gate_satisfied": manifest.capabilities.rights_confirmed,
            "instrument_id": manifest.instrument_id,
            "schema_version": 1,
            "source_manifest_sha256": manifest.source_manifest_sha256,
        }
    )
    if evidence_paths["rights"].read_bytes() != rights_expected:
        raise ValueError("rights evidence does not canonically bind its declared gate")

    provenance_header = _canonical_evidence_line(
        {
            "artifact_type": "chmt-q-ordered-provenance-binding",
            "binding": binding,
            "gate": "provenance_complete",
            "gate_satisfied": manifest.capabilities.provenance_complete,
            "schema_version": 1,
        }
    )
    with evidence_paths["provenance"].open("rb") as stream:
        if stream.readline() != provenance_header:
            raise ValueError("provenance evidence does not canonically bind the dataset manifest")
        for index, item in enumerate(metadata):
            expected = _canonical_evidence_line(
                {
                    "example_id": item.example_id,
                    "provenance_sha256": item.provenance_sha256,
                }
            )
            if stream.readline() != expected:
                raise ValueError(
                    "ordered provenance inventory differs from verified dataset row "
                    f"{index} ({item.example_id})"
                )
        if stream.read(1):
            raise ValueError(
                "ordered provenance inventory contains rows not present in the dataset"
            )


def _dataset_evidence_binding(
    *,
    instrument_id: str,
    source_manifest_sha256: str,
    builder_sha256: str,
    arrow_schema_sha256: str,
    row_count: int,
) -> dict[str, Any]:
    return {
        "arrow_schema_sha256": arrow_schema_sha256,
        "builder_sha256": builder_sha256,
        "dataset_format": DATASET_FORMAT,
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "instrument_id": instrument_id,
        "row_count": row_count,
        "source_manifest_sha256": source_manifest_sha256,
        "target_schema_sha256": TARGET_SCHEMA_SHA256,
        "tensor_schema_sha256": TENSOR_SCHEMA_SHA256,
    }


def _canonical_evidence_line(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("ascii")


def _evidence_artifact(
    purpose: Literal["readiness", "rights", "provenance"],
    relative: str,
    path: Path,
) -> DatasetEvidenceArtifact:
    return DatasetEvidenceArtifact(
        purpose=purpose,
        path=relative,
        byte_size=path.stat().st_size,
        sha256=_file_sha256(path),
    )


def _verify_dataset(
    root: Path, manifest: DatasetManifest
) -> tuple[tuple[_ExampleLocation, ...], tuple[tuple[int, ...], ...]]:
    if arrow_schema_sha256() != manifest.arrow_schema_sha256:
        raise ValueError("manifest Arrow schema hash differs from the fixed runtime contract")
    locations: list[_ExampleLocation] = []
    shard_record_batch_rows: list[tuple[int, ...]] = []
    for shard_index, shard in enumerate(manifest.shards):
        shard_metadata: list[ExampleMetadata] = []
        record_batch_rows: list[int] = []
        with _open_verified_reader(root, shard, manifest) as reader:
            for record_batch_index in range(reader.num_record_batches):
                record_batch = reader.get_batch(record_batch_index)
                if not 1 <= record_batch.num_rows <= MAX_RECORD_BATCH_ROWS:
                    raise ValueError(
                        f"shard {shard.path} record batches must contain between 1 and "
                        f"{MAX_RECORD_BATCH_ROWS} rows"
                    )
                batch_metadata = _metadata_from_table(record_batch)
                _validate_schema_hash_columns(record_batch, manifest)
                _validate_single_instrument(batch_metadata, manifest.instrument_id)
                engineered_batch = _batch_from_record_batch(record_batch)
                del engineered_batch
                locations.extend(
                    _ExampleLocation(
                        item,
                        shard_index,
                        record_batch_index,
                        row_index,
                    )
                    for row_index, item in enumerate(batch_metadata)
                )
                shard_metadata.extend(batch_metadata)
                record_batch_rows.append(record_batch.num_rows)
                del record_batch
        if sum(record_batch_rows) != shard.row_count:
            raise ValueError(f"shard {shard.path} row count does not match its manifest")
        if not shard_metadata:
            raise ValueError(f"shard {shard.path} is empty")
        first = shard_metadata[0]
        last = shard_metadata[-1]
        observed_bounds = (
            first.example_id,
            last.example_id,
            first.decision_ns,
            last.decision_ns,
        )
        expected_bounds = (
            shard.first_example_id,
            shard.last_example_id,
            shard.first_decision_ns,
            shard.last_decision_ns,
        )
        if observed_bounds != expected_bounds:
            raise ValueError(f"shard {shard.path} identity/decision bounds do not match")
        shard_record_batch_rows.append(tuple(record_batch_rows))
    if len(locations) != manifest.row_count:
        raise ValueError("verified row count differs from manifest row_count")
    _validate_metadata_sequence(tuple(item.metadata for item in locations))
    return tuple(locations), tuple(shard_record_batch_rows)


@contextmanager
def _open_verified_reader(
    root: Path,
    shard: DatasetShard,
    manifest: DatasetManifest,
    *,
    expected_record_batch_rows: tuple[int, ...] | None = None,
) -> Iterator[Any]:
    pa = _require_pyarrow()
    path = _resolve_shard_path(root, shard.path)
    if path.stat().st_size != shard.byte_size:
        raise ValueError(f"shard {shard.path} byte size does not match its manifest")
    if _file_sha256(path) != shard.sha256:
        raise ValueError(f"shard {shard.path} SHA-256 does not match its manifest")
    if manifest.arrow_schema_sha256 != shard.schema_sha256:
        raise ValueError(f"shard {shard.path} schema is not the manifest schema")
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_file(source)
        schema = reader.schema
        if _schema_sha256(schema) != shard.schema_sha256:
            raise ValueError(f"shard {shard.path} schema SHA-256 does not match")
        if not schema.equals(chmt_q_arrow_schema(), check_metadata=True):
            raise ValueError(f"shard {shard.path} does not use the exact CHMT-Q Arrow schema")
        if expected_record_batch_rows is not None and reader.num_record_batches != len(
            expected_record_batch_rows
        ):
            raise ValueError(f"shard {shard.path} record-batch layout changed")
        yield reader


def _metadata_from_table(table: Any) -> tuple[ExampleMetadata, ...]:
    names = (
        "example_id",
        "session_id",
        "feature_cutoff_ns",
        "decision_ns",
        "dependency_start_ns",
        "dependency_end_ns",
        "instrument_id",
        "anchor_source",
        "anchor_price",
        "tick_size",
        "log_volume_reference",
        "provenance_sha256",
    )
    columns = {name: _column_by_name(table, name).to_pylist() for name in names}
    return tuple(
        ExampleMetadata(**{name: columns[name][index] for name in names})
        for index in range(table.num_rows)
    )


def _validate_schema_hash_columns(table: Any, manifest: DatasetManifest) -> None:
    features = _column_by_name(table, "feature_schema_sha256").to_pylist()
    targets = _column_by_name(table, "target_schema_sha256").to_pylist()
    if any(value != manifest.feature_schema_sha256 for value in features):
        raise ValueError("a row feature schema hash differs from the manifest")
    if any(value != manifest.target_schema_sha256 for value in targets):
        raise ValueError("a row target schema hash differs from the manifest")


def _validate_metadata_sequence(examples: Sequence[ExampleMetadata]) -> None:
    if not examples:
        raise ValueError("dataset must contain at least one example")
    ids = tuple(item.example_id for item in examples)
    if len(set(ids)) != len(ids):
        raise ValueError("dataset contains duplicate example_ids")
    keys = tuple(item.chronology_key for item in examples)
    if keys != tuple(sorted(keys)):
        raise ValueError("dataset rows must be in deterministic chronological order")

    closed_sessions: set[str] = set()
    current = examples[0].session_id
    session_last_decision = examples[0].decision_ns
    for item in examples[1:]:
        if item.session_id == current:
            session_last_decision = item.decision_ns
            continue
        closed_sessions.add(current)
        if item.session_id in closed_sessions:
            raise ValueError("a session appears in multiple non-contiguous row ranges")
        if session_last_decision >= item.decision_ns:
            raise ValueError("session decision ranges interleave or share a boundary")
        current = item.session_id
        session_last_decision = item.decision_ns


def _validate_single_instrument(
    examples: Sequence[ExampleMetadata], expected_instrument_id: str
) -> None:
    _validate_instrument_id(expected_instrument_id, "instrument_id")
    mismatched = tuple(
        item.example_id for item in examples if item.instrument_id != expected_instrument_id
    )
    if mismatched:
        raise ValueError(
            "dataset rows must all match the manifest instrument_id; mismatched examples: "
            + ", ".join(mismatched[:5])
        )


def _validate_engineered_batch(batch: ChmtQBatch) -> None:
    batch.validate(
        require_targets=True,
        require_presence=True,
        require_supervision=True,
        require_feature_domains=True,
    )
    assert batch.target_mask is not None
    assert batch.presence_mask is not None
    supervised = batch.target_mask.flatten(1).any(1) | batch.presence_mask.flatten(1).any(1)
    if not bool(supervised.all().item()):
        raise ValueError("every dataset row must contain at least one supervised label")


def _ordered_location_groups(
    locations: Sequence[_ExampleLocation],
    *,
    partition: Literal["train", "evaluation"],
    seed: int | None,
    dataset_id: str,
    shards: Sequence[DatasetShard],
) -> tuple[tuple[int, tuple[tuple[int, tuple[_ExampleLocation, ...]], ...]], ...]:
    grouped: dict[int, dict[int, list[_ExampleLocation]]] = {}
    for location in locations:
        grouped.setdefault(location.shard_index, {}).setdefault(
            location.record_batch_index, []
        ).append(location)

    if partition == "evaluation":
        shard_order = sorted(grouped)
    else:
        assert seed is not None
        shard_order = sorted(
            grouped,
            key=lambda shard_index: _seeded_stream_key(
                seed,
                dataset_id,
                "shard",
                shards[shard_index].path,
            ),
        )

    ordered: list[tuple[int, tuple[tuple[int, tuple[_ExampleLocation, ...]], ...]]] = []
    for shard_index in shard_order:
        record_batches = grouped[shard_index]
        shard_path = shards[shard_index].path
        if partition == "evaluation":
            record_batch_order = sorted(record_batches)
        else:
            assert seed is not None
            record_batch_order = sorted(
                record_batches,
                key=lambda record_batch_index: _seeded_stream_key(
                    seed,
                    dataset_id,
                    "record-batch",
                    shard_path,
                    str(record_batch_index),
                ),
            )
        ordered_record_batches: list[tuple[int, tuple[_ExampleLocation, ...]]] = []
        for record_batch_index in record_batch_order:
            batch_locations = record_batches[record_batch_index]
            if partition == "evaluation":
                batch_locations.sort(key=lambda item: item.row_index)
            else:
                assert seed is not None
                batch_locations.sort(
                    key=lambda item: _seeded_stream_key(
                        seed,
                        dataset_id,
                        "row",
                        shard_path,
                        str(record_batch_index),
                        item.metadata.example_id,
                    )
                )
            ordered_record_batches.append((record_batch_index, tuple(batch_locations)))
        ordered.append((shard_index, tuple(ordered_record_batches)))
    return tuple(ordered)


def _materialized_batch(
    locations: Sequence[_ExampleLocation], rows: Sequence[dict[str, Any]]
) -> MaterializedBatch:
    if len(locations) != len(rows) or not rows:
        raise RuntimeError("internal error: streamed batch metadata and rows differ")
    return _materialized_tensor_batch(locations, _batch_from_rows(rows))


def _materialized_tensor_batch(
    locations: Sequence[_ExampleLocation], batch: ChmtQBatch
) -> MaterializedBatch:
    if len(locations) != batch.batch_size or not locations:
        raise RuntimeError("internal error: streamed batch metadata and tensors differ")
    metadata = tuple(item.metadata for item in locations)
    return MaterializedBatch(
        example_ids=tuple(item.example_id for item in metadata),
        session_ids=tuple(item.session_id for item in metadata),
        feature_cutoff_ns=tuple(item.feature_cutoff_ns for item in metadata),
        decision_ns=tuple(item.decision_ns for item in metadata),
        dependency_start_ns=tuple(item.dependency_start_ns for item in metadata),
        dependency_end_ns=tuple(item.dependency_end_ns for item in metadata),
        instrument_ids=tuple(item.instrument_id for item in metadata),
        anchor_sources=tuple(item.anchor_source for item in metadata),
        anchor_prices=tuple(item.anchor_price for item in metadata),
        tick_sizes=tuple(item.tick_size for item in metadata),
        log_volume_references=tuple(item.log_volume_reference for item in metadata),
        provenance_sha256=tuple(item.provenance_sha256 for item in metadata),
        batch=batch,
    )


_DATASET_BATCH_TENSOR_NAMES = (
    "bars",
    "bars_mask",
    "l1",
    "l1_mask",
    "l2",
    "l2_level_mask",
    "l2_snapshot_mask",
    "quality",
    "targets",
    "target_mask",
    "presence_labels",
    "presence_mask",
)


def _required_batch_tensor(batch: ChmtQBatch, name: str) -> torch.Tensor:
    value = getattr(batch, name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"verified dataset batch is missing tensor {name}")
    return value


def _slice_batch_rows(batch: ChmtQBatch, start: int, count: int) -> ChmtQBatch:
    if start < 0 or count <= 0 or start + count > batch.batch_size:
        raise RuntimeError("internal error: invalid dataset batch slice")
    return ChmtQBatch(
        **{
            name: _required_batch_tensor(batch, name).narrow(0, start, count)
            for name in _DATASET_BATCH_TENSOR_NAMES
        }
    )


def _concatenate_batches(batches: Sequence[ChmtQBatch]) -> ChmtQBatch:
    if not batches:
        raise RuntimeError("internal error: cannot concatenate zero dataset batches")
    if len(batches) == 1:
        return batches[0]
    return ChmtQBatch(
        **{
            name: torch.cat(
                tuple(_required_batch_tensor(batch, name) for batch in batches),
                dim=0,
            )
            for name in _DATASET_BATCH_TENSOR_NAMES
        }
    )


def _batch_from_rows(rows: Sequence[dict[str, Any]]) -> ChmtQBatch:
    if not rows:
        raise ValueError("cannot materialize an empty batch")

    def tensor(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.tensor([row[name] for row in rows], dtype=dtype).reshape(len(rows), *shape)

    batch = ChmtQBatch(
        bars=tensor("bars", torch.float32, (64, 16)),
        bars_mask=tensor("bars_mask", torch.bool, (64,)),
        l1=tensor("l1", torch.float32, (240, 16)),
        l1_mask=tensor("l1_mask", torch.bool, (240,)),
        l2=tensor("l2", torch.float32, (240, 2, 10, 12)),
        l2_level_mask=tensor("l2_level_mask", torch.bool, (240, 2, 10)),
        l2_snapshot_mask=tensor("l2_snapshot_mask", torch.bool, (240,)),
        quality=tensor("quality", torch.float32, (16,)),
        targets=tensor("targets", torch.float32, (5, 5)),
        target_mask=tensor("target_mask", torch.bool, (5, 5)),
        presence_labels=tensor("presence_labels", torch.float32, (5, 2)),
        presence_mask=tensor("presence_mask", torch.bool, (5, 2)),
    )
    _validate_engineered_batch(batch)
    return batch


def _batch_from_record_batch(record_batch: Any) -> ChmtQBatch:
    """Materialize one verified Arrow record batch without Python row expansion."""

    if record_batch.num_rows <= 0:
        raise ValueError("cannot materialize an empty record batch")

    def tensor(name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> torch.Tensor:
        column = _column_by_name(record_batch, name)
        if column.null_count:
            raise ValueError(f"{name} contains null rows")
        flattened = column.flatten()
        if flattened.null_count:
            raise ValueError(f"{name} contains null values")
        expected_values = record_batch.num_rows * math.prod(shape)
        if len(flattened) != expected_values:
            raise ValueError(f"{name} must contain exactly {expected_values} primitive values")
        result = torch.from_numpy(flattened.to_numpy(zero_copy_only=False).copy())
        if result.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}")
        return result.reshape(record_batch.num_rows, *shape)

    batch = ChmtQBatch(
        bars=tensor("bars", torch.float32, (64, 16)),
        bars_mask=tensor("bars_mask", torch.bool, (64,)),
        l1=tensor("l1", torch.float32, (240, 16)),
        l1_mask=tensor("l1_mask", torch.bool, (240,)),
        l2=tensor("l2", torch.float32, (240, 2, 10, 12)),
        l2_level_mask=tensor("l2_level_mask", torch.bool, (240, 2, 10)),
        l2_snapshot_mask=tensor("l2_snapshot_mask", torch.bool, (240,)),
        quality=tensor("quality", torch.float32, (16,)),
        targets=tensor("targets", torch.float32, (5, 5)),
        target_mask=tensor("target_mask", torch.bool, (5, 5)),
        presence_labels=tensor("presence_labels", torch.float32, (5, 2)),
        presence_mask=tensor("presence_mask", torch.bool, (5, 2)),
    )
    _validate_engineered_batch(batch)
    return batch


def _tensor_row(table: Any, row_index: int) -> dict[str, Any]:
    return {
        name: _column_by_name(table, name)[row_index].as_py()
        for name in (
            "bars",
            "bars_mask",
            "l1",
            "l1_mask",
            "l2",
            "l2_level_mask",
            "l2_snapshot_mask",
            "quality",
            "targets",
            "target_mask",
            "presence_labels",
            "presence_mask",
        )
    }


def _table_from_engineered(examples: Sequence[EngineeredExample], schema: Any) -> Any:
    pa = _require_pyarrow()
    values: dict[str, list[Any]] = {field.name: [] for field in schema}
    tensor_names = (
        "bars",
        "bars_mask",
        "l1",
        "l1_mask",
        "l2",
        "l2_level_mask",
        "l2_snapshot_mask",
        "quality",
        "targets",
        "target_mask",
        "presence_labels",
        "presence_mask",
    )
    for item in examples:
        metadata = item.metadata
        values["example_id"].append(metadata.example_id)
        values["session_id"].append(metadata.session_id)
        values["feature_cutoff_ns"].append(metadata.feature_cutoff_ns)
        values["decision_ns"].append(metadata.decision_ns)
        values["dependency_start_ns"].append(metadata.dependency_start_ns)
        values["dependency_end_ns"].append(metadata.dependency_end_ns)
        values["instrument_id"].append(metadata.instrument_id)
        values["anchor_source"].append(metadata.anchor_source)
        values["anchor_price"].append(metadata.anchor_price)
        values["tick_size"].append(metadata.tick_size)
        values["log_volume_reference"].append(metadata.log_volume_reference)
        values["provenance_sha256"].append(metadata.provenance_sha256)
        values["feature_schema_sha256"].append(FEATURE_SCHEMA_SHA256)
        values["target_schema_sha256"].append(TARGET_SCHEMA_SHA256)
    arrays = []
    for field in schema:
        if field.name not in tensor_names:
            arrays.append(pa.array(values[field.name], type=field.type))
            continue
        tensors = []
        for item in examples:
            tensor = getattr(item.batch, field.name)
            assert tensor is not None
            tensors.append(tensor.detach().cpu())
        flattened = torch.cat(tensors, dim=0).contiguous().reshape(-1).numpy()
        child = pa.array(flattened, type=field.type.value_type)
        arrays.append(pa.FixedSizeListArray.from_arrays(child, list_size=field.type.list_size))
    return pa.Table.from_arrays(arrays, schema=schema)


def _column_by_name(value: Any, name: str) -> Any:
    return value.column(value.schema.get_field_index(name))


def _seeded_stream_key(
    seed: int, dataset_id: str, scope: str, *identity: str
) -> tuple[bytes, tuple[str, ...]]:
    parts = (str(seed), dataset_id, scope, *identity)
    material = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(material).digest(), identity


def _validate_relative_shard_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("shard path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        or path.suffix not in {".arrow", ".ipc"}
    ):
        raise ValueError("shard path must be normalized, relative, and traversal-free")


def _validate_relative_evidence_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("evidence path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
        or value.casefold() == DATASET_MANIFEST_NAME.casefold()
    ):
        raise ValueError(
            "evidence path must be normalized, relative, traversal-free, and non-reserved"
        )


def _resolve_shard_path(root: Path, value: str) -> Path:
    _validate_relative_shard_path(value)
    lexical = root.joinpath(*PurePosixPath(value).parts)
    if lexical.is_symlink():
        raise ValueError(f"shard {value} must not be a symbolic link")
    resolved = lexical.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"shard {value} resolves outside the dataset root") from error
    if not resolved.is_file():
        raise ValueError(f"shard {value} must be a regular file")
    return resolved


def _resolve_evidence_path(root: Path, value: str) -> Path:
    _validate_relative_evidence_path(value)
    lexical = root
    for part in PurePosixPath(value).parts:
        lexical = lexical / part
        if lexical.is_symlink():
            raise ValueError(f"evidence artifact {value} must not use symbolic links")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"evidence artifact {value} is missing") from error
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"evidence artifact {value} resolves outside the dataset root") from error
    if not stat.S_ISREG(resolved.stat(follow_symlinks=False).st_mode):
        raise ValueError(f"evidence artifact {value} must be a regular file")
    return resolved


def _schema_sha256(schema: Any) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError(
            "dataset operations require the optional data dependency: "
            "pip install 'daxalgo-numeric-transformer[data]'"
        ) from error
    return pa


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def _validate_instrument_id(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a canonical non-empty string")


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a JSON array")
    return value


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
