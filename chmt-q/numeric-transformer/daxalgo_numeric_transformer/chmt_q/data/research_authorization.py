"""Fail-closed authorization for local CHMT-Q development research.

This module does not change a compiled dataset's capability claims.  It records a
separate, content-addressed exception that is intentionally limited to local
development training and validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from .dataset import DatasetManifest

AUTHORIZATION_SCHEMA_VERSION: Final = 1
AUTHORIZATION_KIND: Final = "chmt-q-local-research-authorization-v1"
RESEARCH_DATASET_ID_SCHEME: Final = "chmt-q-local-research-dataset-v1"
COMPILER_REPORT_NAME: Final = "okx-compiler-report.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_AUTHORIZATION_BYTES = 1024 * 1024
_MAX_COMPILER_REPORT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class LocalResearchAuthorization:
    """An immutable exception for local development train/validation only."""

    dataset_id: str
    instrument_id: str
    source_manifest_sha256: str
    dataset_capabilities_sha256: str
    compiler_report_sha256: str
    authorizer: str
    authorized_at_utc: str
    reason: str
    rights_confirmed: bool
    incomplete_readiness_acknowledged: bool
    incomplete_provenance_acknowledged: bool
    local_development_training_allowed: bool = True
    local_development_validation_allowed: bool = True
    promotion_allowed: bool = False
    export_allowed: bool = False
    serving_allowed: bool = False
    outer_holdout_allowed: bool = False
    schema_version: int = AUTHORIZATION_SCHEMA_VERSION
    kind: str = AUTHORIZATION_KIND

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != AUTHORIZATION_SCHEMA_VERSION
        ):
            raise ValueError("unsupported research-authorization schema_version")
        if self.kind != AUTHORIZATION_KIND:
            raise ValueError("unsupported research-authorization kind")
        for name in (
            "dataset_id",
            "source_manifest_sha256",
            "dataset_capabilities_sha256",
            "compiler_report_sha256",
        ):
            _validate_sha256(getattr(self, name), name)
        _validate_canonical_text(self.instrument_id, "instrument_id")
        _validate_canonical_text(self.authorizer, "authorizer")
        _validate_canonical_text(self.reason, "reason")
        _validate_utc_timestamp(self.authorized_at_utc)

        for name in (
            "rights_confirmed",
            "incomplete_readiness_acknowledged",
            "incomplete_provenance_acknowledged",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} must be explicitly true")
        for name in (
            "local_development_training_allowed",
            "local_development_validation_allowed",
        ):
            if getattr(self, name) is not True:
                raise ValueError(f"{name} is fixed to true")
        for name in (
            "promotion_allowed",
            "export_allowed",
            "serving_allowed",
            "outer_holdout_allowed",
        ):
            if getattr(self, name) is not False:
                raise ValueError(f"{name} is fixed to false")

    @classmethod
    def create(
        cls,
        manifest: DatasetManifest,
        dataset_root: str | Path,
        *,
        authorizer: str,
        reason: str,
        rights_confirmed: bool,
        incomplete_readiness_acknowledged: bool,
        incomplete_provenance_acknowledged: bool,
        authorized_at_utc: datetime | str | None = None,
    ) -> LocalResearchAuthorization:
        """Create an authorization bound to the current manifest and compiler report."""

        _validate_manifest_for_research_exception(manifest)
        for name, value in (
            ("rights_confirmed", rights_confirmed),
            (
                "incomplete_readiness_acknowledged",
                incomplete_readiness_acknowledged,
            ),
            (
                "incomplete_provenance_acknowledged",
                incomplete_provenance_acknowledged,
            ),
        ):
            if value is not True:
                raise ValueError(f"{name} must be explicitly true")

        report_bytes, report = _load_compiler_report(dataset_root)
        _validate_compiler_report_binding(report, manifest)
        return cls(
            dataset_id=manifest.dataset_id,
            instrument_id=manifest.instrument_id,
            source_manifest_sha256=manifest.source_manifest_sha256,
            dataset_capabilities_sha256=_capabilities_sha256(manifest),
            compiler_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
            authorizer=authorizer,
            authorized_at_utc=_format_utc_timestamp(authorized_at_utc),
            reason=reason,
            rights_confirmed=rights_confirmed,
            incomplete_readiness_acknowledged=incomplete_readiness_acknowledged,
            incomplete_provenance_acknowledged=incomplete_provenance_acknowledged,
        )

    def payload(self) -> dict[str, Any]:
        """Return the complete canonical authorization payload."""

        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "dataset_id": self.dataset_id,
            "instrument_id": self.instrument_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "dataset_capabilities_sha256": self.dataset_capabilities_sha256,
            "compiler_report_sha256": self.compiler_report_sha256,
            "authorizer": self.authorizer,
            "authorized_at_utc": self.authorized_at_utc,
            "reason": self.reason,
            "rights_confirmed": self.rights_confirmed,
            "incomplete_readiness_acknowledged": (self.incomplete_readiness_acknowledged),
            "incomplete_provenance_acknowledged": (self.incomplete_provenance_acknowledged),
            "local_development_training_allowed": (self.local_development_training_allowed),
            "local_development_validation_allowed": (self.local_development_validation_allowed),
            "promotion_allowed": self.promotion_allowed,
            "export_allowed": self.export_allowed,
            "serving_allowed": self.serving_allowed,
            "outer_holdout_allowed": self.outer_holdout_allowed,
        }

    @property
    def authorization_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("ascii")).hexdigest()

    @property
    def research_dataset_id(self) -> str:
        identity = {
            "scheme": RESEARCH_DATASET_ID_SCHEME,
            "source_dataset_id": self.dataset_id,
            "authorization_sha256": self.authorization_sha256,
        }
        return hashlib.sha256(_canonical_json(identity).encode("ascii")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json(
            {
                "authorization_sha256": self.authorization_sha256,
                "research_dataset_id": self.research_dataset_id,
                "payload": self.payload(),
            }
        )

    @classmethod
    def from_json(cls, value: str) -> LocalResearchAuthorization:
        document = _parse_strict_json(value, "research authorization")
        root = _require_object(
            document,
            "research authorization",
            {"authorization_sha256", "research_dataset_id", "payload"},
        )
        authorization_sha256 = _require_string(root["authorization_sha256"], "authorization_sha256")
        research_dataset_id = _require_string(root["research_dataset_id"], "research_dataset_id")
        _validate_sha256(authorization_sha256, "authorization_sha256")
        _validate_sha256(research_dataset_id, "research_dataset_id")
        expected = {
            "schema_version",
            "kind",
            "dataset_id",
            "instrument_id",
            "source_manifest_sha256",
            "dataset_capabilities_sha256",
            "compiler_report_sha256",
            "authorizer",
            "authorized_at_utc",
            "reason",
            "rights_confirmed",
            "incomplete_readiness_acknowledged",
            "incomplete_provenance_acknowledged",
            "local_development_training_allowed",
            "local_development_validation_allowed",
            "promotion_allowed",
            "export_allowed",
            "serving_allowed",
            "outer_holdout_allowed",
        }
        payload = _require_object(root["payload"], "authorization payload", expected)
        authorization = cls(
            schema_version=_require_integer(payload["schema_version"], "schema_version"),
            kind=_require_string(payload["kind"], "kind"),
            dataset_id=_require_string(payload["dataset_id"], "dataset_id"),
            instrument_id=_require_string(payload["instrument_id"], "instrument_id"),
            source_manifest_sha256=_require_string(
                payload["source_manifest_sha256"], "source_manifest_sha256"
            ),
            dataset_capabilities_sha256=_require_string(
                payload["dataset_capabilities_sha256"],
                "dataset_capabilities_sha256",
            ),
            compiler_report_sha256=_require_string(
                payload["compiler_report_sha256"], "compiler_report_sha256"
            ),
            authorizer=_require_string(payload["authorizer"], "authorizer"),
            authorized_at_utc=_require_string(payload["authorized_at_utc"], "authorized_at_utc"),
            reason=_require_string(payload["reason"], "reason"),
            rights_confirmed=_require_bool(payload["rights_confirmed"], "rights_confirmed"),
            incomplete_readiness_acknowledged=_require_bool(
                payload["incomplete_readiness_acknowledged"],
                "incomplete_readiness_acknowledged",
            ),
            incomplete_provenance_acknowledged=_require_bool(
                payload["incomplete_provenance_acknowledged"],
                "incomplete_provenance_acknowledged",
            ),
            local_development_training_allowed=_require_bool(
                payload["local_development_training_allowed"],
                "local_development_training_allowed",
            ),
            local_development_validation_allowed=_require_bool(
                payload["local_development_validation_allowed"],
                "local_development_validation_allowed",
            ),
            promotion_allowed=_require_bool(payload["promotion_allowed"], "promotion_allowed"),
            export_allowed=_require_bool(payload["export_allowed"], "export_allowed"),
            serving_allowed=_require_bool(payload["serving_allowed"], "serving_allowed"),
            outer_holdout_allowed=_require_bool(
                payload["outer_holdout_allowed"], "outer_holdout_allowed"
            ),
        )
        if authorization.authorization_sha256 != authorization_sha256:
            raise ValueError("authorization SHA-256 does not match its canonical payload")
        if authorization.research_dataset_id != research_dataset_id:
            raise ValueError("research_dataset_id does not match the authorization")
        if value != authorization.to_json():
            raise ValueError("research authorization JSON is not in canonical form")
        return authorization

    def validate_against(self, manifest: DatasetManifest, dataset_root: str | Path) -> None:
        """Fail unless every dataset and report binding still matches exactly."""

        _validate_manifest_for_research_exception(manifest)
        bindings = (
            ("dataset_id", self.dataset_id, manifest.dataset_id),
            ("instrument_id", self.instrument_id, manifest.instrument_id),
            (
                "source_manifest_sha256",
                self.source_manifest_sha256,
                manifest.source_manifest_sha256,
            ),
            (
                "dataset_capabilities_sha256",
                self.dataset_capabilities_sha256,
                _capabilities_sha256(manifest),
            ),
        )
        for name, authorized, observed in bindings:
            if authorized != observed:
                raise ValueError(f"authorization {name} differs from the dataset manifest")

        report_bytes, report = _load_compiler_report(dataset_root)
        observed_report_sha256 = hashlib.sha256(report_bytes).hexdigest()
        if observed_report_sha256 != self.compiler_report_sha256:
            raise ValueError("authorization compiler-report SHA-256 differs from the dataset")
        _validate_compiler_report_binding(report, manifest)

    def write_new(self, path: str | Path) -> Path:
        """Atomically publish a new immutable authorization without replacement."""

        target = _absolute_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        _validate_entry(target.parent, "authorization parent", final_kind="directory")
        _reject_existing_write_target(target)

        content = self.to_json().encode("ascii")
        staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(staging, flags, 0o600)
            offset = 0
            while offset < len(content):
                written = os.write(descriptor, content[offset:])
                if written <= 0:
                    raise OSError("authorization write made no progress")
                offset += written
            os.fsync(descriptor)
            created = os.fstat(descriptor)
            if not stat.S_ISREG(created.st_mode):
                raise RuntimeError("authorization staging path is not a regular file")
            os.close(descriptor)
            descriptor = None

            try:
                os.link(staging, target)
            except FileExistsError as error:
                _raise_existing_write_target(target, error)
            published = _validate_entry(target, "authorization", final_kind="file")
            if not os.path.samestat(created, published):
                raise RuntimeError("authorization changed during atomic publication")
            return target
        finally:
            if descriptor is not None:
                os.close(descriptor)
            try:
                staging.unlink()
            except FileNotFoundError:
                pass

    @classmethod
    def read(cls, path: str | Path) -> LocalResearchAuthorization:
        """Read a canonical authorization from a regular, non-symlink file."""

        content = _read_regular_bytes(
            _absolute_path(path),
            "research authorization",
            maximum_bytes=_MAX_AUTHORIZATION_BYTES,
        )
        try:
            value = content.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("research authorization must be canonical ASCII JSON") from error
        return cls.from_json(value)


def _validate_manifest_for_research_exception(manifest: DatasetManifest) -> None:
    if not isinstance(manifest, DatasetManifest):
        raise TypeError("manifest must be a DatasetManifest")
    if manifest.capabilities.training_ready:
        raise ValueError("local research authorization requires incomplete training readiness")
    if manifest.capabilities.provenance_complete:
        raise ValueError("local research authorization requires incomplete provenance")


def _capabilities_sha256(manifest: DatasetManifest) -> str:
    payload = manifest.capabilities.to_payload()
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _load_compiler_report(dataset_root: str | Path) -> tuple[bytes, dict[str, Any]]:
    root = _absolute_path(dataset_root)
    _validate_entry(root, "dataset_root", final_kind="directory")
    report_bytes = _read_regular_bytes(
        root / COMPILER_REPORT_NAME,
        COMPILER_REPORT_NAME,
        maximum_bytes=_MAX_COMPILER_REPORT_BYTES,
    )
    try:
        report_text = report_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("compiler report must be UTF-8 JSON") from error
    report = _parse_strict_json(report_text, "compiler report")
    if not isinstance(report, dict):
        raise ValueError("compiler report must be a JSON object")
    return report_bytes, report


def _validate_compiler_report_binding(report: dict[str, Any], manifest: DatasetManifest) -> None:
    required = {
        "status",
        "dataset_id",
        "instrument_id",
        "source_manifest_sha256",
        "capabilities",
    }
    missing = sorted(required - set(report))
    if missing:
        raise ValueError("compiler report is missing binding fields: " + ", ".join(missing))
    expected = {
        "status": "compiled_non_promotable",
        "dataset_id": manifest.dataset_id,
        "instrument_id": manifest.instrument_id,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "capabilities": manifest.capabilities.to_payload(),
    }
    for name, value in expected.items():
        if report[name] != value:
            raise ValueError(f"compiler report {name} differs from the dataset manifest")


def _format_utc_timestamp(value: datetime | str | None) -> str:
    if value is None:
        timestamp = datetime.now(UTC)
    elif isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        _validate_utc_timestamp(value)
        return value
    else:
        raise TypeError("authorized_at_utc must be a datetime, string, or None")
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("authorized_at_utc must be timezone-aware UTC")
    if timestamp.utcoffset().total_seconds() != 0:
        raise ValueError("authorized_at_utc must use UTC")
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError("authorized_at_utc must be canonical UTC with a Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError("authorized_at_utc must be a valid ISO-8601 timestamp") from error
    canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if value != canonical:
        raise ValueError("authorized_at_utc must be canonical ISO-8601 UTC")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _parse_strict_json(value: str, name: str) -> Any:
    if not isinstance(value, str):
        raise TypeError(f"{name} JSON must be a string")

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
        raise ValueError(f"{name} is not valid JSON") from error


def _validate_sha256(value: Any, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def _validate_canonical_text(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in value)
    ):
        raise ValueError(f"{name} must be a canonical non-empty string")


def _require_object(value: Any, name: str, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(value)))


def _is_link_or_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(value.st_mode) or bool(attributes & reparse_flag)


def _validate_entry(
    path: Path,
    name: str,
    *,
    final_kind: Literal["file", "directory"],
) -> os.stat_result:
    try:
        observed = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{name} does not exist") from error
    except OSError as error:
        raise ValueError(f"{name} cannot be inspected") from error
    if _is_link_or_reparse(observed):
        raise ValueError(f"{name} must not be a symbolic link or reparse point")
    is_expected = (
        stat.S_ISREG(observed.st_mode) if final_kind == "file" else stat.S_ISDIR(observed.st_mode)
    )
    if not is_expected:
        raise ValueError(f"{name} must be a regular {final_kind}")
    return observed


def _read_regular_bytes(path: Path, name: str, *, maximum_bytes: int) -> bytes:
    before = _validate_entry(path, name, final_kind="file")
    if before.st_size > maximum_bytes:
        raise ValueError(f"{name} exceeds the maximum permitted size")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} could not be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
            raise ValueError(f"{name} changed while it was being opened")
        chunks: list[bytes] = []
        length = 0
        while chunk := os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - length)):
            chunks.append(chunk)
            length += len(chunk)
            if length > maximum_bytes:
                raise ValueError(f"{name} exceeds the maximum permitted size")
        after_read = os.fstat(descriptor)
        if not os.path.samestat(opened, after_read) or after_read.st_size != length:
            raise ValueError(f"{name} changed while it was being read")
    finally:
        os.close(descriptor)
    after = _validate_entry(path, name, final_kind="file")
    if not os.path.samestat(opened, after):
        raise ValueError(f"{name} changed while it was being verified")
    return b"".join(chunks)


def _reject_existing_write_target(path: Path) -> None:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise ValueError("authorization path cannot be inspected") from error
    if _is_link_or_reparse(observed):
        raise ValueError("authorization path must not be a symbolic link or reparse point")
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError("authorization path is occupied by a non-regular file")
    raise FileExistsError(f"authorization already exists: {path}")


def _raise_existing_write_target(path: Path, error: FileExistsError) -> None:
    try:
        _reject_existing_write_target(path)
    except (FileExistsError, ValueError) as inspected:
        raise inspected from error
    raise FileExistsError(f"authorization already exists: {path}") from error
