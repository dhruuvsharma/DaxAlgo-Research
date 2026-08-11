"""Deterministic session-group chronological splits with exact leakage purging."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

ROLE_ORDER = ("development", "train", "validation", "calibration", "outer_test")
INTERVAL_SEMANTICS = "half-open [dependency_start_ns, dependency_end_ns)"
BUILD_METHOD = "chronological-session-groups-exact-interval-purge-v1"


@dataclass(frozen=True, slots=True)
class ExampleDependency:
    """One example and the complete feature-through-label interval it depends on."""

    example_id: str
    session_id: str
    decision_ns: int
    dependency_start_ns: int
    dependency_end_ns: int

    def __post_init__(self) -> None:
        if not self.example_id or not self.session_id:
            raise ValueError("example_id and session_id must not be empty")
        for name in ("decision_ns", "dependency_start_ns", "dependency_end_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer nanosecond timestamp")
        if not self.dependency_start_ns <= self.decision_ns < self.dependency_end_ns:
            raise ValueError(
                "dependency interval must contain decision_ns and have positive duration"
            )

    def overlaps(self, other: ExampleDependency) -> bool:
        """Return exact overlap under the declared half-open interval convention."""

        return (
            self.dependency_start_ns < other.dependency_end_ns
            and other.dependency_start_ns < self.dependency_end_ns
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "session_id": self.session_id,
            "decision_ns": self.decision_ns,
            "dependency_start_ns": self.dependency_start_ns,
            "dependency_end_ns": self.dependency_end_ns,
        }


@dataclass(frozen=True, slots=True)
class RoleSplit:
    """Session assignment and retained/purged examples for one chronological role."""

    role: str
    session_ids: tuple[str, ...]
    example_ids: tuple[str, ...]
    purged_example_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in ROLE_ORDER:
            raise ValueError(f"unknown split role: {self.role}")
        if not self.session_ids:
            raise ValueError(f"{self.role} must contain at least one session")
        if len(set(self.session_ids)) != len(self.session_ids):
            raise ValueError(f"{self.role} contains duplicate sessions")
        if not self.example_ids:
            raise ValueError(f"{self.role} has no retained examples after purge/embargo")
        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError(f"{self.role} contains duplicate retained examples")
        if len(set(self.purged_example_ids)) != len(self.purged_example_ids):
            raise ValueError(f"{self.role} contains duplicate purged examples")
        if set(self.example_ids) & set(self.purged_example_ids):
            raise ValueError(f"{self.role} retains and purges the same example")

    def to_payload(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "session_ids": list(self.session_ids),
            "example_ids": list(self.example_ids),
            "purged_example_ids": list(self.purged_example_ids),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> RoleSplit:
        expected = {"role", "session_ids", "example_ids", "purged_example_ids"}
        if set(payload) != expected:
            raise ValueError("role payload has missing or unexpected fields")
        return cls(
            role=str(payload["role"]),
            session_ids=tuple(str(value) for value in payload["session_ids"]),
            example_ids=tuple(str(value) for value in payload["example_ids"]),
            purged_example_ids=tuple(
                str(value) for value in payload["purged_example_ids"]
            ),
        )


@dataclass(frozen=True, slots=True)
class PurgeRecord:
    """Auditable reason an earlier-side example was removed."""

    example_id: str
    assigned_role: str
    against_role: str
    reason: str

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("purged example_id must not be empty")
        if self.assigned_role not in ROLE_ORDER or self.against_role not in ROLE_ORDER:
            raise ValueError("purge record contains an unknown role")
        if ROLE_ORDER.index(self.assigned_role) >= ROLE_ORDER.index(self.against_role):
            raise ValueError("purge must remove an earlier role against a later role")
        if self.reason not in {"dependency_overlap", "embargo"}:
            raise ValueError("purge reason must be dependency_overlap or embargo")

    def to_payload(self) -> dict[str, str]:
        return {
            "example_id": self.example_id,
            "assigned_role": self.assigned_role,
            "against_role": self.against_role,
            "reason": self.reason,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PurgeRecord:
        expected = {"example_id", "assigned_role", "against_role", "reason"}
        if set(payload) != expected:
            raise ValueError("purge payload has missing or unexpected fields")
        return cls(
            example_id=str(payload["example_id"]),
            assigned_role=str(payload["assigned_role"]),
            against_role=str(payload["against_role"]),
            reason=str(payload["reason"]),
        )


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Content-addressed chronological split and its exact purge decisions."""

    source_dataset_hash: str
    embargo_ns: int
    roles: tuple[RoleSplit, ...]
    purges: tuple[PurgeRecord, ...]
    schema_version: int = 1
    interval_semantics: str = INTERVAL_SEMANTICS
    build_method: str = BUILD_METHOD

    ROLE_ORDER: ClassVar[tuple[str, ...]] = ROLE_ORDER

    def __post_init__(self) -> None:
        _validate_sha256(self.source_dataset_hash, "source_dataset_hash")
        if isinstance(self.embargo_ns, bool) or not isinstance(self.embargo_ns, int):
            raise TypeError("embargo_ns must be an integer")
        if self.embargo_ns < 0:
            raise ValueError("embargo_ns must be non-negative")
        if self.schema_version != 1:
            raise ValueError("unsupported split schema_version")
        if self.interval_semantics != INTERVAL_SEMANTICS or self.build_method != BUILD_METHOD:
            raise ValueError("split interval semantics and build method are versioned constants")
        if tuple(role.role for role in self.roles) != ROLE_ORDER:
            raise ValueError("split roles must appear exactly once in chronological role order")

        sessions = [session for role in self.roles for session in role.session_ids]
        if len(set(sessions)) != len(sessions):
            raise ValueError("a session is assigned to more than one role")
        retained = [example for role in self.roles for example in role.example_ids]
        purged = [example for role in self.roles for example in role.purged_example_ids]
        if len(set(retained)) != len(retained) or len(set(purged)) != len(purged):
            raise ValueError("an example appears more than once in the split")
        if set(retained) & set(purged):
            raise ValueError("an example is both retained and purged")
        if tuple(record.example_id for record in self.purges) != tuple(purged):
            raise ValueError("purge records must match role purged_example_ids in order")

    def role(self, name: str) -> RoleSplit:
        for role in self.roles:
            if role.role == name:
                return role
        raise KeyError(name)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_dataset_hash": self.source_dataset_hash,
            "embargo_ns": self.embargo_ns,
            "interval_semantics": self.interval_semantics,
            "build_method": self.build_method,
            "roles": [role.to_payload() for role in self.roles],
            "purges": [record.to_payload() for record in self.purges],
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json(
            {"artifact_sha256": self.artifact_sha256, "payload": self.payload()}
        )

    @classmethod
    def from_json(cls, value: str) -> SplitManifest:
        parsed = json.loads(value)
        if not isinstance(parsed, dict) or set(parsed) != {"artifact_sha256", "payload"}:
            raise ValueError("split JSON must contain artifact_sha256 and payload")
        payload = parsed["payload"]
        if not isinstance(payload, dict):
            raise ValueError("split payload must be an object")
        expected_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if parsed["artifact_sha256"] != expected_hash:
            raise ValueError("split artifact SHA-256 does not match its payload")
        expected = {
            "schema_version",
            "source_dataset_hash",
            "embargo_ns",
            "interval_semantics",
            "build_method",
            "roles",
            "purges",
        }
        if set(payload) != expected:
            raise ValueError("split payload has missing or unexpected fields")
        manifest = cls(
            schema_version=int(payload["schema_version"]),
            source_dataset_hash=str(payload["source_dataset_hash"]),
            embargo_ns=int(payload["embargo_ns"]),
            interval_semantics=str(payload["interval_semantics"]),
            build_method=str(payload["build_method"]),
            roles=tuple(RoleSplit.from_payload(item) for item in payload["roles"]),
            purges=tuple(PurgeRecord.from_payload(item) for item in payload["purges"]),
        )
        if manifest.artifact_sha256 != expected_hash:
            raise ValueError("reconstructed split identity differs from serialized payload")
        return manifest

    def validate(self, examples: Iterable[ExampleDependency]) -> SplitManifest:
        """Recheck membership, chronology, purge reasons, and retained leakage boundaries."""

        ordered = _validate_and_order_examples(examples)
        by_id = {example.example_id: example for example in ordered}
        assigned = {
            example_id
            for role in self.roles
            for example_id in (*role.example_ids, *role.purged_example_ids)
        }
        if assigned != set(by_id):
            raise ValueError("manifest example membership differs from the supplied dataset")
        observed_sessions = {example.session_id for example in ordered}
        assigned_sessions = {session for role in self.roles for session in role.session_ids}
        if observed_sessions != assigned_sessions:
            raise ValueError("manifest session membership differs from the supplied dataset")

        role_by_session = {
            session: role.role for role in self.roles for session in role.session_ids
        }
        for role in self.roles:
            for example_id in (*role.example_ids, *role.purged_example_ids):
                if role_by_session[by_id[example_id].session_id] != role.role:
                    raise ValueError("example is assigned outside its session's role")
            chronological_sessions = tuple(
                sorted(
                    role.session_ids,
                    key=lambda session: (
                        min(
                            item.decision_ns
                            for item in ordered
                            if item.session_id == session
                        ),
                        session,
                    ),
                )
            )
            if chronological_sessions != role.session_ids:
                raise ValueError("role sessions are not in deterministic chronological order")
            if tuple(sorted(role.example_ids, key=lambda item: _example_key(by_id[item]))) != (
                role.example_ids
            ):
                raise ValueError("role examples are not in deterministic chronological order")

        retained_by_role = {
            role.role: tuple(by_id[item] for item in role.example_ids) for role in self.roles
        }
        for earlier_index, earlier_role in enumerate(ROLE_ORDER[:-1]):
            for later_role in ROLE_ORDER[earlier_index + 1 :]:
                for earlier in retained_by_role[earlier_role]:
                    for later in retained_by_role[later_role]:
                        if _conflict_reason(earlier, later, self.embargo_ns) is not None:
                            raise ValueError(
                                f"retained leakage between {earlier_role} and {later_role}"
                            )

        train = retained_by_role["train"]
        for later_role in ("validation", "calibration", "outer_test"):
            later = retained_by_role[later_role]
            if max(item.decision_ns for item in train) >= min(
                item.decision_ns for item in later
            ):
                raise ValueError(f"train must precede {later_role}")

        records = {record.example_id: record for record in self.purges}
        for role in self.roles:
            for example_id in role.purged_example_ids:
                record = records[example_id]
                earlier = by_id[example_id]
                later_examples = retained_by_role[record.against_role]
                reasons = {
                    _conflict_reason(earlier, later, self.embargo_ns)
                    for later in later_examples
                }
                if record.reason not in reasons:
                    raise ValueError("purge record cannot be reproduced from dependency intervals")
        return self


def build_chronological_split(
    examples: Iterable[ExampleDependency],
    *,
    role_session_counts: Mapping[str, int],
    source_dataset_hash: str,
    embargo_ns: int = 0,
) -> SplitManifest:
    """Assign complete sessions chronologically and purge conflicting earlier examples."""

    _validate_sha256(source_dataset_hash, "source_dataset_hash")
    if isinstance(embargo_ns, bool) or not isinstance(embargo_ns, int):
        raise TypeError("embargo_ns must be an integer")
    if embargo_ns < 0:
        raise ValueError("embargo_ns must be non-negative")
    if set(role_session_counts) != set(ROLE_ORDER):
        raise ValueError("role_session_counts must contain every role exactly once")
    counts: dict[str, int] = {}
    for role in ROLE_ORDER:
        value = role_session_counts[role]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("every role must receive a positive integer session count")
        counts[role] = value

    ordered = _validate_and_order_examples(examples)
    grouped: dict[str, list[ExampleDependency]] = {}
    for example in ordered:
        grouped.setdefault(example.session_id, []).append(example)
    sessions = sorted(
        grouped,
        key=lambda session: (
            min(item.decision_ns for item in grouped[session]),
            session,
        ),
    )
    if sum(counts.values()) != len(sessions):
        raise ValueError("role session counts must exactly consume all sessions")
    for left, right in zip(sessions, sessions[1:]):
        if max(item.decision_ns for item in grouped[left]) >= min(
            item.decision_ns for item in grouped[right]
        ):
            raise ValueError("session decision ranges interleave; group chronology is ambiguous")

    role_sessions: dict[str, tuple[str, ...]] = {}
    role_examples: dict[str, tuple[ExampleDependency, ...]] = {}
    cursor = 0
    for role in ROLE_ORDER:
        selected = tuple(sessions[cursor : cursor + counts[role]])
        cursor += counts[role]
        role_sessions[role] = selected
        role_examples[role] = tuple(
            sorted(
                (item for session in selected for item in grouped[session]),
                key=_example_key,
            )
        )

    retained_by_role: dict[str, tuple[ExampleDependency, ...]] = {}
    purged_by_role: dict[str, tuple[ExampleDependency, ...]] = {}
    records_by_role: dict[str, tuple[PurgeRecord, ...]] = {}
    for role_index in range(len(ROLE_ORDER) - 1, -1, -1):
        role = ROLE_ORDER[role_index]
        retained: list[str] = []
        purged: list[str] = []
        records: list[PurgeRecord] = []
        for example in role_examples[role]:
            record = _first_conflict(
                example,
                role_index,
                retained_by_role,
                embargo_ns,
            )
            if record is None:
                retained.append(example.example_id)
            else:
                against_role, reason = record
                purged.append(example.example_id)
                records.append(PurgeRecord(example.example_id, role, against_role, reason))
        retained_ids = set(retained)
        purged_ids = set(purged)
        retained_by_role[role] = tuple(
            item for item in role_examples[role] if item.example_id in retained_ids
        )
        purged_by_role[role] = tuple(
            item for item in role_examples[role] if item.example_id in purged_ids
        )
        records_by_role[role] = tuple(records)

    role_splits: list[RoleSplit] = []
    purge_records: list[PurgeRecord] = []
    for role in ROLE_ORDER:
        role_splits.append(
            RoleSplit(
                role=role,
                session_ids=role_sessions[role],
                example_ids=tuple(item.example_id for item in retained_by_role[role]),
                purged_example_ids=tuple(item.example_id for item in purged_by_role[role]),
            )
        )
        purge_records.extend(records_by_role[role])

    manifest = SplitManifest(
        source_dataset_hash=source_dataset_hash,
        embargo_ns=embargo_ns,
        roles=tuple(role_splits),
        purges=tuple(purge_records),
    )
    return manifest.validate(ordered)


def _validate_and_order_examples(
    examples: Iterable[ExampleDependency],
) -> tuple[ExampleDependency, ...]:
    materialized = tuple(examples)
    if not materialized:
        raise ValueError("at least one dependency example is required")
    if any(not isinstance(item, ExampleDependency) for item in materialized):
        raise TypeError("examples must be ExampleDependency instances")
    ids = [item.example_id for item in materialized]
    if len(set(ids)) != len(ids):
        raise ValueError("example_id values must be unique")
    return tuple(sorted(materialized, key=_example_key))


def _example_key(example: ExampleDependency) -> tuple[int, int, int, str]:
    return (
        example.decision_ns,
        example.dependency_start_ns,
        example.dependency_end_ns,
        example.example_id,
    )


def _first_conflict(
    example: ExampleDependency,
    role_index: int,
    role_examples: Mapping[str, tuple[ExampleDependency, ...]],
    embargo_ns: int,
) -> tuple[str, str] | None:
    for later_role in ROLE_ORDER[role_index + 1 :]:
        for later in role_examples[later_role]:
            reason = _conflict_reason(example, later, embargo_ns)
            if reason is not None:
                return later_role, reason
    return None


def _conflict_reason(
    earlier: ExampleDependency, later: ExampleDependency, embargo_ns: int
) -> str | None:
    if earlier.overlaps(later):
        return "dependency_overlap"
    if (
        embargo_ns > 0
        and earlier.dependency_end_ns <= later.dependency_start_ns
        and earlier.dependency_end_ns > later.dependency_start_ns - embargo_ns
    ):
        return "embargo"
    return None


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
