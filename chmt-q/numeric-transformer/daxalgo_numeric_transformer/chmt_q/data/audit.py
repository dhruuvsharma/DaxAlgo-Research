"""Deterministic, read-only audit for the observed LSE BTC/USD tick Parquet schema.

The auditor describes source records exactly as stored. It never deduplicates rows,
assigns event semantics, or treats the numeric ``side`` field as an aggressor label.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_LSE_TICK_SCHEMA: tuple[tuple[str, str, bool], ...] = (
    ("ts", "timestamp[us, tz=UTC]", False),
    ("symbol", "string", False),
    ("price", "double", False),
    ("volume", "double", False),
    ("side", "uint8", False),
    ("ask", "double", True),
)
"""Name, Arrow dtype, and nullability for the observed source schema."""

_EXPECTED_COLUMNS = tuple(field[0] for field in EXPECTED_LSE_TICK_SCHEMA)
_NOMINAL_WINDOW = re.compile(
    r"(?P<start>\d{1,2})(?P<month>[A-Za-z]+)_(?P<end>\d{1,2})(?P=month)",
    re.IGNORECASE,
)
_MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


class FindingStatus(StrEnum):
    """Severity attached to an audit finding."""

    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """One deterministic, machine-readable audit conclusion."""

    code: str
    status: FindingStatus
    message: str
    file: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        result: dict[str, Any] = {
            "code": self.code,
            "status": self.status.value,
            "message": self.message,
        }
        if self.file is not None:
            result["file"] = self.file
        if self.details is not None:
            result["details"] = self.details
        return result


@dataclass(frozen=True, slots=True)
class ParquetFileAudit:
    """Audit result for one immutable Parquet source file."""

    path: str
    size_bytes: int
    sha256: str
    schema_sha256: str
    schema: tuple[dict[str, Any], ...]
    schema_valid: bool
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "schema": list(self.schema),
            "schema_valid": self.schema_valid,
            "metrics": self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ParquetAuditReport:
    """Serializable corpus audit with a fail-closed CHMT-Q readiness gate."""

    report_version: str
    files: tuple[ParquetFileAudit, ...]
    aggregate: dict[str, Any]
    findings: tuple[AuditFinding, ...]
    source_limitations: tuple[str, ...]
    training_blockers: tuple[str, ...]
    chmt_q_training_ready: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a fully JSON-compatible report."""

        return {
            "report_version": self.report_version,
            "files": [file.to_dict() for file in self.files],
            "aggregate": self.aggregate,
            "findings": [finding.to_dict() for finding in self.findings],
            "source_limitations": list(self.source_limitations),
            "training_blockers": list(self.training_blockers),
            "chmt_q_training_ready": self.chmt_q_training_ready,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize deterministically; non-finite JSON values are forbidden."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            indent=indent,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
        )


def audit_lse_tick_parquet(
    source: str | Path | Iterable[str | Path],
    *,
    boundary_tolerance_seconds: float = 60.0,
    internal_gap_warning_seconds: float = 10.0,
) -> ParquetAuditReport:
    """Audit one file, a directory, or an iterable of Parquet paths.

    Processing is bounded to one file at a time. The function only opens source
    files for reading. It computes exact within-file duplicate counts without
    deleting, filtering, or assigning meaning to any record.
    """

    if not math.isfinite(boundary_tolerance_seconds) or boundary_tolerance_seconds < 0:
        raise ValueError("boundary_tolerance_seconds must be finite and non-negative")
    if not math.isfinite(internal_gap_warning_seconds) or internal_gap_warning_seconds < 0:
        raise ValueError("internal_gap_warning_seconds must be finite and non-negative")

    pa, pc, pq = _require_pyarrow()
    paths = _resolve_paths(source)
    findings: list[AuditFinding] = []
    file_audits: list[ParquetFileAudit] = []

    for path in paths:
        file_audit, file_findings = _audit_file(
            path,
            pa=pa,
            pc=pc,
            pq=pq,
            boundary_tolerance_seconds=boundary_tolerance_seconds,
            internal_gap_warning_seconds=internal_gap_warning_seconds,
        )
        file_audits.append(file_audit)
        findings.extend(file_findings)

    file_audits.sort(key=_file_time_sort_key)
    aggregate, aggregate_findings = _aggregate(file_audits)
    findings.extend(aggregate_findings)

    source_limitations = (
        "The schema has one timestamp only; event, receive, and availability clocks "
        "and their provenance are absent.",
        "No source epoch, sequence number, trade ID, order ID, venue, or continuity "
        "marker can prove ordering, replay handling, or completeness.",
        "The numeric side field has no embedded aggressor-side semantics and must not "
        "be interpreted from its values.",
        "There is one ambiguous volume field rather than independent bid and ask sizes.",
        "No L2 side/rank levels, depth quantities, actions, or order lifecycle exist.",
        "No event-type or execution flag proves that rows form a genuine trade tape.",
        "File names and timestamps alone cannot prove full nominal-window capture.",
    )
    training_blockers = _training_blockers(file_audits, aggregate)
    findings.extend(_source_limitation_findings(training_blockers))

    status_counts = Counter(finding.status.value for finding in findings)
    aggregate["finding_counts"] = {
        status.value: int(status_counts.get(status.value, 0)) for status in FindingStatus
    }
    aggregate["source_manifest_sha256"] = _source_manifest_sha256(file_audits)

    # This observed schema cannot satisfy CHMT-Q's provenance, modality, or label contract.
    ready = not training_blockers and not any(
        finding.status is FindingStatus.FAIL for finding in findings
    )
    return ParquetAuditReport(
        report_version="1.0",
        files=tuple(file_audits),
        aggregate=aggregate,
        findings=tuple(findings),
        source_limitations=source_limitations,
        training_blockers=training_blockers,
        chmt_q_training_ready=ready,
    )


def _require_pyarrow() -> tuple[Any, Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on the caller's environment
        raise ImportError(
            "CHMT-Q Parquet audit requires the optional 'pyarrow' package; install "
            "pyarrow in the audit environment. Core model imports do not require it."
        ) from exc
    return pa, pc, pq


def _resolve_paths(source: str | Path | Iterable[str | Path]) -> tuple[Path, ...]:
    if isinstance(source, (str, Path)):
        candidate = Path(source).expanduser()
        candidates = list(candidate.glob("*.parquet")) if candidate.is_dir() else [candidate]
    else:
        candidates = [Path(item).expanduser() for item in source]

    normalized: dict[str, Path] = {}
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Parquet source does not exist: {resolved}")
        if resolved.suffix.casefold() != ".parquet":
            raise ValueError(f"source must be a .parquet file: {resolved}")
        normalized[str(resolved).casefold()] = resolved
    if not normalized:
        raise ValueError("no Parquet files were found")
    return tuple(normalized[key] for key in sorted(normalized))


def _audit_file(
    path: Path,
    *,
    pa: Any,
    pc: Any,
    pq: Any,
    boundary_tolerance_seconds: float,
    internal_gap_warning_seconds: float,
) -> tuple[ParquetFileAudit, list[AuditFinding]]:
    file_name = path.name
    digest = _sha256_file(path)
    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow
    schema_description = _schema_description(schema)
    schema_sha = _schema_sha256(schema_description)
    schema_valid = _schema_signature(schema) == EXPECTED_LSE_TICK_SCHEMA
    findings: list[AuditFinding] = []
    row_count = int(parquet_file.metadata.num_rows)
    metrics: dict[str, Any] = {
        "row_count": row_count,
        "row_group_count": int(parquet_file.metadata.num_row_groups),
        "created_by": parquet_file.metadata.created_by,
    }

    if not schema_valid:
        findings.append(
            AuditFinding(
                code="schema_mismatch",
                status=FindingStatus.FAIL,
                file=file_name,
                message="Parquet schema does not exactly match the observed LSE tick contract.",
                details={
                    "expected": [list(field) for field in EXPECTED_LSE_TICK_SCHEMA],
                    "actual": list(schema_description),
                },
            )
        )
        return (
            ParquetFileAudit(
                path=str(path),
                size_bytes=path.stat().st_size,
                sha256=digest,
                schema_sha256=schema_sha,
                schema=schema_description,
                schema_valid=False,
                metrics=metrics,
            ),
            findings,
        )

    findings.append(
        AuditFinding(
            code="schema_match",
            status=FindingStatus.PASS,
            file=file_name,
            message="Columns, Arrow dtypes, nullability, and UTC timestamp schema match.",
        )
    )

    table = parquet_file.read(columns=list(_EXPECTED_COLUMNS), use_threads=True)
    if table.num_rows != row_count:
        findings.append(
            AuditFinding(
                code="metadata_row_count_mismatch",
                status=FindingStatus.FAIL,
                file=file_name,
                message="Parquet metadata and decoded row counts differ.",
                details={"metadata": row_count, "decoded": int(table.num_rows)},
            )
        )
    if row_count == 0:
        findings.append(
            AuditFinding(
                code="empty_file",
                status=FindingStatus.FAIL,
                file=file_name,
                message="The source contains no rows.",
            )
        )
        metrics.update(_empty_metrics())
        return (
            ParquetFileAudit(
                path=str(path),
                size_bytes=path.stat().st_size,
                sha256=digest,
                schema_sha256=schema_sha,
                schema=schema_description,
                schema_valid=True,
                metrics=metrics,
            ),
            findings,
        )

    null_counts = {name: int(table[name].null_count) for name in _EXPECTED_COLUMNS}
    numeric = {
        name: _numeric_numpy(table[name], pc=pc)
        for name in ("price", "volume", "side", "ask")
    }
    nonfinite_counts = {
        name: int(np.count_nonzero(~np.isfinite(values)))
        for name, values in numeric.items()
    }
    nonpositive_counts = {
        name: int(np.count_nonzero(values <= 0.0))
        for name, values in numeric.items()
        if name in {"price", "volume", "ask"}
    }
    integrity_failed = any(null_counts.values()) or any(nonfinite_counts.values())
    integrity_failed = integrity_failed or any(nonpositive_counts.values())
    findings.append(
        AuditFinding(
            code="value_integrity",
            status=FindingStatus.FAIL if integrity_failed else FindingStatus.PASS,
            file=file_name,
            message=(
                "Null, non-finite, or non-positive required numeric values were found."
                if integrity_failed
                else (
                    "Required fields are non-null; prices, volume, and ask are "
                    "finite and positive."
                )
            ),
            details={
                "null_counts": null_counts,
                "nonfinite_counts": nonfinite_counts,
                "nonpositive_counts": nonpositive_counts,
            },
        )
    )

    timestamps_valid = null_counts["ts"] == 0
    timestamp_us = _timestamp_numpy(table["ts"], pa=pa, pc=pc)
    timestamp_diffs = np.diff(timestamp_us) if timestamps_valid else np.array([], dtype=np.int64)
    negative_diffs = int(np.count_nonzero(timestamp_diffs < 0))
    sorted_non_decreasing = timestamps_valid and negative_diffs == 0
    findings.append(
        AuditFinding(
            code="timestamp_order",
            status=FindingStatus.PASS if sorted_non_decreasing else FindingStatus.FAIL,
            file=file_name,
            message=(
                "Timestamps are non-decreasing."
                if sorted_non_decreasing
                else "Timestamps are null or decrease in file order."
            ),
            details={"negative_interarrival_count": negative_diffs},
        )
    )

    timestamp_groups = table.group_by(["ts"]).aggregate([("ts", "count")])
    row_groups = table.group_by(list(_EXPECTED_COLUMNS)).aggregate([("ts", "count")])
    timestamp_group_counts = _integer_numpy(timestamp_groups["ts_count"], pc=pc)
    row_group_counts = _integer_numpy(row_groups["ts_count"], pc=pc)
    unique_timestamp_count = int(timestamp_groups.num_rows)
    unique_row_count = int(row_groups.num_rows)
    timestamp_collision_surplus = row_count - unique_timestamp_count
    exact_duplicate_surplus = row_count - unique_row_count
    max_timestamp_multiplicity = int(timestamp_group_counts.max(initial=0))
    max_exact_duplicate_multiplicity = int(row_group_counts.max(initial=0))
    findings.append(
        AuditFinding(
            code="timestamp_collisions",
            status=(
                FindingStatus.WARNING
                if timestamp_collision_surplus
                else FindingStatus.PASS
            ),
            file=file_name,
            message=(
                "Multiple rows share timestamps; no sequence field orders tied events."
                if timestamp_collision_surplus
                else "Every row has a distinct timestamp."
            ),
            details={
                "collision_surplus": timestamp_collision_surplus,
                "maximum_multiplicity": max_timestamp_multiplicity,
            },
        )
    )
    findings.append(
        AuditFinding(
            code="exact_duplicates",
            status=FindingStatus.WARNING if exact_duplicate_surplus else FindingStatus.PASS,
            file=file_name,
            message=(
                "Exact complete-row field tuples repeat; the audit retains every row."
                if exact_duplicate_surplus
                else "No exact complete-row duplicate was found."
            ),
            details={
                "duplicate_surplus": exact_duplicate_surplus,
                "maximum_multiplicity": max_exact_duplicate_multiplicity,
            },
        )
    )

    symbol_counts = _value_counts(table, "symbol")
    side_counts = _value_counts(table, "side")
    findings.append(
        AuditFinding(
            code="symbol_distribution",
            status=(
                FindingStatus.PASS
                if symbol_counts == {"BTC/USD": row_count}
                else FindingStatus.WARNING
            ),
            file=file_name,
            message="Symbol distribution was recorded without remapping values.",
            details={"counts": symbol_counts},
        )
    )
    meaningful_side = len(side_counts) >= 2
    findings.append(
        AuditFinding(
            code="aggressor_side_distribution",
            status=FindingStatus.WARNING if meaningful_side else FindingStatus.FAIL,
            file=file_name,
            message=(
                "Multiple numeric side values exist, but their semantics remain unproven."
                if meaningful_side
                else "The side field is constant and cannot provide aggressor-side labels."
            ),
            details={"counts": side_counts},
        )
    )

    price = numeric["price"]
    volume = numeric["volume"]
    ask = numeric["ask"]
    ask_minus_price = ask - price
    ask_minus_price_finite = ask_minus_price[np.isfinite(ask_minus_price)]
    nonpositive_ask_minus_price = int(
        np.count_nonzero(ask_minus_price_finite <= 0.0)
    )
    rounded_ask_minus_price = np.round(ask_minus_price_finite, decimals=8)
    ask_minus_price_values, ask_minus_price_counts = np.unique(
        rounded_ask_minus_price, return_counts=True
    )
    ask_minus_price_changes = int(
        np.count_nonzero(np.diff(rounded_ask_minus_price) != 0.0)
    )
    ask_minus_price_jumps = np.abs(np.diff(rounded_ask_minus_price))
    max_ask_minus_price_jump = float(ask_minus_price_jumps.max(initial=0.0))
    ask_minus_price_min = _finite_min(ask_minus_price)
    ask_minus_price_max = _finite_max(ask_minus_price)
    ask_minus_price_scale_separation = bool(
        ask_minus_price_min is not None
        and ask_minus_price_min > 0.0
        and ask_minus_price_max is not None
        and ask_minus_price_max / ask_minus_price_min >= 10.0
    )
    ask_minus_price_summary = _ask_minus_price_value_summary(
        ask_minus_price_values, ask_minus_price_counts
    )
    if nonpositive_ask_minus_price:
        ask_minus_price_status = FindingStatus.FAIL
        ask_minus_price_message = (
            "Ask minus price is non-positive for one or more rows."
        )
    elif ask_minus_price_scale_separation:
        ask_minus_price_status = FindingStatus.WARNING
        ask_minus_price_message = (
            "Ask-minus-price values occupy scales separated by at least tenfold; "
            "the field semantics remain unproven."
        )
    else:
        ask_minus_price_status = FindingStatus.PASS
        ask_minus_price_message = (
            "Ask-minus-price values are positive with no tenfold scale separation."
        )
    findings.append(
        AuditFinding(
            code="ask_minus_price_scale_separation",
            status=ask_minus_price_status,
            file=file_name,
            message=ask_minus_price_message,
            details={
                "nonpositive_count": nonpositive_ask_minus_price,
                "distinct_rounded_8dp": int(ask_minus_price_values.size),
                "change_count": ask_minus_price_changes,
                "maximum_adjacent_jump": max_ask_minus_price_jump,
                "values": ask_minus_price_summary,
            },
        )
    )

    gaps_over_one_second = int(np.count_nonzero(timestamp_diffs > 1_000_000))
    gap_threshold_us = int(round(internal_gap_warning_seconds * 1_000_000))
    gaps_over_warning = int(np.count_nonzero(timestamp_diffs > gap_threshold_us))
    max_gap_seconds = (
        float(max(0, int(timestamp_diffs.max(initial=0))) / 1_000_000)
        if timestamp_diffs.size
        else 0.0
    )
    findings.append(
        AuditFinding(
            code="internal_gaps",
            status=FindingStatus.WARNING if gaps_over_warning else FindingStatus.PASS,
            file=file_name,
            message=(
                "Internal gaps exceed the configured warning threshold."
                if gaps_over_warning
                else "No internal gap exceeds the configured warning threshold."
            ),
            details={
                "warning_threshold_seconds": internal_gap_warning_seconds,
                "gaps_over_warning": gaps_over_warning,
                "maximum_gap_seconds": max_gap_seconds,
            },
        )
    )

    first_timestamp = _timestamp_iso(table["ts"], pc=pc, minimum=True)
    last_timestamp = _timestamp_iso(table["ts"], pc=pc, minimum=False)
    nominal_endpoint_span = _nominal_endpoint_span(
        path.name,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        boundary_tolerance_seconds=boundary_tolerance_seconds,
    )
    endpoints_span_nominal_window = bool(
        nominal_endpoint_span.get("endpoints_span_nominal_window", False)
    )
    findings.append(
        AuditFinding(
            code="nominal_window_endpoint_span",
            status=(
                FindingStatus.PASS
                if endpoints_span_nominal_window
                else FindingStatus.FAIL
            ),
            file=file_name,
            message=(
                "Observed endpoints span the parsed nominal window within tolerance; "
                "this does not establish capture completeness."
                if endpoints_span_nominal_window
                else "Observed endpoints do not span the parsed nominal window."
            ),
            details=nominal_endpoint_span,
        )
    )

    metrics.update(
        {
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "duration_seconds": _duration_seconds(first_timestamp, last_timestamp),
            "sorted_non_decreasing": sorted_non_decreasing,
            "negative_interarrival_count": negative_diffs,
            "zero_interarrival_count": int(np.count_nonzero(timestamp_diffs == 0)),
            "unique_timestamp_count": unique_timestamp_count,
            "timestamp_collision_surplus": timestamp_collision_surplus,
            "maximum_timestamp_multiplicity": max_timestamp_multiplicity,
            "unique_complete_row_count": unique_row_count,
            "exact_duplicate_surplus": exact_duplicate_surplus,
            "maximum_exact_duplicate_multiplicity": max_exact_duplicate_multiplicity,
            "null_counts": null_counts,
            "nonfinite_counts": nonfinite_counts,
            "nonpositive_counts": nonpositive_counts,
            "symbol_counts": symbol_counts,
            "side_counts": side_counts,
            "price": _numeric_summary(price),
            "volume": _numeric_summary(volume),
            "ask": _numeric_summary(ask),
            "ask_minus_price": {
                **_numeric_summary(ask_minus_price),
                "nonpositive_count": nonpositive_ask_minus_price,
                "distinct_rounded_8dp": int(ask_minus_price_values.size),
                "change_count": ask_minus_price_changes,
                "maximum_adjacent_jump": max_ask_minus_price_jump,
                "tenfold_scale_separation": ask_minus_price_scale_separation,
                "value_counts": ask_minus_price_summary,
            },
            "gaps_over_one_second": gaps_over_one_second,
            "gaps_over_warning_threshold": gaps_over_warning,
            "maximum_internal_gap_seconds": max_gap_seconds,
            "nominal_endpoint_span": nominal_endpoint_span,
        }
    )
    return (
        ParquetFileAudit(
            path=str(path),
            size_bytes=path.stat().st_size,
            sha256=digest,
            schema_sha256=schema_sha,
            schema=schema_description,
            schema_valid=True,
            metrics=metrics,
        ),
        findings,
    )


def _aggregate(
    files: list[ParquetFileAudit],
) -> tuple[dict[str, Any], list[AuditFinding]]:
    findings: list[AuditFinding] = []
    valid = [file for file in files if file.schema_valid and file.metrics.get("first_timestamp")]
    schema_hashes = {file.schema_sha256 for file in files}
    schemas_consistent = len(valid) == len(files) and len(schema_hashes) == 1
    findings.append(
        AuditFinding(
            code="schema_consistency",
            status=FindingStatus.PASS if schemas_consistent else FindingStatus.FAIL,
            message=(
                "Every file has the same required schema."
                if schemas_consistent
                else "One or more files have an invalid or inconsistent schema."
            ),
            details={"distinct_schema_hashes": len(schema_hashes)},
        )
    )
    total_rows = sum(int(file.metrics["row_count"]) for file in files)
    symbol_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    for file in valid:
        symbol_counts.update(file.metrics["symbol_counts"])
        side_counts.update(file.metrics["side_counts"])

    ordered = sorted(valid, key=_file_time_sort_key)
    cross_file_gaps: list[dict[str, Any]] = []
    overlap = False
    for previous, current in zip(ordered, ordered[1:], strict=False):
        previous_end = _parse_timestamp(previous.metrics["last_timestamp"])
        current_start = _parse_timestamp(current.metrics["first_timestamp"])
        gap = (current_start - previous_end).total_seconds()
        # Timestamp endpoints are observed values, so equality means both shards
        # contain the same boundary instant. Without global grouping we cannot
        # claim cross-file duplicate totals are exact in that case.
        overlap = overlap or gap <= 0.0
        cross_file_gaps.append(
            {
                "from_file": Path(previous.path).name,
                "to_file": Path(current.path).name,
                "seconds": gap,
            }
        )
    if overlap:
        findings.append(
            AuditFinding(
                code="cross_file_overlap",
                status=FindingStatus.FAIL,
                message="File time ranges overlap; aggregate duplicate sums are within-file only.",
            )
        )
    else:
        findings.append(
            AuditFinding(
                code="cross_file_overlap",
                status=FindingStatus.PASS,
                message="No decoded file time ranges overlap.",
            )
        )

    maximum_cross_gap = max(
        (item["seconds"] for item in cross_file_gaps if item["seconds"] >= 0.0),
        default=0.0,
    )
    if maximum_cross_gap > 0.0:
        findings.append(
            AuditFinding(
                code="cross_file_gaps",
                status=FindingStatus.WARNING,
                message="Positive gaps exist between adjacent file time ranges.",
                details={"maximum_seconds": maximum_cross_gap},
            )
        )
    else:
        findings.append(
            AuditFinding(
                code="cross_file_gaps",
                status=FindingStatus.PASS,
                message="Adjacent file time ranges have no positive gap.",
            )
        )

    nominal_seconds = sum(
        float(file.metrics["nominal_endpoint_span"].get("nominal_seconds") or 0.0)
        for file in valid
    )
    endpoint_span_seconds = sum(
        float(
            file.metrics["nominal_endpoint_span"].get(
                "endpoint_span_overlap_seconds"
            )
            or 0.0
        )
        for file in valid
    )
    all_endpoints_span_nominal_windows = bool(valid) and len(valid) == len(files) and all(
        bool(
            file.metrics["nominal_endpoint_span"].get(
                "endpoints_span_nominal_window"
            )
        )
        for file in valid
    )
    min_timestamp = min(
        (file.metrics["first_timestamp"] for file in valid), default=None
    )
    max_timestamp = max(
        (file.metrics["last_timestamp"] for file in valid), default=None
    )
    aggregate_ask_minus_price = _aggregate_range(valid, "ask_minus_price")
    aggregate_ask_minus_price_min = aggregate_ask_minus_price["min"]
    aggregate_ask_minus_price_max = aggregate_ask_minus_price["max"]
    aggregate_ask_minus_price_scale_separation = bool(
        aggregate_ask_minus_price_min is not None
        and aggregate_ask_minus_price_min > 0.0
        and aggregate_ask_minus_price_max is not None
        and aggregate_ask_minus_price_max / aggregate_ask_minus_price_min >= 10.0
    )
    aggregate_ask_minus_price["tenfold_scale_separation"] = (
        aggregate_ask_minus_price_scale_separation
    )
    findings.append(
        AuditFinding(
            code="aggregate_ask_minus_price_scale_separation",
            status=(
                FindingStatus.WARNING
                if aggregate_ask_minus_price_scale_separation
                else FindingStatus.PASS
            ),
            message=(
                "Corpus-level ask-minus-price values occupy scales separated by at least tenfold."
                if aggregate_ask_minus_price_scale_separation
                else "Corpus-level ask-minus-price values have no tenfold scale separation."
            ),
            details=aggregate_ask_minus_price,
        )
    )

    aggregate: dict[str, Any] = {
        "file_count": len(files),
        "schema_valid_file_count": len(valid),
        "total_rows": total_rows,
        "first_timestamp": min_timestamp,
        "last_timestamp": max_timestamp,
        "corpus_span_seconds": _duration_seconds(min_timestamp, max_timestamp),
        "within_file_unique_timestamp_total": sum(
            int(file.metrics["unique_timestamp_count"]) for file in valid
        ),
        "within_file_timestamp_collision_surplus": sum(
            int(file.metrics["timestamp_collision_surplus"]) for file in valid
        ),
        "within_file_unique_complete_row_total": sum(
            int(file.metrics["unique_complete_row_count"]) for file in valid
        ),
        "within_file_exact_duplicate_surplus": sum(
            int(file.metrics["exact_duplicate_surplus"]) for file in valid
        ),
        "aggregate_duplicate_counts_globally_exact": not overlap,
        "symbol_counts": dict(sorted(symbol_counts.items())),
        "side_counts": dict(sorted(side_counts.items())),
        "price": _aggregate_range(valid, "price"),
        "volume": _aggregate_range(valid, "volume"),
        "ask": _aggregate_range(valid, "ask"),
        "ask_minus_price": aggregate_ask_minus_price,
        "maximum_internal_gap_seconds": max(
            (
                float(file.metrics["maximum_internal_gap_seconds"])
                for file in valid
            ),
            default=0.0,
        ),
        "cross_file_gaps": cross_file_gaps,
        "maximum_cross_file_gap_seconds": maximum_cross_gap,
        "summed_nominal_window_seconds": nominal_seconds or None,
        "summed_nominal_endpoint_span_seconds": endpoint_span_seconds or None,
        "nominal_endpoint_span_ratio": (
            endpoint_span_seconds / nominal_seconds if nominal_seconds > 0.0 else None
        ),
        "all_endpoints_span_nominal_windows": all_endpoints_span_nominal_windows,
    }
    return aggregate, findings


def _training_blockers(
    files: list[ParquetFileAudit], aggregate: dict[str, Any]
) -> tuple[str, ...]:
    blockers = {
        "availability_and_provenance_missing",
        "sequence_and_source_epoch_missing",
        "aggressor_side_semantics_missing",
        "bid_and_ask_sizes_missing",
        "l2_depth_missing",
        "trade_event_semantics_missing",
        "capture_completeness_unproven",
    }
    if not bool(aggregate.get("all_endpoints_span_nominal_windows")):
        blockers.add("nominal_endpoint_span_incomplete")
    if any(not file.schema_valid for file in files):
        blockers.add("schema_mismatch")
    if any(not bool(file.metrics.get("sorted_non_decreasing", False)) for file in files):
        blockers.add("timestamp_order_invalid")
    return tuple(sorted(blockers))


def _source_limitation_findings(blockers: tuple[str, ...]) -> list[AuditFinding]:
    messages = {
        "availability_and_provenance_missing": (
            "Event/receive/availability provenance is required for causal features."
        ),
        "sequence_and_source_epoch_missing": (
            "Sequence and source-epoch identity are required for continuity and replay policy."
        ),
        "aggressor_side_semantics_missing": (
            "A meaningful, documented aggressor side is required for buy/sell footprint labels."
        ),
        "bid_and_ask_sizes_missing": (
            "Independent bid and ask sizes are required for the selected L1 contract."
        ),
        "l2_depth_missing": "Two-sided ranked depth is required for the selected L2 contract.",
        "trade_event_semantics_missing": (
            "An execution/event contract is required before price-volume rows can be tape labels."
        ),
        "capture_completeness_unproven": (
            "Capture completeness cannot be proven from timestamps alone."
        ),
        "nominal_endpoint_span_incomplete": (
            "One or more file endpoint spans do not reach their nominal window boundaries."
        ),
        "schema_mismatch": "One or more files do not match the required source schema.",
        "timestamp_order_invalid": "One or more files contain invalid timestamp order.",
    }
    return [
        AuditFinding(
            code=f"training_blocker.{blocker}",
            status=FindingStatus.FAIL,
            message=messages[blocker],
        )
        for blocker in blockers
    ]


def _schema_description(schema: Any) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    )


def _schema_signature(schema: Any) -> tuple[tuple[str, str, bool], ...]:
    return tuple(
        (field.name, str(field.type), bool(field.nullable)) for field in schema
    )


def _schema_sha256(schema: tuple[dict[str, Any], ...]) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_manifest_sha256(files: list[ParquetFileAudit]) -> str:
    manifest = [
        {
            # Content identity must survive relocation of an immutable corpus.
            "file_name": Path(file.path).name,
            "size_bytes": file.size_bytes,
            "sha256": file.sha256,
            "schema_sha256": file.schema_sha256,
        }
        for file in sorted(
            files,
            key=lambda item: (
                Path(item.path).name.casefold(),
                item.sha256,
                item.schema_sha256,
                item.size_bytes,
            ),
        )
    ]
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()


def _numeric_numpy(column: Any, *, pc: Any) -> np.ndarray:
    filled = pc.fill_null(column, float("nan")) if column.null_count else column
    return np.asarray(filled.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.float64)


def _integer_numpy(column: Any, *, pc: Any) -> np.ndarray:
    filled = pc.fill_null(column, 0) if column.null_count else column
    return np.asarray(filled.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)


def _timestamp_numpy(column: Any, *, pa: Any, pc: Any) -> np.ndarray:
    if column.null_count:
        return np.array([], dtype=np.int64)
    cast = pc.cast(column, pa.int64())
    return np.asarray(cast.combine_chunks().to_numpy(zero_copy_only=False), dtype=np.int64)


def _value_counts(table: Any, column: str) -> dict[str, int]:
    grouped = table.group_by([column]).aggregate([(column, "count")])
    values = grouped[column].to_pylist()
    counts = grouped[f"{column}_count"].to_pylist()
    rendered = (
        ("<null>" if value is None else str(value), int(count))
        for value, count in zip(values, counts, strict=True)
    )
    return dict(sorted(rendered))


def _numeric_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": None, "median": None, "p99": None, "max": None}
    return {
        "min": float(finite.min()),
        "median": float(np.quantile(finite, 0.5)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(finite.max()),
    }


def _finite_min(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.min()) if finite.size else None


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else None


def _ask_minus_price_value_summary(
    values: np.ndarray, counts: np.ndarray, *, limit: int = 128
) -> list[dict[str, float | int]]:
    pairs = [(float(value), int(count)) for value, count in zip(values, counts, strict=True)]
    if len(pairs) > limit:
        pairs = sorted(pairs, key=lambda item: (-item[1], item[0]))[:limit]
    else:
        pairs.sort(key=lambda item: item[0])
    return [{"ask_minus_price": value, "count": count} for value, count in pairs]


def _timestamp_iso(column: Any, *, pc: Any, minimum: bool) -> str | None:
    scalar = pc.min(column) if minimum else pc.max(column)
    value = scalar.as_py()
    return value.isoformat() if value is not None else None


def _duration_seconds(first: str | None, last: str | None) -> float | None:
    if first is None or last is None:
        return None
    return (_parse_timestamp(last) - _parse_timestamp(first)).total_seconds()


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp is not timezone-aware: {value}")
    return parsed.astimezone(UTC)


def _nominal_endpoint_span(
    file_name: str,
    *,
    first_timestamp: str | None,
    last_timestamp: str | None,
    boundary_tolerance_seconds: float,
) -> dict[str, Any]:
    match = _NOMINAL_WINDOW.search(file_name)
    if match is None or first_timestamp is None or last_timestamp is None:
        return {
            "parsed": False,
            "nominal_start": None,
            "nominal_end_exclusive": None,
            "nominal_seconds": None,
            "endpoint_span_overlap_seconds": None,
            "endpoint_span_ratio": None,
            "endpoints_span_nominal_window": False,
        }
    month = _MONTHS.get(match.group("month").casefold())
    start_day = int(match.group("start"))
    end_day = int(match.group("end"))
    first = _parse_timestamp(first_timestamp)
    last = _parse_timestamp(last_timestamp)
    if month is None or end_day < start_day:
        return {
            "parsed": False,
            "nominal_start": None,
            "nominal_end_exclusive": None,
            "nominal_seconds": None,
            "endpoint_span_overlap_seconds": None,
            "endpoint_span_ratio": None,
            "endpoints_span_nominal_window": False,
        }
    nominal_start = datetime(first.year, month, start_day, tzinfo=UTC)
    nominal_end = datetime(first.year, month, end_day, tzinfo=UTC) + timedelta(days=1)
    nominal_seconds = (nominal_end - nominal_start).total_seconds()
    overlap_start = max(first, nominal_start)
    overlap_end = min(last, nominal_end)
    overlap_seconds = max(0.0, (overlap_end - overlap_start).total_seconds())
    start_lag = (first - nominal_start).total_seconds()
    end_shortfall = (nominal_end - last).total_seconds()
    within_window = first >= nominal_start and last < nominal_end
    full = (
        within_window
        and start_lag <= boundary_tolerance_seconds
        and end_shortfall <= boundary_tolerance_seconds
    )
    return {
        "parsed": True,
        "nominal_start": nominal_start.isoformat(),
        "nominal_end_exclusive": nominal_end.isoformat(),
        "nominal_seconds": nominal_seconds,
        "endpoint_span_overlap_seconds": overlap_seconds,
        "endpoint_span_ratio": overlap_seconds / nominal_seconds,
        "start_lag_seconds": start_lag,
        "end_shortfall_seconds": end_shortfall,
        "boundary_tolerance_seconds": boundary_tolerance_seconds,
        "endpoints_span_nominal_window": full,
    }


def _aggregate_range(files: list[ParquetFileAudit], key: str) -> dict[str, float | None]:
    minima = [
        float(file.metrics[key]["min"])
        for file in files
        if file.metrics[key]["min"] is not None
    ]
    maxima = [
        float(file.metrics[key]["max"])
        for file in files
        if file.metrics[key]["max"] is not None
    ]
    return {
        "min": min(minima) if minima else None,
        "max": max(maxima) if maxima else None,
    }


def _file_time_sort_key(file: ParquetFileAudit) -> tuple[str, str]:
    timestamp = file.metrics.get("first_timestamp") or "9999-12-31T23:59:59+00:00"
    return str(timestamp), file.path.casefold()


def _empty_metrics() -> dict[str, Any]:
    return {
        "first_timestamp": None,
        "last_timestamp": None,
        "duration_seconds": None,
        "sorted_non_decreasing": False,
        "null_counts": {},
        "nonfinite_counts": {},
        "nonpositive_counts": {},
        "symbol_counts": {},
        "side_counts": {},
        "nominal_endpoint_span": {
            "parsed": False,
            "endpoints_span_nominal_window": False,
        },
    }
