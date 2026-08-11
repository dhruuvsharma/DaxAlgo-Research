from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from daxalgo_numeric_transformer.chmt_q.data import (  # noqa: E402
    FindingStatus,
    audit_lse_tick_parquet,
)

SOURCE_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("volume", pa.float64(), nullable=False),
        pa.field("side", pa.uint8(), nullable=False),
        pa.field("ask", pa.float64(), nullable=True),
    ]
)


def _row(
    timestamp: datetime,
    *,
    price: float = 100.0,
    volume: float = 0.25,
    side: int = 0,
    ask: float | None = None,
) -> dict[str, object]:
    return {
        "ts": timestamp,
        "symbol": "BTC/USD",
        "price": price,
        "volume": volume,
        "side": side,
        "ask": price + 0.01 if ask is None else ask,
    }


def _write(
    path: Path,
    rows: list[dict[str, object]],
    *,
    schema: object = SOURCE_SCHEMA,
) -> Path:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, path)
    return path


def _finding(report, code: str, file: str | None = None):
    matches = [
        finding
        for finding in report.findings
        if finding.code == code and (file is None or finding.file == file)
    ]
    assert len(matches) == 1
    return matches[0]


def test_audit_is_serializable_exact_and_fail_closed(tmp_path: Path) -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    end = datetime(2026, 7, 3, tzinfo=UTC)
    first = _row(start)
    path = _write(
        tmp_path / "crypto_BTC_USD_tick_1July_2July.parquet",
        [
            first,
            dict(first),
            _row(start + timedelta(seconds=2), price=100.01),
            _row(end - timedelta(seconds=1), price=100.02),
        ],
    )

    report = audit_lse_tick_parquet(path)
    file = report.files[0]

    assert file.schema_valid
    assert file.metrics["row_count"] == 4
    assert file.metrics["sorted_non_decreasing"] is True
    assert file.metrics["unique_timestamp_count"] == 3
    assert file.metrics["timestamp_collision_surplus"] == 1
    assert file.metrics["exact_duplicate_surplus"] == 1
    assert file.metrics["maximum_exact_duplicate_multiplicity"] == 2
    assert file.metrics["side_counts"] == {"0": 4}
    assert (
        file.metrics["nominal_endpoint_span"]["endpoints_span_nominal_window"]
        is True
    )
    assert file.metrics["nominal_endpoint_span"]["endpoint_span_ratio"] < 1.0
    assert file.sha256 == sha256(path.read_bytes()).hexdigest()

    assert _finding(report, "schema_match", path.name).status is FindingStatus.PASS
    assert _finding(report, "exact_duplicates", path.name).status is FindingStatus.WARNING
    assert (
        _finding(report, "aggressor_side_distribution", path.name).status
        is FindingStatus.FAIL
    )
    assert report.aggregate["within_file_exact_duplicate_surplus"] == 1
    assert "aggressor_side_semantics_missing" in report.training_blockers
    assert report.chmt_q_training_ready is False

    payload = report.to_dict()
    assert json.loads(report.to_json(indent=None)) == payload
    assert report.to_json(indent=None) == report.to_json(indent=None)
    assert set(payload["aggregate"]["finding_counts"]) == {"PASS", "WARNING", "FAIL"}
    assert "nominal_endpoint_span_ratio" in payload["aggregate"]
    assert "nominal_coverage_ratio" not in payload["aggregate"]


def test_bad_order_and_large_gap_are_reported_without_reordering(tmp_path: Path) -> None:
    start = datetime(2026, 7, 3, tzinfo=UTC)
    path = _write(
        tmp_path / "crypto_BTC_USD_tick_3July_4July.parquet",
        [
            _row(start + timedelta(seconds=3), price=101.0),
            _row(start + timedelta(seconds=1), price=100.0),
            _row(start + timedelta(seconds=20), price=102.0),
        ],
    )

    report = audit_lse_tick_parquet(path, internal_gap_warning_seconds=10.0)
    file = report.files[0]

    assert file.metrics["sorted_non_decreasing"] is False
    assert file.metrics["negative_interarrival_count"] == 1
    assert file.metrics["maximum_internal_gap_seconds"] == pytest.approx(19.0)
    assert _finding(report, "timestamp_order", path.name).status is FindingStatus.FAIL
    assert _finding(report, "internal_gaps", path.name).status is FindingStatus.WARNING
    assert "timestamp_order_invalid" in report.training_blockers
    assert report.chmt_q_training_ready is False


def test_schema_mismatch_is_a_structured_failure(tmp_path: Path) -> None:
    wrong_schema = pa.schema(
        [
            pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("price", pa.float64(), nullable=False),
            pa.field("volume", pa.float64(), nullable=False),
            pa.field("side", pa.int64(), nullable=False),
            pa.field("ask", pa.float64(), nullable=True),
        ]
    )
    path = _write(
        tmp_path / "wrong.parquet",
        [_row(datetime(2026, 7, 1, tzinfo=UTC))],
        schema=wrong_schema,
    )

    report = audit_lse_tick_parquet(path)

    assert report.files[0].schema_valid is False
    assert _finding(report, "schema_mismatch", path.name).status is FindingStatus.FAIL
    assert "schema_mismatch" in report.training_blockers
    assert report.aggregate["schema_valid_file_count"] == 0
    assert report.chmt_q_training_ready is False


def test_ask_minus_price_scales_cross_file_gaps_and_value_failures(
    tmp_path: Path,
) -> None:
    first_start = datetime(2026, 7, 5, tzinfo=UTC)
    first = _write(
        tmp_path / "crypto_BTC_USD_tick_5July_5July.parquet",
        [
            _row(first_start, ask=100.01),
            _row(first_start + timedelta(seconds=1), price=100.0, ask=110.0),
        ],
    )
    second_start = datetime(2026, 7, 6, tzinfo=UTC) + timedelta(seconds=30)
    second = _write(
        tmp_path / "crypto_BTC_USD_tick_6July_6July.parquet",
        [
            _row(second_start, price=float("nan"), ask=101.0),
            _row(second_start + timedelta(seconds=1), price=101.0, volume=0.0),
        ],
    )

    report = audit_lse_tick_parquet([second, first])
    first_audit = next(file for file in report.files if Path(file.path).name == first.name)
    second_audit = next(file for file in report.files if Path(file.path).name == second.name)

    assert first_audit.metrics["ask_minus_price"]["tenfold_scale_separation"] is True
    assert (
        _finding(report, "ask_minus_price_scale_separation", first.name).status
        is FindingStatus.WARNING
    )
    assert report.aggregate["ask_minus_price"]["tenfold_scale_separation"] is True
    assert (
        _finding(report, "aggregate_ask_minus_price_scale_separation").status
        is FindingStatus.WARNING
    )
    assert second_audit.metrics["nonfinite_counts"]["price"] == 1
    assert second_audit.metrics["nonpositive_counts"]["volume"] == 1
    assert _finding(report, "value_integrity", second.name).status is FindingStatus.FAIL
    assert report.aggregate["cross_file_gaps"][0]["seconds"] == pytest.approx(86_429.0)
    assert _finding(report, "cross_file_gaps").status is FindingStatus.WARNING
    assert report.chmt_q_training_ready is False


def test_source_manifest_identity_survives_corpus_relocation(tmp_path: Path) -> None:
    source_directory = tmp_path / "source"
    relocated_directory = tmp_path / "relocated"
    source_directory.mkdir()
    relocated_directory.mkdir()
    path = _write(
        source_directory / "crypto_BTC_USD_tick_1July_1July.parquet",
        [_row(datetime(2026, 7, 1, tzinfo=UTC))],
    )
    relocated = relocated_directory / path.name
    shutil.copy2(path, relocated)

    original_report = audit_lse_tick_parquet(path)
    relocated_report = audit_lse_tick_parquet(relocated)

    assert original_report.files[0].path != relocated_report.files[0].path
    assert (
        original_report.aggregate["source_manifest_sha256"]
        == relocated_report.aggregate["source_manifest_sha256"]
    )


def test_equal_adjacent_shard_endpoints_fail_closed_for_global_duplicates(
    tmp_path: Path,
) -> None:
    start = datetime(2026, 7, 1, tzinfo=UTC)
    boundary = start + timedelta(seconds=1)
    shared = _row(boundary, price=100.01)
    first = _write(
        tmp_path / "crypto_BTC_USD_tick_1July_1July_part1.parquet",
        [_row(start), shared],
    )
    second = _write(
        tmp_path / "crypto_BTC_USD_tick_1July_1July_part2.parquet",
        [dict(shared), _row(boundary + timedelta(seconds=1), price=100.02)],
    )

    report = audit_lse_tick_parquet([first, second])

    assert _finding(report, "cross_file_overlap").status is FindingStatus.FAIL
    assert report.aggregate["aggregate_duplicate_counts_globally_exact"] is False
