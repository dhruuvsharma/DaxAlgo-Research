from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from daxalgo_numeric_transformer.chmt_q.data import ArrowDataset
from daxalgo_numeric_transformer.okx_compile import (
    COMPILER_REPORT_NAME,
    _OrderBook,
    _parse_fixed,
    _parse_source_fixed,
    compile_okx_dataset,
)
from daxalgo_numeric_transformer.okx_history import AUDIT_NAME, MANIFEST_NAME


@pytest.mark.parametrize(
    ("value", "decimals", "expected"),
    [
        (b"123.4", 1, 1_234),
        ("0.00000001", 8, 1),
        ("+1e-8", 8, 1),
        ("-1.25e2", 1, -1_250),
        (".1", 1, 1),
        ("1.2300", 2, 123),
    ],
)
def test_parse_fixed_is_exact(value: bytes | str, decimals: int, expected: int) -> None:
    assert _parse_fixed(value, decimals) == expected


@pytest.mark.parametrize("value", ["", "1.2.3", "1e", "0.01", "1e-9"])
def test_parse_fixed_rejects_invalid_or_unrepresentable_values(value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        _parse_fixed(value, 1 if value != "1e-9" else 8)


def test_source_parser_repairs_only_sub_unit_float_text_noise() -> None:
    units, error = _parse_source_fixed("0.00017852999999999999", 8)
    assert units == 17_853
    assert 0.0 < error <= 1.0e-6

    with pytest.raises(ValueError, match="source tolerance"):
        _parse_source_fixed("0.00017852999", 8)


def test_order_book_replaces_snapshot_and_applies_sorted_updates() -> None:
    book = _OrderBook()
    book.apply(
        {
            "action": "snapshot",
            "bids": [["99.9", "2", "1"], ["99.8", "3", "1"]],
            "asks": [["100.1", "4", "1"], ["100.2", "5", "1"]],
        },
        1_000,
    )

    assert book.top("bid") == [(999, 200_000_000), (998, 300_000_000)]
    assert book.top("ask") == [(1_001, 400_000_000), (1_002, 500_000_000)]

    book.apply(
        {
            "action": "update",
            "bids": [["99.9", "0", "0"], ["99.8", "1.5", "1"], ["99.7", "2", "1"]],
            "asks": [["100.1", "6", "1"], ["100.3", "2", "1"]],
        },
        1_500,
    )

    assert book.top("bid") == [(998, 150_000_000), (997, 200_000_000)]
    assert book.top("ask") == [
        (1_001, 600_000_000),
        (1_002, 500_000_000),
        (1_003, 200_000_000),
    ]
    assert book.last_event_ms == 1_500


def test_compile_synthetic_okx_archives_to_non_promotable_arrow(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    source = tmp_path / "source"
    source.mkdir()
    analysis_day = date(2026, 6, 1)
    day_start_ms = int(
        datetime(analysis_day.year, analysis_day.month, analysis_day.day, tzinfo=UTC).timestamp()
        * 1_000
    )

    trade_path = source / "trades" / "BTC-USDT-trades-2026-06-01.zip"
    trade_path.parent.mkdir()
    trade_buffer = io.StringIO(newline="")
    writer = csv.writer(trade_buffer)
    writer.writerow(["instrument_name", "trade_id", "side", "price", "size", "created_time"])
    for minute in range(10):
        writer.writerow(
            [
                "BTC-USDT",
                str(10_000 + minute),
                "buy" if minute % 2 == 0 else "sell",
                f"{100 + (minute % 3) / 10:.1f}",
                "0.00000001",
                str(day_start_ms + minute * 60_000 + 250),
            ]
        )
    with zipfile.ZipFile(trade_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("BTC-USDT-trades-2026-06-01.csv", trade_buffer.getvalue())

    spill_day = date(2026, 6, 2)
    spill_trade_path = source / "trades" / "BTC-USDT-trades-2026-06-02.zip"
    spill_buffer = io.StringIO(newline="")
    spill_writer = csv.writer(spill_buffer)
    spill_writer.writerow(
        ["instrument_name", "trade_id", "side", "price", "size", "created_time"]
    )
    spill_writer.writerow(
        ["BTC-USDT", "10010", "buy", "100.0", "0.00000001", str(day_start_ms + 16 * 3_600_000)]
    )
    with zipfile.ZipFile(
        spill_trade_path, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        archive.writestr("BTC-USDT-trades-2026-06-02.csv", spill_buffer.getvalue())

    l2_path = source / "l2-400" / "BTC-USDT-L2orderbook-400lv-2026-06-01.tar.gz"
    l2_path.parent.mkdir()
    l2_records = [
        {
            "instId": "BTC-USDT",
            "action": "snapshot" if second == 0 else "update",
            "ts": str(day_start_ms + second * 1_000),
            "bids": [["99.9", "1", "1"]] if second == 0 else [],
            "asks": [["100.1", "1", "1"]] if second == 0 else [],
        }
        for second in range(5 * 60 + 1)
    ]
    l2_payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode("ascii") + b"\n"
        for record in l2_records
    )
    member = tarfile.TarInfo("BTC-USDT-L2orderbook-400lv-2026-06-01.data")
    member.size = len(l2_payload)
    with tarfile.open(l2_path, "w:gz") as archive:
        archive.addfile(member, io.BytesIO(l2_payload))

    manifest = {
        "schema_version": 1,
        "instrument": "BTC-USDT",
        "start_date": analysis_day.isoformat(),
        "end_date": analysis_day.isoformat(),
        "artifacts": [
            _artifact_binding(source, l2_path, "l2-400", analysis_day),
            _artifact_binding(source, trade_path, "trades", analysis_day),
            _artifact_binding(source, spill_trade_path, "trades", spill_day),
        ],
    }
    manifest_path = source / MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    (source / AUDIT_NAME).write_text(
        json.dumps(
            {
                "status": "pass",
                "source_manifest_sha256": manifest_sha256,
                "instrument_metadata": {
                    "tick_size": "0.1",
                    "lot_size": "0.00000001",
                    "minimum_size": "0.00001",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    destination = tmp_path / "compiled"
    compiled, report = compile_okx_dataset(
        source_root=source,
        destination=destination,
        start=analysis_day,
        end=analysis_day,
        rows_per_shard=1,
        rows_per_record_batch=1,
        compression="zstd",
        max_examples=1,
    )

    assert compiled.row_count == 1
    assert compiled.instrument_id == "BTC-USDT"
    assert compiled.source_manifest_sha256 == manifest_sha256
    assert not compiled.capabilities.training_ready
    assert not compiled.capabilities.rights_confirmed
    assert not compiled.capabilities.provenance_complete
    assert report["status"] == "compiled_non_promotable"
    assert report["dataset_rows"] == 1
    assert (destination / COMPILER_REPORT_NAME).is_file()
    assert (destination / compiled.shards[0].path).is_file()

    inspected = ArrowDataset.open(
        destination,
        expected_dataset_id=compiled.dataset_id,
        expected_instrument_id="BTC-USDT",
        require_training_capabilities=False,
    )
    assert inspected.example_ids == ("okx-btc-usdt-20260601T000500Z",)
    with pytest.raises(PermissionError, match="training_ready"):
        ArrowDataset.open(destination, expected_dataset_id=compiled.dataset_id)


def _artifact_binding(root: Path, path: Path, feed: str, day: date) -> dict[str, object]:
    content = path.read_bytes()
    return {
        "day": day.isoformat(),
        "feed": feed,
        "relative_path": path.relative_to(root).as_posix(),
        "content_length": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "status": "complete",
    }
