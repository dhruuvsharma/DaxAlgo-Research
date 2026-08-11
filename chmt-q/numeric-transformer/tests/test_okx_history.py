from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import zipfile
from datetime import date
from pathlib import Path

import pytest

from daxalgo_numeric_transformer.okx_history import (
    MANIFEST_NAME,
    SCHEMA_VERSION,
    audit_acquisition,
    audit_l2_archive,
    audit_trade_archive,
    build_artifacts,
)


def test_build_artifacts_uses_complete_inclusive_daily_contract() -> None:
    artifacts = build_artifacts(
        "BTC-USDT",
        date(2026, 6, 1),
        date(2026, 6, 2),
    )

    assert [(item.day, item.feed) for item in artifacts] == [
        ("2026-05-31", "l2-400"),
        ("2026-06-01", "l2-400"),
        ("2026-06-01", "trades"),
        ("2026-06-02", "l2-400"),
        ("2026-06-02", "trades"),
        ("2026-06-03", "trades"),
    ]
    assert artifacts[1].relative_path == (
        "l2-400/BTC-USDT-L2orderbook-400lv-2026-06-01.tar.gz"
    )
    assert "/daily/20260601/" in artifacts[1].url
    assert artifacts[2].url.endswith("BTC-USDT-trades-2026-06-01.zip?v=999")


def test_trade_audit_requires_genuine_sided_monotone_tape(tmp_path: Path) -> None:
    path = tmp_path / "BTC-USDT-trades-2026-06-01.zip"
    member = "BTC-USDT-trades-2026-06-01.csv"
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer)
    writer.writerow(["instrument_name", "trade_id", "side", "price", "size", "created_time"])
    writer.writerow(["BTC-USDT", "10", "buy", "100.0", "0.25", "1780243200000"])
    writer.writerow(["BTC-USDT", "11", "sell", "99.9", "0.50", "1780243200500"])
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, buffer.getvalue())

    report = audit_trade_archive(
        path,
        instrument="BTC-USDT",
        expected_day=date(2026, 6, 1),
    )

    assert report["rows"] == 2
    assert report["sides"] == ["buy", "sell"]
    assert report["trade_id_regressions"] == 0
    assert report["out_of_order_rows"] == 0


def test_l2_audit_streams_full_tar_and_reconstructs_updates(tmp_path: Path) -> None:
    path = tmp_path / "BTC-USDT-L2orderbook-400lv-2026-06-01.tar.gz"
    member_name = "BTC-USDT-L2orderbook-400lv-2026-06-01.data"
    records = [
        {
            "instId": "BTC-USDT",
            "action": "snapshot",
            "ts": "1780272000000",
            "bids": [["99.9", "2", "1"], ["99.8", "3", "1"]],
            "asks": [["100.1", "4", "1"], ["100.2", "5", "1"]],
        },
        {
            "instId": "BTC-USDT",
            "action": "update",
            "ts": "1780272000010",
            "bids": [["99.9", "0", "0"], ["99.95", "1", "1"]],
            "asks": [],
        },
    ]
    payload = b"".join(
        json.dumps(record, separators=(",", ":")).encode("utf-8") + b"\n"
        for record in records
    )
    info = tarfile.TarInfo(member_name)
    info.size = len(payload)
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(payload))

    report = audit_l2_archive(
        path,
        instrument="BTC-USDT",
        expected_day=date(2026, 6, 1),
        parsed_prefix_rows=1,
    )

    assert report["rows"] == 2
    assert report["first_action"] == "snapshot"
    assert report["last_action"] == "update"
    assert report["parsed_prefix_rows"] == 1
    assert report["prefix_best_bid"] == pytest.approx(99.9)
    assert report["prefix_best_ask"] == pytest.approx(100.1)


def test_acquisition_audit_binds_instrument_metadata(tmp_path: Path) -> None:
    instrument = "BTC-USDT"
    day = date(2026, 6, 1)
    trade_path = tmp_path / "trades" / "BTC-USDT-trades-2026-06-01.zip"
    trade_path.parent.mkdir()
    trade_buffer = io.StringIO(newline="")
    writer = csv.writer(trade_buffer)
    writer.writerow(["instrument_name", "trade_id", "side", "price", "size", "created_time"])
    writer.writerow([instrument, "10", "buy", "100", "0.25", "1780272000127"])
    with zipfile.ZipFile(trade_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "BTC-USDT-trades-2026-06-01.csv",
            trade_buffer.getvalue(),
        )

    l2_path = tmp_path / "l2-400" / "BTC-USDT-L2orderbook-400lv-2026-06-01.tar.gz"
    l2_path.parent.mkdir()
    l2_record = {
        "instId": instrument,
        "action": "snapshot",
        "ts": "1780272000004",
        "bids": [["99.9", "2", "1"]],
        "asks": [["100.1", "3", "1"]],
    }
    l2_payload = json.dumps(l2_record, separators=(",", ":")).encode() + b"\n"
    info = tarfile.TarInfo("BTC-USDT-L2orderbook-400lv-2026-06-01.data")
    info.size = len(l2_payload)
    with tarfile.open(l2_path, "w:gz") as archive:
        archive.addfile(info, io.BytesIO(l2_payload))

    metadata_path = tmp_path / "okx-instrument-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "historical_effectivity_proven": False,
                "response": {
                    "data": [
                        {
                            "instId": instrument,
                            "tickSz": "0.1",
                            "lotSz": "0.00000001",
                            "minSz": "0.00001",
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    def binding(path: Path, feed: str, archive_day: str) -> dict[str, object]:
        payload = path.read_bytes()
        return {
            "day": archive_day,
            "feed": feed,
            "relative_path": path.relative_to(tmp_path).as_posix(),
            "content_length": len(payload),
            "bytes": len(payload),
            "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "status": "complete",
        }

    metadata_bytes = metadata_path.read_bytes()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "instrument": instrument,
        "start_date": day.isoformat(),
        "end_date": day.isoformat(),
        "instrument_metadata": {
            "relative_path": metadata_path.name,
            "bytes": len(metadata_bytes),
            "sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        },
        "artifacts": [
            binding(l2_path, "l2-400", day.isoformat()),
            binding(trade_path, "trades", day.isoformat()),
        ],
    }
    (tmp_path / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")

    report = audit_acquisition(tmp_path, parsed_prefix_rows=1)

    assert report["status"] == "pass"
    assert report["instrument_metadata"]["tick_size"] == "0.1"
    assert report["utc_window"]["trade_rows"] == 1

    metadata_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="metadata binding mismatch"):
        audit_acquisition(tmp_path, parsed_prefix_rows=1)
