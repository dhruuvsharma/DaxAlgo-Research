"""Acquire and audit immutable OKX historical trade and L2 archives.

The downloader uses only the Python standard library, preserves compressed source
files, resumes partial transfers, and writes a hash-bearing source manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO

PORTAL_URL = "https://www.okx.com/historical-data"
MANIFEST_NAME = "okx-source-manifest.json"
AUDIT_NAME = "okx-source-audit.json"
INSTRUMENT_METADATA_NAME = "okx-instrument-metadata.json"
SCHEMA_VERSION = 1
DEFAULT_USER_AGENT = "DaxAlgo-OKX-Historical-Acquirer/1.0"
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
TRADE_COLUMNS = (
    "instrument_name",
    "trade_id",
    "side",
    "price",
    "size",
    "created_time",
)


@dataclass(frozen=True, order=True)
class ArtifactSpec:
    """One immutable daily OKX archive."""

    day: str
    feed: str
    relative_path: str
    url: str


@dataclass(frozen=True)
class RemoteMetadata:
    """HTTP identity and size observed before acquisition."""

    content_length: int
    content_type: str | None
    accept_ranges: str | None
    etag: str | None
    last_modified: str | None


@dataclass(frozen=True)
class LocalArtifact:
    """Verified local result for one archive."""

    bytes: int
    md5: str
    sha256: str
    status: str = "complete"


def build_artifacts(
    instrument: str,
    start: date,
    end: date,
    feeds: Iterable[str] = ("l2-400", "trades"),
) -> tuple[ArtifactSpec, ...]:
    """Build deterministic official daily archive URLs for an inclusive date range."""

    if not re.fullmatch(r"[A-Z0-9]+-[A-Z0-9]+", instrument):
        raise ValueError("instrument must be an uppercase OKX spot instrument such as BTC-USDT")
    if end < start:
        raise ValueError("end must be on or after start")
    selected = tuple(dict.fromkeys(feeds))
    unknown = set(selected) - {"l2-400", "trades"}
    if unknown:
        raise ValueError(f"unsupported feeds: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("at least one feed is required")

    artifacts: list[ArtifactSpec] = []
    current = start - timedelta(days=1) if "l2-400" in selected else start
    final_archive_day = end + timedelta(days=1) if "trades" in selected else end
    while current <= final_archive_day:
        iso = current.isoformat()
        compact = current.strftime("%Y%m%d")
        for feed in selected:
            if feed == "l2-400" and current > end:
                continue
            if feed == "trades" and current < start:
                continue
            if feed == "l2-400":
                filename = f"{instrument}-L2orderbook-400lv-{iso}.tar.gz"
                url = (
                    "https://static.okx.com/cdn/okx/match/orderbook/L2/400lv/daily/"
                    f"{compact}/{filename}"
                )
            else:
                filename = f"{instrument}-trades-{iso}.zip"
                url = (
                    "https://static.okx.com/cdn/okex/traderecords/trades/daily/"
                    f"{compact}/{filename}?v=999"
                )
            artifacts.append(
                ArtifactSpec(
                    day=iso,
                    feed=feed,
                    relative_path=f"{feed}/{filename}",
                    url=url,
                )
            )
        current += timedelta(days=1)
    return tuple(sorted(artifacts))


def probe_artifact(
    artifact: ArtifactSpec,
    *,
    attempts: int,
    timeout_seconds: float,
) -> RemoteMetadata:
    """Read immutable HTTP metadata without downloading an archive body."""

    request = urllib.request.Request(
        artifact.url,
        method="HEAD",
        headers={"Accept-Encoding": "identity", "User-Agent": DEFAULT_USER_AGENT},
    )
    with _open_with_retry(request, attempts=attempts, timeout_seconds=timeout_seconds) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is None or not content_length.isdigit() or int(content_length) <= 0:
            raise RuntimeError(f"missing or invalid Content-Length for {artifact.url}")
        return RemoteMetadata(
            content_length=int(content_length),
            content_type=response.headers.get("Content-Type"),
            accept_ranges=response.headers.get("Accept-Ranges"),
            etag=response.headers.get("ETag"),
            last_modified=response.headers.get("Last-Modified"),
        )


def fetch_instrument_metadata(
    instrument: str,
    *,
    attempts: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Capture the official current instrument contract without claiming historic effectivity."""

    query = urllib.parse.urlencode({"instType": "SPOT", "instId": instrument})
    url = f"https://www.okx.com/api/v5/public/instruments?{query}"
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": DEFAULT_USER_AGENT},
    )
    with _open_with_retry(request, attempts=attempts, timeout_seconds=timeout_seconds) as response:
        try:
            payload = json.load(response)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("OKX instrument metadata was not valid JSON") from error
    if not isinstance(payload, dict) or str(payload.get("code")) != "0":
        raise RuntimeError("OKX instrument metadata request failed")
    matches = [
        item
        for item in payload.get("data", [])
        if isinstance(item, dict) and item.get("instId") == instrument
    ]
    if len(matches) != 1:
        raise RuntimeError(f"OKX returned {len(matches)} metadata rows for {instrument}")
    required = {"baseCcy", "quoteCcy", "tickSz", "lotSz", "minSz", "state"}
    missing = sorted(name for name in required if not matches[0].get(name))
    if missing:
        raise RuntimeError(f"OKX instrument metadata omitted: {', '.join(missing)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "OKX public instruments API",
        "request_url": url,
        "retrieved_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "historical_effectivity_proven": False,
        "response": payload,
    }


def acquire_artifact(
    root: Path,
    artifact: ArtifactSpec,
    remote: RemoteMetadata,
    *,
    attempts: int,
    timeout_seconds: float,
) -> LocalArtifact:
    """Download one archive with safe resume and verify its size and strong hashes."""

    target = root / Path(artifact.relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(f"{target.name}.part")

    if target.exists():
        if target.is_file() and target.stat().st_size == remote.content_length:
            result = _hash_file(target)
            _verify_etag(artifact, remote, result)
            return result
        _quarantine(target)
    invalid_partial = partial.exists() and (
        not partial.is_file() or partial.stat().st_size > remote.content_length
    )
    if invalid_partial:
        _quarantine(partial)

    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"Accept-Encoding": "identity", "User-Agent": DEFAULT_USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(artifact.url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", response.getcode())
                append = offset > 0 and status == 206
                if append:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-"):
                        raise RuntimeError(
                            f"invalid Content-Range while resuming {artifact.relative_path}: "
                            f"{content_range!r}"
                        )
                elif status != 200:
                    raise RuntimeError(
                        f"unexpected HTTP {status} for {artifact.relative_path}"
                    )
                mode = "ab" if append else "wb"
                with partial.open(mode) as output:
                    _copy_stream(response, output)

            actual = partial.stat().st_size
            if actual == remote.content_length:
                os.replace(partial, target)
                result = _hash_file(target)
                _verify_etag(artifact, remote, result)
                return result
            if actual > remote.content_length:
                _quarantine(partial)
                raise RuntimeError(
                    f"download exceeded Content-Length for {artifact.relative_path}"
                )
            last_error = RuntimeError(
                f"short download for {artifact.relative_path}: "
                f"{actual}/{remote.content_length} bytes"
            )
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in RETRYABLE_HTTP_STATUS:
                raise
        except (
            OSError,
            TimeoutError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            last_error = error

        if attempt < attempts:
            time.sleep(_retry_delay(attempt))
    raise RuntimeError(f"failed to download {artifact.relative_path}") from last_error


def acquire_month(
    *,
    root: Path,
    instrument: str,
    start: date,
    end: date,
    feeds: Iterable[str] = ("l2-400", "trades"),
    workers: int = 3,
    attempts: int = 8,
    timeout_seconds: float = 60.0,
    manifest_only: bool = False,
) -> dict[str, Any]:
    """Probe, acquire, and manifest all selected daily archives."""

    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if attempts < 1:
        raise ValueError("attempts must be positive")
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts = build_artifacts(instrument, start, end, feeds)
    instrument_snapshot = fetch_instrument_metadata(
        instrument,
        attempts=attempts,
        timeout_seconds=timeout_seconds,
    )
    instrument_path = root / INSTRUMENT_METADATA_NAME
    _write_json_atomic(instrument_path, instrument_snapshot)
    instrument_local = _hash_file(instrument_path)
    instrument_record = {
        "relative_path": INSTRUMENT_METADATA_NAME,
        "sha256": instrument_local.sha256,
        "bytes": instrument_local.bytes,
        "retrieved_at_utc": instrument_snapshot["retrieved_at_utc"],
        "historical_effectivity_proven": False,
    }

    remote_by_artifact: dict[ArtifactSpec, RemoteMetadata] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                probe_artifact,
                artifact,
                attempts=attempts,
                timeout_seconds=timeout_seconds,
            ): artifact
            for artifact in artifacts
        }
        for future in as_completed(futures):
            artifact = futures[future]
            remote_by_artifact[artifact] = future.result()

    local_by_artifact: dict[ArtifactSpec, LocalArtifact] = {}
    manifest = _build_manifest(
        root=root,
        instrument=instrument,
        start=start,
        end=end,
        artifacts=artifacts,
        remote_by_artifact=remote_by_artifact,
        local_by_artifact=local_by_artifact,
        instrument_metadata=instrument_record,
    )
    if manifest_only:
        _write_json_atomic(root / MANIFEST_NAME, manifest)
        return manifest
    if not (root / MANIFEST_NAME).exists():
        _write_json_atomic(root / MANIFEST_NAME, manifest)

    _require_capacity(root, artifacts, remote_by_artifact)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        download_futures: dict[Future[LocalArtifact], ArtifactSpec] = {
            executor.submit(
                acquire_artifact,
                root,
                artifact,
                remote_by_artifact[artifact],
                attempts=attempts,
                timeout_seconds=timeout_seconds,
            ): artifact
            for artifact in artifacts
        }
        try:
            for future in as_completed(download_futures):
                artifact = download_futures[future]
                local_by_artifact[artifact] = future.result()
                manifest = _build_manifest(
                    root=root,
                    instrument=instrument,
                    start=start,
                    end=end,
                    artifacts=artifacts,
                    remote_by_artifact=remote_by_artifact,
                    local_by_artifact=local_by_artifact,
                    instrument_metadata=instrument_record,
                )
                _write_json_atomic(root / MANIFEST_NAME, manifest)
                print(
                    json.dumps(
                        {
                            "event": "downloaded",
                            "feed": artifact.feed,
                            "day": artifact.day,
                            "completed": len(local_by_artifact),
                            "total": len(artifacts),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            manifest = _build_manifest(
                root=root,
                instrument=instrument,
                start=start,
                end=end,
                artifacts=artifacts,
                remote_by_artifact=remote_by_artifact,
                local_by_artifact=local_by_artifact,
                instrument_metadata=instrument_record,
            )
            _write_json_atomic(root / MANIFEST_NAME, manifest)
    return manifest


def audit_trade_archive(
    path: Path,
    *,
    instrument: str,
    expected_day: date,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
) -> dict[str, Any]:
    """Validate a complete OKX daily trade ZIP and its causal label fields."""

    expected_name = f"{instrument}-trades-{expected_day.isoformat()}.csv"
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if names != [expected_name]:
            raise ValueError(f"unexpected ZIP members in {path.name}: {names}")
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"CRC failure in {path.name}: {corrupt_member}")
        with archive.open(expected_name) as raw:
            reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
            if tuple(reader.fieldnames or ()) != TRADE_COLUMNS:
                raise ValueError(f"unexpected trade schema in {path.name}: {reader.fieldnames}")
            count = 0
            first_timestamp: int | None = None
            last_timestamp: int | None = None
            first_trade_id: int | None = None
            last_trade_id: int | None = None
            out_of_order = 0
            trade_id_regressions = 0
            sides: set[str] = set()
            window_rows = 0
            first_window_timestamp: int | None = None
            last_window_timestamp: int | None = None
            for row in reader:
                if row["instrument_name"] != instrument:
                    raise ValueError(f"instrument mismatch in {path.name}")
                timestamp = int(row["created_time"])
                trade_id = int(row["trade_id"])
                price = float(row["price"])
                size = float(row["size"])
                side = row["side"]
                if price <= 0 or size <= 0 or side not in {"buy", "sell"}:
                    raise ValueError(f"invalid trade row in {path.name}")
                if last_timestamp is not None and timestamp < last_timestamp:
                    out_of_order += 1
                if last_trade_id is not None and trade_id <= last_trade_id:
                    trade_id_regressions += 1
                first_timestamp = timestamp if first_timestamp is None else first_timestamp
                first_trade_id = trade_id if first_trade_id is None else first_trade_id
                last_timestamp = timestamp
                last_trade_id = trade_id
                sides.add(side)
                count += 1
                if (
                    window_start_ms is not None
                    and window_end_ms is not None
                    and window_start_ms <= timestamp < window_end_ms
                ):
                    first_window_timestamp = (
                        timestamp if first_window_timestamp is None else first_window_timestamp
                    )
                    last_window_timestamp = timestamp
                    window_rows += 1
    if count == 0:
        raise ValueError(f"empty trade archive: {path.name}")
    _require_trade_partition(first_timestamp, expected_day, path)
    _require_trade_partition(last_timestamp, expected_day, path)
    return {
        "rows": count,
        "first_timestamp_ms": first_timestamp,
        "last_timestamp_ms": last_timestamp,
        "first_trade_id": first_trade_id,
        "last_trade_id": last_trade_id,
        "out_of_order_rows": out_of_order,
        "trade_id_regressions": trade_id_regressions,
        "sides": sorted(sides),
        "utc_window_rows": window_rows,
        "first_utc_window_timestamp_ms": first_window_timestamp,
        "last_utc_window_timestamp_ms": last_window_timestamp,
    }


def audit_l2_archive(
    path: Path,
    *,
    instrument: str,
    expected_day: date,
    parsed_prefix_rows: int = 1_000,
) -> dict[str, Any]:
    """Stream-test an L2 tarball and reconstruct a bounded prefix of its updates."""

    expected_name = f"{instrument}-L2orderbook-400lv-{expected_day.isoformat()}.data"
    member_names: list[str] = []
    line_count = 0
    first_record: dict[str, Any] | None = None
    last_line: bytes | None = None
    bids: dict[str, str] = {}
    asks: dict[str, str] = {}
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_names.append(member.name)
            if member.name != expected_name:
                raise ValueError(f"unexpected tar member in {path.name}: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read tar member in {path.name}")
            while line_count < parsed_prefix_rows:
                raw_line = stream.readline()
                if not raw_line:
                    break
                if not raw_line.strip():
                    raise ValueError(f"blank L2 record in sampled prefix of {path.name}")
                line_count += 1
                last_line = raw_line
                record = _parse_l2_record(raw_line, instrument, path)
                if first_record is None:
                    first_record = record
                    if record["action"] != "snapshot":
                        raise ValueError(f"first L2 record is not a snapshot in {path.name}")
                _apply_book_levels(bids, record["bids"])
                _apply_book_levels(asks, record["asks"])
            line_count, last_line = _stream_remaining_l2_lines(
                stream,
                line_count=line_count,
                last_line=last_line,
            )
    if member_names != [expected_name]:
        raise ValueError(f"unexpected tar members in {path.name}: {member_names}")
    if first_record is None or last_line is None:
        raise ValueError(f"empty L2 archive: {path.name}")
    last_record = _parse_l2_record(last_line, instrument, path)
    first_timestamp = int(first_record["ts"])
    last_timestamp = int(last_record["ts"])
    _require_timestamp_day(first_timestamp, expected_day, path)
    _require_timestamp_day(last_timestamp, expected_day, path)
    if last_timestamp < first_timestamp:
        raise ValueError(f"L2 timestamp regression across {path.name}")
    if not bids or not asks:
        raise ValueError(f"bounded L2 reconstruction is one-sided in {path.name}")
    best_bid = max(float(price) for price in bids)
    best_ask = min(float(price) for price in asks)
    if best_bid >= best_ask:
        raise ValueError(f"bounded L2 reconstruction is crossed in {path.name}")
    return {
        "rows": line_count,
        "parsed_prefix_rows": min(line_count, parsed_prefix_rows),
        "first_timestamp_ms": first_timestamp,
        "last_timestamp_ms": last_timestamp,
        "first_action": first_record["action"],
        "last_action": last_record["action"],
        "prefix_bid_levels": len(bids),
        "prefix_ask_levels": len(asks),
        "prefix_best_bid": best_bid,
        "prefix_best_ask": best_ask,
    }


def audit_acquisition(root: Path, *, parsed_prefix_rows: int = 1_000) -> dict[str, Any]:
    """Verify hashes, full archive decompression, and bounded L2 source semantics."""

    root = root.expanduser().resolve()
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported OKX source manifest schema")
    instrument = str(manifest["instrument"])
    window_start = datetime.combine(
        date.fromisoformat(manifest["start_date"]),
        datetime.min.time(),
        tzinfo=UTC,
    )
    window_end = datetime.combine(
        date.fromisoformat(manifest["end_date"]) + timedelta(days=1),
        datetime.min.time(),
        tzinfo=UTC,
    )
    window_start_ms = int(window_start.timestamp() * 1000)
    window_end_ms = int(window_end.timestamp() * 1000)
    metadata_binding = manifest.get("instrument_metadata")
    if not isinstance(metadata_binding, dict):
        raise ValueError("source manifest has no instrument metadata binding")
    metadata_path = _resolve_manifest_path(root, metadata_binding.get("relative_path"))
    metadata_local = _hash_file(metadata_path)
    if (
        metadata_local.bytes != metadata_binding.get("bytes")
        or metadata_local.sha256 != metadata_binding.get("sha256")
    ):
        raise ValueError("instrument metadata binding mismatch")
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_rows = [
        row
        for row in metadata_payload.get("response", {}).get("data", [])
        if isinstance(row, dict) and row.get("instId") == instrument
    ]
    if len(metadata_rows) != 1:
        raise ValueError("instrument metadata does not contain exactly one matching row")
    reports: list[dict[str, Any]] = []
    for item in manifest["artifacts"]:
        if item.get("status") != "complete":
            raise ValueError(f"incomplete manifest artifact: {item['relative_path']}")
        path = _resolve_manifest_path(root, item["relative_path"])
        if not path.is_file() or path.stat().st_size != int(item["content_length"]):
            raise ValueError(f"missing or wrong-sized artifact: {path}")
        local = _hash_file(path)
        if local.sha256 != item["sha256"] or local.md5 != item["md5"]:
            raise ValueError(f"hash mismatch: {path}")
        expected_day = date.fromisoformat(item["day"])
        if item["feed"] == "trades":
            detail = audit_trade_archive(
                path,
                instrument=instrument,
                expected_day=expected_day,
                window_start_ms=window_start_ms,
                window_end_ms=window_end_ms,
            )
        elif item["feed"] == "l2-400":
            detail = audit_l2_archive(
                path,
                instrument=instrument,
                expected_day=expected_day,
                parsed_prefix_rows=parsed_prefix_rows,
            )
        else:
            raise ValueError(f"unexpected feed in manifest: {item['feed']}")
        reports.append(
            {
                "day": item["day"],
                "feed": item["feed"],
                "relative_path": item["relative_path"],
                "sha256": local.sha256,
                "detail": detail,
            }
        )
        print(
            json.dumps(
                {
                    "event": "audited",
                    "feed": item["feed"],
                    "day": item["day"],
                    "completed": len(reports),
                    "total": len(manifest["artifacts"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    trade_details = [
        item["detail"] for item in reports if item["feed"] == "trades"
    ]
    window_trade_rows = sum(item["utc_window_rows"] for item in trade_details)
    if window_trade_rows == 0:
        raise ValueError("no trades fall inside the requested UTC window")
    first_window_trade = min(
        item["first_utc_window_timestamp_ms"]
        for item in trade_details
        if item["first_utc_window_timestamp_ms"] is not None
    )
    last_window_trade = max(
        item["last_utc_window_timestamp_ms"]
        for item in trade_details
        if item["last_utc_window_timestamp_ms"] is not None
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest_sha256": _hash_file(root / MANIFEST_NAME).sha256,
        "instrument": instrument,
        "start_date": manifest["start_date"],
        "end_date": manifest["end_date"],
        "artifact_count": len(reports),
        "status": "pass",
        "instrument_metadata": {
            "relative_path": str(metadata_path.relative_to(root)),
            "sha256": metadata_local.sha256,
            "historical_effectivity_proven": bool(
                metadata_payload.get("historical_effectivity_proven")
            ),
            "tick_size": metadata_rows[0].get("tickSz"),
            "lot_size": metadata_rows[0].get("lotSz"),
            "minimum_size": metadata_rows[0].get("minSz"),
        },
        "utc_window": {
            "start_inclusive": window_start.isoformat().replace("+00:00", "Z"),
            "end_exclusive": window_end.isoformat().replace("+00:00", "Z"),
            "trade_rows": window_trade_rows,
            "first_trade_timestamp_ms": first_window_trade,
            "last_trade_timestamp_ms": last_window_trade,
        },
        "limitations": [
            "OKX archives expose exchange event time but no independent local receive time.",
            "OKX L2 archive rows do not expose a native sequence identifier.",
            (
                "L2 gzip/tar integrity is fully streamed, but JSON semantics and book "
                "reconstruction are sampled at the bounded prefix plus final row."
            ),
            "The captured instrument metadata is current, not proven effective-dated for June.",
        ],
        "artifacts": reports,
    }
    _write_json_atomic(root / AUDIT_NAME, report)
    return report


def _build_manifest(
    *,
    root: Path,
    instrument: str,
    start: date,
    end: date,
    artifacts: Sequence[ArtifactSpec],
    remote_by_artifact: dict[ArtifactSpec, RemoteMetadata],
    local_by_artifact: dict[ArtifactSpec, LocalArtifact],
    instrument_metadata: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for artifact in artifacts:
        remote = remote_by_artifact[artifact]
        local = local_by_artifact.get(artifact)
        row: dict[str, Any] = {
            **asdict(artifact),
            **asdict(remote),
            "status": "planned",
            "bytes": None,
            "md5": None,
            "sha256": None,
        }
        if local is not None:
            row.update(asdict(local))
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": "OKX official historical market data",
        "portal_url": PORTAL_URL,
        "instrument": instrument,
        "instrument_type": "SPOT",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "root": str(root),
        "instrument_metadata": instrument_metadata,
        "feed_contract": {
            "l2-400": {
                "role": (
                    "10 ms L2 snapshots and updates; derive L1 and 500 ms top-10 grids; "
                    "the prior archive day supplies the initial two-minute lookback"
                ),
                "archive_format": "tar.gz containing newline-delimited JSON",
            },
            "trades": {
                "role": (
                    "genuine trade tape and footprint labels; derive one-minute OHLCV; "
                    "UTC+8 partitions include the end-date plus one spill file"
                ),
                "archive_format": "ZIP containing CSV",
            },
        },
        "artifact_count": len(rows),
        "complete_count": len(local_by_artifact),
        "total_content_length": sum(item.content_length for item in remote_by_artifact.values()),
        "artifacts": rows,
    }


def _open_with_retry(
    request: urllib.request.Request,
    *,
    attempts: int,
    timeout_seconds: float,
) -> Any:
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return urllib.request.urlopen(request, timeout=timeout_seconds)
        except urllib.error.HTTPError as error:
            last_error = error
            if error.code not in RETRYABLE_HTTP_STATUS:
                raise
        except (
            OSError,
            TimeoutError,
            http.client.HTTPException,
            urllib.error.URLError,
        ) as error:
            last_error = error
        if attempt < attempts:
            time.sleep(_retry_delay(attempt))
    raise RuntimeError(f"HTTP request failed: {request.full_url}") from last_error


def _retry_delay(attempt: int) -> float:
    return min(30.0, 0.75 * (2 ** (attempt - 1)))


def _copy_stream(source: BinaryIO, destination: BinaryIO) -> None:
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            return
        destination.write(chunk)


def _hash_file(path: Path) -> LocalArtifact:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(4 * 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            md5.update(chunk)
            sha256.update(chunk)
    return LocalArtifact(bytes=size, md5=md5.hexdigest(), sha256=sha256.hexdigest())


def _verify_etag(
    artifact: ArtifactSpec,
    remote: RemoteMetadata,
    local: LocalArtifact,
) -> None:
    if remote.etag is None:
        return
    normalized = remote.etag.strip().removeprefix("W/").strip('"').lower()
    if re.fullmatch(r"[0-9a-f]{32}", normalized) and normalized != local.md5:
        raise ValueError(f"ETag/MD5 mismatch for {artifact.relative_path}")


def _quarantine(path: Path) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = path.with_name(f"{path.name}.invalid-{stamp}")
    os.replace(path, destination)
    return destination


def _require_capacity(
    root: Path,
    artifacts: Sequence[ArtifactSpec],
    remote_by_artifact: dict[ArtifactSpec, RemoteMetadata],
) -> None:
    remaining = 0
    for artifact in artifacts:
        target = root / Path(artifact.relative_path)
        partial = target.with_name(f"{target.name}.part")
        expected = remote_by_artifact[artifact].content_length
        if target.is_file() and target.stat().st_size == expected:
            continue
        present = partial.stat().st_size if partial.is_file() else 0
        remaining += max(0, expected - present)
    reserve = 1024 * 1024 * 1024
    free = shutil.disk_usage(root).free
    if free < remaining + reserve:
        raise OSError(
            f"insufficient free space: need {remaining + reserve} bytes including reserve, "
            f"have {free}"
        )


def _parse_l2_record(raw_line: bytes, instrument: str, path: Path) -> dict[str, Any]:
    try:
        record = json.loads(raw_line)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid L2 JSON in {path.name}") from error
    if not isinstance(record, dict):
        raise ValueError(f"non-object L2 record in {path.name}")
    if record.get("instId") != instrument:
        raise ValueError(f"L2 instrument mismatch in {path.name}")
    if record.get("action") not in {"snapshot", "update"}:
        raise ValueError(f"invalid L2 action in {path.name}")
    try:
        int(record["ts"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid L2 timestamp in {path.name}") from error
    for side in ("bids", "asks"):
        levels = record.get(side)
        if not isinstance(levels, list):
            raise ValueError(f"invalid L2 {side} in {path.name}")
        for level in levels:
            if not isinstance(level, list) or len(level) < 3:
                raise ValueError(f"invalid L2 level in {path.name}")
            try:
                float(level[0])
                float(level[1])
                int(level[2])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid L2 level values in {path.name}") from error
    return record


def _apply_book_levels(book: dict[str, str], levels: list[list[Any]]) -> None:
    for level in levels:
        price = str(level[0])
        size = str(level[1])
        if float(size) == 0:
            book.pop(price, None)
        else:
            book[price] = size


def _stream_remaining_l2_lines(
    stream: BinaryIO,
    *,
    line_count: int,
    last_line: bytes | None,
) -> tuple[int, bytes | None]:
    """Consume a source member in large chunks while retaining its final record."""

    carry = b""
    while True:
        chunk = stream.read(8 * 1024 * 1024)
        if not chunk:
            break
        newline_count = chunk.count(b"\n")
        line_count += newline_count
        if newline_count == 0:
            carry += chunk
            continue
        prefix, next_carry = chunk.rsplit(b"\n", 1)
        prior_newline = prefix.rfind(b"\n")
        candidate = (
            prefix[prior_newline + 1 :]
            if prior_newline >= 0
            else carry + prefix
        )
        if candidate:
            last_line = candidate
        carry = next_carry
    if carry:
        line_count += 1
        last_line = carry
    return line_count, last_line


def _require_timestamp_day(timestamp_ms: int | None, expected_day: date, path: Path) -> None:
    if timestamp_ms is None:
        raise ValueError(f"missing timestamp in {path.name}")
    observed = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).date()
    if observed != expected_day:
        raise ValueError(
            f"timestamp day mismatch in {path.name}: expected {expected_day}, got {observed}"
        )


def _require_trade_partition(timestamp_ms: int | None, archive_day: date, path: Path) -> None:
    if timestamp_ms is None:
        raise ValueError(f"missing timestamp in {path.name}")
    partition_start = datetime(
        archive_day.year,
        archive_day.month,
        archive_day.day,
        tzinfo=UTC,
    ) - timedelta(hours=8)
    start_ms = int(partition_start.timestamp() * 1000)
    end_ms = start_ms + 24 * 60 * 60 * 1000
    if not start_ms <= timestamp_ms < end_ms:
        observed = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).isoformat()
        raise ValueError(
            f"trade timestamp outside UTC+8 partition in {path.name}: {observed}"
        )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _resolve_manifest_path(root: Path, relative_path: Any) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("manifest relative_path must be a non-empty string")
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("manifest relative_path must not be absolute")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("manifest relative_path escapes the acquisition root")
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"manifest artifact is not a regular non-symlink file: {relative_path}")
    return resolved


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO date (YYYY-MM-DD)") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire", help="probe and download official daily archives")
    acquire.add_argument("--root", type=Path, required=True)
    acquire.add_argument("--instrument", default="BTC-USDT")
    acquire.add_argument("--start", type=_parse_date, required=True)
    acquire.add_argument("--end", type=_parse_date, required=True)
    acquire.add_argument(
        "--feed",
        action="append",
        choices=("l2-400", "trades"),
        dest="feeds",
        help="repeat to select feeds; defaults to both required feeds",
    )
    acquire.add_argument("--workers", type=int, default=3)
    acquire.add_argument("--attempts", type=int, default=8)
    acquire.add_argument("--timeout-seconds", type=float, default=60.0)
    acquire.add_argument("--manifest-only", action="store_true")

    audit = subparsers.add_parser("audit", help="verify hashes, archives, and source schemas")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--parsed-prefix-rows", type=int, default=1_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "acquire":
            manifest = acquire_month(
                root=args.root,
                instrument=args.instrument,
                start=args.start,
                end=args.end,
                feeds=args.feeds or ("l2-400", "trades"),
                workers=args.workers,
                attempts=args.attempts,
                timeout_seconds=args.timeout_seconds,
                manifest_only=args.manifest_only,
            )
            print(
                json.dumps(
                    {
                        "manifest": str(args.root.expanduser().resolve() / MANIFEST_NAME),
                        "complete": manifest["complete_count"],
                        "total": manifest["artifact_count"],
                        "bytes": manifest["total_content_length"],
                    },
                    sort_keys=True,
                )
            )
        else:
            report = audit_acquisition(
                args.root,
                parsed_prefix_rows=args.parsed_prefix_rows,
            )
            print(
                json.dumps(
                    {
                        "audit": str(args.root.expanduser().resolve() / AUDIT_NAME),
                        "status": report["status"],
                        "artifacts": report["artifact_count"],
                    },
                    sort_keys=True,
                )
            )
        return 0
    except (OSError, RuntimeError, ValueError, urllib.error.URLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
