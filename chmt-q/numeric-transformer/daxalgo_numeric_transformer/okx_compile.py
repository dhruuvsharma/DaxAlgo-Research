"""Stream verified OKX archives into raw engineered CHMT-Q Arrow tensors.

Historical OKX archives expose exchange event time but not an independent receipt
clock or native L2 sequence. This compiler therefore emits useful research tensors
with explicit approximation features and fail-closed training capabilities.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
import tarfile
import time
import zipfile
from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta
from datetime import time as datetime_time
from functools import lru_cache
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, BinaryIO, Literal

import numpy as np
import torch

from .chmt_q.config import ChmtQConfig
from .chmt_q.contracts import ChmtQBatch
from .chmt_q.data import (
    DatasetCapabilities,
    DatasetManifest,
    EngineeredExample,
    ExampleMetadata,
    write_engineered_dataset,
)
from .chmt_q.experiment import source_tree_sha256
from .okx_history import AUDIT_NAME, MANIFEST_NAME

try:
    import orjson
except ImportError:  # pragma: no cover - environment-dependent optional accelerator
    orjson = None


COMPILER_REPORT_NAME = "okx-compiler-report.json"
COMPILER_SCHEMA_VERSION = 1
DAY_MS = 86_400_000
MINUTE_MS = 60_000
GRID_MS = 500
QUANTITY_SCALE = 100_000_000
SOURCE_UNIT_TOLERANCE_NUMERATOR = 1
SOURCE_UNIT_TOLERANCE_DENOMINATOR = 1_000_000
POWERS_OF_TEN = tuple(10**value for value in range(101))
TICK_SIZE_TEXT = "0.1"
TICK_SIZE = 0.1
ZERO_SHA256 = "0" * 64


@dataclass(frozen=True, slots=True)
class OkxCompilerPolicy:
    """Versioned choices that resolve the source-specific compiler coordinate."""

    schema_version: int = COMPILER_SCHEMA_VERSION
    instrument_id: str = "BTC-USDT"
    session_calendar: str = "utc-analysis-day-v1"
    session_reset_policy: str = "reset-at-each-utc-analysis-day-v1"
    side_order: tuple[str, str] = ("bid", "ask")
    decision_lag_ms: int = 0
    label_grace_ms: int = 0
    grid_ms: int = GRID_MS
    l1_max_age_ms: int = 2_000
    l2_max_age_ms: int = 1_000
    l2_state_reset_gap_ms: int = 5_000
    bar_anchor_max_age_slots: int = 2
    bar_state_reset_gap_slots: int = 5
    ema_span: int = 20
    ema_min_valid_bars: int = 5
    first_cutoff_minute: int = 5
    final_cutoff_minute: int = 23 * 60 + 55
    availability_policy: str = "exchange-event-time-proxy"
    unavailable_sequence_gap_value: float = 0.0
    sequence_gap_policy: str = "unsupported-zero-with-provenance-blocker-v1"
    carried_flag_policy: str = "same-selected-observation-as-prior-grid-v1"
    one_sided_l2_reference_policy: str = "prior-fresh-two-sided-derived-l1-v1"
    paired_rank_imbalance_policy: str = "bid-minus-ask-repeated-v1"
    source_decimal_quantization: str = "nearest-unit-half-away-max-error-1e-6-unit-v1"
    quantity_scale: int = QUANTITY_SCALE
    target_tape_policy: str = "archive-crc-and-contiguous-trade-id-v1"
    aggressor_side_policy: str = "okx-archive-side-buy-positive-v1"

    @property
    def ema_alpha(self) -> float:
        return 2.0 / (self.ema_span + 1.0)

    def payload(self) -> dict[str, Any]:
        value = asdict(self)
        value["side_order"] = list(self.side_order)
        value["ema_alpha"] = self.ema_alpha
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.payload())


@dataclass(frozen=True, slots=True)
class SourceArtifact:
    day: date
    feed: Literal["l2-400", "trades"]
    path: Path
    sha256: str
    content_length: int


@dataclass(frozen=True, slots=True)
class SourceContract:
    root: Path
    instrument_id: str
    source_manifest_sha256: str
    source_audit_sha256: str
    tick_size: str
    lot_size: str
    minimum_size: str
    artifacts: tuple[SourceArtifact, ...]

    def for_feed(self, feed: str) -> tuple[SourceArtifact, ...]:
        return tuple(item for item in self.artifacts if item.feed == feed)


@dataclass(frozen=True, slots=True)
class MinuteTape:
    minute_start_ms: int
    open_tick: int
    high_tick: int
    low_tick: int
    close_tick: int
    total_units: int
    buy_units: int
    sell_units: int
    poc_tick: int
    buy_poc_tick: int | None
    sell_poc_tick: int | None
    trade_count: int
    first_trade_id: int
    last_trade_id: int


@dataclass(slots=True)
class TradeLoadStats:
    archive_count: int = 0
    row_count: int = 0
    minute_count: int = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None
    first_trade_id: int | None = None
    last_trade_id: int | None = None
    id_gap_count: int = 0
    out_of_order_count: int = 0
    price_quantization_count: int = 0
    size_quantization_count: int = 0
    max_price_quantization_error_units: float = 0.0
    max_size_quantization_error_units: float = 0.0

    @property
    def tape_complete(self) -> bool:
        return (
            self.row_count > 0
            and self.id_gap_count == 0
            and self.out_of_order_count == 0
        )


@dataclass(slots=True)
class _MinuteAccumulator:
    minute_start_ms: int
    open_tick: int
    high_tick: int
    low_tick: int
    close_tick: int
    total_units: int = 0
    buy_units: int = 0
    sell_units: int = 0
    trade_count: int = 0
    first_trade_id: int = 0
    last_trade_id: int = 0
    total_by_tick: dict[int, int] = field(default_factory=dict)
    buy_by_tick: dict[int, int] = field(default_factory=dict)
    sell_by_tick: dict[int, int] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        *,
        minute_start_ms: int,
        price_tick: int,
        trade_id: int,
    ) -> _MinuteAccumulator:
        return cls(
            minute_start_ms=minute_start_ms,
            open_tick=price_tick,
            high_tick=price_tick,
            low_tick=price_tick,
            close_tick=price_tick,
            first_trade_id=trade_id,
            last_trade_id=trade_id,
        )

    def add(self, *, price_tick: int, size_units: int, side: str, trade_id: int) -> None:
        self.high_tick = max(self.high_tick, price_tick)
        self.low_tick = min(self.low_tick, price_tick)
        self.close_tick = price_tick
        self.total_units += size_units
        self.trade_count += 1
        self.last_trade_id = trade_id
        self.total_by_tick[price_tick] = self.total_by_tick.get(price_tick, 0) + size_units
        if side == "buy":
            self.buy_units += size_units
            self.buy_by_tick[price_tick] = self.buy_by_tick.get(price_tick, 0) + size_units
        else:
            self.sell_units += size_units
            self.sell_by_tick[price_tick] = self.sell_by_tick.get(price_tick, 0) + size_units

    def finish(self) -> MinuteTape:
        return MinuteTape(
            minute_start_ms=self.minute_start_ms,
            open_tick=self.open_tick,
            high_tick=self.high_tick,
            low_tick=self.low_tick,
            close_tick=self.close_tick,
            total_units=self.total_units,
            buy_units=self.buy_units,
            sell_units=self.sell_units,
            poc_tick=_poc_tick(self.total_by_tick),
            buy_poc_tick=_poc_tick(self.buy_by_tick) if self.buy_by_tick else None,
            sell_poc_tick=_poc_tick(self.sell_by_tick) if self.sell_by_tick else None,
            trade_count=self.trade_count,
            first_trade_id=self.first_trade_id,
            last_trade_id=self.last_trade_id,
        )


@dataclass(frozen=True, slots=True)
class _BarToken:
    values: np.ndarray
    valid: bool
    source_valid: bool
    close_tick: int | None
    close_ms: int
    ema_after: float | None
    valid_bar_count: int


@dataclass(frozen=True, slots=True)
class _MicroFrame:
    l1: np.ndarray
    l1_valid: bool
    l1_mid_ticks: float
    l2: np.ndarray
    l2_level_mask: np.ndarray
    l2_valid: bool
    l1_carried: bool
    l2_carried: bool
    approximate: bool
    one_sided: bool
    rejected: bool


@dataclass(frozen=True, slots=True)
class _L1Observation:
    mid_ticks: float
    bid_tick: int
    ask_tick: int
    bid_units: int
    ask_units: int
    event_ms: int

    @property
    def identity(self) -> tuple[int, int, int, int]:
        return self.bid_tick, self.ask_tick, self.bid_units, self.ask_units


@dataclass(slots=True)
class L2CompileStats:
    archive_count: int = 0
    parsed_rows: int = 0
    emitted_frames: int = 0
    emitted_examples: int = 0
    snapshot_count: int = 0
    update_count: int = 0
    rejected_frames: int = 0
    one_sided_frames: int = 0
    price_quantization_count: int = 0
    size_quantization_count: int = 0
    max_price_quantization_error_units: float = 0.0
    max_size_quantization_error_units: float = 0.0
    maximum_event_gap_ms: int = 0
    first_cutoff_ms: int | None = None
    last_cutoff_ms: int | None = None
    started_at: float = field(default_factory=time.monotonic)

    def payload(self) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1.0e-9)
        return {
            "archive_count": self.archive_count,
            "parsed_rows": self.parsed_rows,
            "emitted_frames": self.emitted_frames,
            "emitted_examples": self.emitted_examples,
            "snapshot_count": self.snapshot_count,
            "update_count": self.update_count,
            "rejected_frames": self.rejected_frames,
            "one_sided_frames": self.one_sided_frames,
            "price_quantization_count": self.price_quantization_count,
            "size_quantization_count": self.size_quantization_count,
            "max_price_quantization_error_units": self.max_price_quantization_error_units,
            "max_size_quantization_error_units": self.max_size_quantization_error_units,
            "maximum_event_gap_ms": self.maximum_event_gap_ms,
            "first_cutoff_ms": self.first_cutoff_ms,
            "last_cutoff_ms": self.last_cutoff_ms,
            "elapsed_seconds": elapsed,
            "parsed_rows_per_second": self.parsed_rows / elapsed,
        }


class _OrderBook:
    def __init__(self, stats: L2CompileStats | None = None) -> None:
        self.bid_sizes: dict[int, int] = {}
        self.ask_sizes: dict[int, int] = {}
        self.bid_prices: list[int] = []
        self.ask_prices: list[int] = []
        self.last_event_ms: int | None = None
        self.revision = 0
        self.stats = stats

    def apply(self, record: dict[str, Any], timestamp_ms: int) -> None:
        action = record.get("action")
        if action == "snapshot":
            self.bid_sizes = self._snapshot_side(record.get("bids"), "bids")
            self.ask_sizes = self._snapshot_side(record.get("asks"), "asks")
            self.bid_prices = sorted(self.bid_sizes)
            self.ask_prices = sorted(self.ask_sizes)
        elif action == "update":
            self._update_side(self.bid_sizes, self.bid_prices, record.get("bids"), "bids")
            self._update_side(self.ask_sizes, self.ask_prices, record.get("asks"), "asks")
        else:
            raise ValueError(f"unsupported L2 action: {action!r}")
        self.last_event_ms = timestamp_ms
        self.revision += 1

    def top(self, side: Literal["bid", "ask"], count: int = 10) -> list[tuple[int, int]]:
        if side == "bid":
            prices = reversed(self.bid_prices[-count:])
            sizes = self.bid_sizes
        else:
            prices = iter(self.ask_prices[:count])
            sizes = self.ask_sizes
        return [(price, sizes[price]) for price in prices]

    def _snapshot_side(self, value: Any, name: str) -> dict[int, int]:
        if not isinstance(value, list):
            raise ValueError(f"L2 snapshot {name} must be an array")
        result: dict[int, int] = {}
        for level in value:
            price, size = _parse_l2_level(level, self.stats)
            if size == 0:
                raise ValueError(f"L2 snapshot {name} contains a zero-size level")
            if price in result:
                raise ValueError(f"L2 snapshot {name} contains a duplicate price")
            result[price] = size
        return result

    def _update_side(
        self,
        sizes: dict[int, int],
        prices: list[int],
        value: Any,
        name: str,
    ) -> None:
        if not isinstance(value, list):
            raise ValueError(f"L2 update {name} must be an array")
        seen: set[int] = set()
        for level in value:
            price, size = _parse_l2_level(level, self.stats)
            if price in seen:
                raise ValueError(f"L2 update {name} contains a duplicate price")
            seen.add(price)
            present = price in sizes
            if size == 0:
                if present:
                    del sizes[price]
                    index = bisect.bisect_left(prices, price)
                    if index >= len(prices) or prices[index] != price:
                        raise RuntimeError("book price index diverged from size map")
                    prices.pop(index)
            else:
                sizes[price] = size
                if not present:
                    bisect.insort(prices, price)


def load_source_contract(root: str | Path, policy: OkxCompilerPolicy) -> SourceContract:
    """Bind the compiler to the audited source manifest and captured instrument contract."""

    source_root = Path(root).expanduser().resolve()
    manifest_path = source_root / MANIFEST_NAME
    audit_path = source_root / AUDIT_NAME
    if not manifest_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("OKX source manifest and audit report are required")
    manifest_bytes = manifest_path.read_bytes()
    audit_bytes = audit_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    audit = json.loads(audit_bytes)
    if audit.get("status") != "pass" or audit.get("source_manifest_sha256") != manifest_sha:
        raise ValueError("OKX source audit does not pass or bind the current manifest")
    if manifest.get("instrument") != policy.instrument_id:
        raise ValueError("source instrument differs from compiler policy")
    metadata = audit.get("instrument_metadata", {})
    if metadata.get("tick_size") != TICK_SIZE_TEXT:
        raise ValueError("captured OKX tick size differs from the frozen compiler coordinate")
    if metadata.get("lot_size") != "0.00000001":
        raise ValueError("captured OKX lot size differs from the frozen compiler coordinate")
    if metadata.get("minimum_size") != "0.00001":
        raise ValueError("captured OKX minimum size differs from the frozen compiler coordinate")

    artifacts: list[SourceArtifact] = []
    identities: set[tuple[str, date]] = set()
    for item in manifest.get("artifacts", []):
        feed = item.get("feed")
        if feed not in {"l2-400", "trades"} or item.get("status") != "complete":
            raise ValueError("source manifest contains an incomplete or unsupported artifact")
        path = _resolve_relative(source_root, item.get("relative_path"))
        expected_size = int(item["content_length"])
        if path.stat().st_size != expected_size:
            raise ValueError(f"source artifact size changed: {path.name}")
        expected_sha256 = str(item["sha256"])
        _validate_sha256(expected_sha256, f"source artifact SHA-256 ({path.name})")
        if _file_sha256(path) != expected_sha256:
            raise ValueError(f"source artifact SHA-256 changed: {path.name}")
        artifact_day = date.fromisoformat(item["day"])
        identity = (feed, artifact_day)
        if identity in identities:
            raise ValueError(f"duplicate source artifact for {feed} on {artifact_day}")
        identities.add(identity)
        artifacts.append(
            SourceArtifact(
                day=artifact_day,
                feed=feed,
                path=path,
                sha256=expected_sha256,
                content_length=expected_size,
            )
        )
    if not artifacts:
        raise ValueError("source manifest contains no artifacts")
    return SourceContract(
        root=source_root,
        instrument_id=policy.instrument_id,
        source_manifest_sha256=manifest_sha,
        source_audit_sha256=audit_sha,
        tick_size=str(metadata["tick_size"]),
        lot_size=str(metadata["lot_size"]),
        minimum_size=str(metadata["minimum_size"]),
        artifacts=tuple(sorted(artifacts, key=lambda item: (item.day, item.feed))),
    )


def load_trade_minutes(
    source: SourceContract,
    *,
    start: date,
    end: date,
) -> tuple[dict[int, MinuteTape], TradeLoadStats]:
    """Aggregate genuine trades into exact UTC-minute OHLCV and footprint summaries."""

    requested_start = _day_start_ms(start)
    requested_end = _day_start_ms(end + timedelta(days=1))
    selected = [
        artifact
        for artifact in source.for_feed("trades")
        if _trade_partition_end_ms(artifact.day) > requested_start
        and _trade_partition_start_ms(artifact.day) < requested_end
    ]
    if not selected:
        raise ValueError("no trade archives overlap the requested compiler window")

    summaries: dict[int, MinuteTape] = {}
    stats = TradeLoadStats(archive_count=len(selected))
    accumulator: _MinuteAccumulator | None = None
    previous_timestamp: int | None = None
    previous_trade_id: int | None = None
    expected_header = b"instrument_name,trade_id,side,price,size,created_time"
    instrument_bytes = source.instrument_id.encode("ascii")

    for artifact in selected:
        partition_start = _trade_partition_start_ms(artifact.day)
        partition_end = _trade_partition_end_ms(artifact.day)
        archive_rows = 0
        member_name = f"{source.instrument_id}-trades-{artifact.day.isoformat()}.csv"
        with zipfile.ZipFile(artifact.path) as archive:
            names = [name for name in archive.namelist() if not name.endswith("/")]
            if names != [member_name]:
                raise ValueError(f"unexpected trade ZIP members in {artifact.path.name}")
            with archive.open(member_name) as stream:
                if stream.readline().rstrip(b"\r\n") != expected_header:
                    raise ValueError(f"unexpected trade schema in {artifact.path.name}")
                for raw_line in stream:
                    parts = raw_line.rstrip(b"\r\n").split(b",")
                    if len(parts) != 6 or parts[0] != instrument_bytes:
                        raise ValueError(f"invalid trade row in {artifact.path.name}")
                    trade_id = int(parts[1])
                    side = parts[2].decode("ascii")
                    if side not in {"buy", "sell"}:
                        raise ValueError(f"unknown OKX trade side: {side!r}")
                    price_tick, price_error = _parse_source_fixed(parts[3], 1)
                    size_units, size_error = _parse_source_fixed(parts[4], 8)
                    if price_error:
                        stats.price_quantization_count += 1
                        stats.max_price_quantization_error_units = max(
                            stats.max_price_quantization_error_units,
                            price_error,
                        )
                    if size_error:
                        stats.size_quantization_count += 1
                        stats.max_size_quantization_error_units = max(
                            stats.max_size_quantization_error_units,
                            size_error,
                        )
                    timestamp_ms = int(parts[5])
                    if price_tick <= 0 or size_units <= 0:
                        raise ValueError("trade price and size must be positive")
                    if not partition_start <= timestamp_ms < partition_end:
                        raise ValueError(
                            f"trade timestamp falls outside partition {artifact.day}"
                        )
                    if previous_timestamp is not None and timestamp_ms < previous_timestamp:
                        stats.out_of_order_count += 1
                    if previous_trade_id is not None and trade_id != previous_trade_id + 1:
                        stats.id_gap_count += 1
                    previous_timestamp = timestamp_ms
                    previous_trade_id = trade_id
                    stats.first_timestamp_ms = (
                        timestamp_ms
                        if stats.first_timestamp_ms is None
                        else stats.first_timestamp_ms
                    )
                    stats.last_timestamp_ms = timestamp_ms
                    stats.first_trade_id = (
                        trade_id if stats.first_trade_id is None else stats.first_trade_id
                    )
                    stats.last_trade_id = trade_id
                    stats.row_count += 1
                    archive_rows += 1

                    minute_start = timestamp_ms - timestamp_ms % MINUTE_MS
                    if accumulator is None or minute_start != accumulator.minute_start_ms:
                        if accumulator is not None:
                            summary = accumulator.finish()
                            summaries[summary.minute_start_ms] = summary
                        accumulator = _MinuteAccumulator.start(
                            minute_start_ms=minute_start,
                            price_tick=price_tick,
                            trade_id=trade_id,
                        )
                    accumulator.add(
                        price_tick=price_tick,
                        size_units=size_units,
                        side=side,
                        trade_id=trade_id,
                    )
        if archive_rows == 0:
            raise ValueError(f"trade archive is empty: {artifact.path.name}")
    if accumulator is not None:
        summary = accumulator.finish()
        summaries[summary.minute_start_ms] = summary
    stats.minute_count = len(summaries)
    if not stats.tape_complete:
        raise ValueError(
            "trade tape is not contiguous: "
            f"id_gaps={stats.id_gap_count}, out_of_order={stats.out_of_order_count}"
        )
    return summaries, stats


def compile_okx_dataset(
    *,
    source_root: str | Path,
    destination: str | Path,
    start: date,
    end: date,
    rows_per_shard: int = 256,
    rows_per_record_batch: int = 64,
    compression: Literal["lz4", "zstd"] | None = "zstd",
    max_examples: int | None = None,
) -> tuple[DatasetManifest, dict[str, Any]]:
    """Compile an inclusive UTC-day range and return its verified dataset manifest/report."""

    if end < start:
        raise ValueError("end must be on or after start")
    if max_examples is not None and max_examples <= 0:
        raise ValueError("max_examples must be positive")
    policy = OkxCompilerPolicy()
    package_root = Path(__file__).resolve().parents[1]
    builder_sha = source_tree_sha256(package_root, include_tests=False)
    source = load_source_contract(source_root, policy)
    _require_source_coverage(source, start=start, end=end)
    trade_minutes, trade_stats = load_trade_minutes(source, start=start, end=end)
    l2_stats = L2CompileStats()
    capabilities = DatasetCapabilities(
        training_ready=False,
        rights_confirmed=False,
        provenance_complete=False,
        readiness_evidence_sha256=ZERO_SHA256,
        rights_evidence_sha256=ZERO_SHA256,
        provenance_evidence_sha256=ZERO_SHA256,
    )
    examples = _identity_checked_examples(
        iter_okx_examples(
            source,
            trade_minutes,
            start=start,
            end=end,
            policy=policy,
            stats=l2_stats,
            max_examples=max_examples,
        ),
        source=source,
        package_root=package_root,
        builder_sha256=builder_sha,
    )
    manifest = write_engineered_dataset(
        destination,
        examples,
        source_manifest_sha256=source.source_manifest_sha256,
        builder_sha256=builder_sha,
        capabilities=capabilities,
        rows_per_shard=rows_per_shard,
        rows_per_record_batch=rows_per_record_batch,
        compression=compression,
    )
    report = {
        "schema_version": COMPILER_SCHEMA_VERSION,
        "status": "compiled_non_promotable",
        "dataset_id": manifest.dataset_id,
        "dataset_rows": manifest.row_count,
        "instrument_id": source.instrument_id,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "max_examples": max_examples,
        "truncated": max_examples is not None and manifest.row_count >= max_examples,
        "first_cutoff_utc": _timestamp_ms_iso(l2_stats.first_cutoff_ms),
        "last_cutoff_utc": _timestamp_ms_iso(l2_stats.last_cutoff_ms),
        "source_manifest_sha256": source.source_manifest_sha256,
        "source_audit_sha256": source.source_audit_sha256,
        "builder_sha256": builder_sha,
        "policy": policy.payload(),
        "policy_sha256": policy.sha256,
        "compression": compression,
        "rows_per_shard": rows_per_shard,
        "rows_per_record_batch": rows_per_record_batch,
        "pyarrow_version": package_version("pyarrow"),
        "json_decoder": "orjson" if orjson is not None else "stdlib-json",
        "trade_stats": asdict(trade_stats),
        "l2_stats": l2_stats.payload(),
        "capabilities": manifest.capabilities.to_payload(),
        "blockers": [
            "OKX archives have no independent receive/availability timestamp.",
            "OKX 400-level archives have no native sequence/checksum continuity.",
            "The captured instrument contract is current, not proven effective-dated for June.",
            "OKX historical-data training rights have not been independently approved.",
        ],
    }
    report_path = Path(destination).resolve() / COMPILER_REPORT_NAME
    _write_json_atomic(report_path, report)
    return manifest, report


def iter_okx_examples(
    source: SourceContract,
    trade_minutes: dict[int, MinuteTape],
    *,
    start: date,
    end: date,
    policy: OkxCompilerPolicy,
    stats: L2CompileStats,
    max_examples: int | None = None,
) -> Iterator[EngineeredExample]:
    """Yield deterministic examples in cutoff order with bounded working memory."""

    l2_by_day = {item.day: item for item in source.for_feed("l2-400")}
    emitted = 0
    current = start
    while current <= end:
        artifact = l2_by_day.get(current)
        if artifact is None:
            raise ValueError(f"missing L2 archive for {current}")
        context = _build_session_bars(current, trade_minutes, policy)
        for example in _iter_day_examples(
            artifact,
            context,
            trade_minutes,
            source=source,
            policy=policy,
            stats=stats,
        ):
            yield example
            emitted += 1
            if max_examples is not None and emitted >= max_examples:
                return
        current += timedelta(days=1)


def _identity_checked_examples(
    examples: Iterable[EngineeredExample],
    *,
    source: SourceContract,
    package_root: Path,
    builder_sha256: str,
) -> Iterator[EngineeredExample]:
    yield from examples
    if source_tree_sha256(package_root, include_tests=False) != builder_sha256:
        raise RuntimeError("compiler source tree changed while the dataset was being built")
    if _file_sha256(source.root / MANIFEST_NAME) != source.source_manifest_sha256:
        raise RuntimeError("source manifest changed while the dataset was being built")
    if _file_sha256(source.root / AUDIT_NAME) != source.source_audit_sha256:
        raise RuntimeError("source audit changed while the dataset was being built")
    for artifact in source.artifacts:
        if (
            artifact.path.stat().st_size != artifact.content_length
            or _file_sha256(artifact.path) != artifact.sha256
        ):
            raise RuntimeError(
                f"source artifact changed while the dataset was being built: {artifact.path.name}"
            )


@dataclass(frozen=True, slots=True)
class _SessionBars:
    day: date
    day_start_ms: int
    tokens: tuple[_BarToken, ...]


def _build_session_bars(
    day: date,
    trade_minutes: dict[int, MinuteTape],
    policy: OkxCompilerPolicy,
) -> _SessionBars:
    day_start = _day_start_ms(day)
    tokens: list[_BarToken] = []
    previous_close: int | None = None
    ema: float | None = None
    valid_bar_count = 0
    missing_slots = 0
    for index in range(24 * 60):
        minute_start = day_start + index * MINUTE_MS
        summary = trade_minutes.get(minute_start)
        close_ms = minute_start + MINUTE_MS
        if summary is None:
            missing_slots += 1
            tokens.append(
                _BarToken(
                    values=np.zeros(16, dtype=np.float32),
                    valid=False,
                    source_valid=False,
                    close_tick=None,
                    close_ms=close_ms,
                    ema_after=ema,
                    valid_bar_count=valid_bar_count,
                )
            )
            continue
        if missing_slots > policy.bar_state_reset_gap_slots:
            previous_close = None
            ema = None
            valid_bar_count = 0
        log_volume = math.log1p(summary.total_units / QUANTITY_SCALE)
        values = np.zeros(16, dtype=np.float32)
        feature_valid = previous_close is not None and ema is not None
        if feature_valid:
            assert previous_close is not None and ema is not None
            values[0] = summary.open_tick - previous_close
            values[1] = summary.high_tick - previous_close
            values[2] = summary.low_tick - previous_close
            values[3] = summary.close_tick - previous_close
            values[4] = math.log(summary.close_tick / previous_close)
            values[5] = summary.high_tick - summary.low_tick
            values[6] = summary.close_tick - summary.open_tick
            values[7] = summary.high_tick - max(summary.open_tick, summary.close_tick)
            values[8] = min(summary.open_tick, summary.close_tick) - summary.low_tick
            values[9] = (summary.close_tick - summary.low_tick) / max(
                summary.high_tick - summary.low_tick,
                1,
            )
            values[10] = log_volume
            values[11] = log_volume - ema
            values[13] = math.log1p(min(missing_slots, 63)) / math.log(64)
            progress = (index + 1) / (24 * 60)
            values[14] = math.sin(2 * math.pi * progress)
            values[15] = math.cos(2 * math.pi * progress)
        ema = (
            log_volume
            if ema is None
            else policy.ema_alpha * log_volume + (1 - policy.ema_alpha) * ema
        )
        previous_close = summary.close_tick
        valid_bar_count += 1
        missing_slots = 0
        tokens.append(
            _BarToken(
                values=values,
                valid=feature_valid,
                source_valid=True,
                close_tick=summary.close_tick,
                close_ms=close_ms,
                ema_after=ema,
                valid_bar_count=valid_bar_count,
            )
        )
    return _SessionBars(day=day, day_start_ms=day_start, tokens=tuple(tokens))


def _iter_day_examples(
    artifact: SourceArtifact,
    context: _SessionBars,
    trade_minutes: dict[int, MinuteTape],
    *,
    source: SourceContract,
    policy: OkxCompilerPolicy,
    stats: L2CompileStats,
) -> Iterator[EngineeredExample]:
    day_start = context.day_start_ms
    final_cutoff = day_start + policy.final_cutoff_minute * MINUTE_MS
    expected_member = (
        f"{source.instrument_id}-L2orderbook-400lv-{artifact.day.isoformat()}.data"
    )
    book = _OrderBook(stats)
    frames: deque[_MicroFrame] = deque(maxlen=ChmtQConfig().micro_steps)
    next_grid = day_start
    previous_valid_l1: _L1Observation | None = None
    previous_grid_l1_identity: tuple[int, int, int, int] | None = None
    previous_grid_l2_revision: int | None = None
    current_timestamp: int | None = None
    first_record = True
    stats.archive_count += 1

    def emit_until(limit_ms: int, *, inclusive: bool) -> Iterator[EngineeredExample]:
        nonlocal next_grid, previous_valid_l1
        nonlocal previous_grid_l1_identity, previous_grid_l2_revision
        comparison = (
            (lambda value: value <= limit_ms)
            if inclusive
            else (lambda value: value < limit_ms)
        )
        while next_grid <= final_cutoff and comparison(next_grid):
            (
                frame,
                previous_valid_l1,
                previous_grid_l1_identity,
                previous_grid_l2_revision,
            ) = _build_micro_frame(
                book,
                grid_ms=next_grid,
                previous_valid_l1=previous_valid_l1,
                previous_grid_l1_identity=previous_grid_l1_identity,
                previous_grid_l2_revision=previous_grid_l2_revision,
                policy=policy,
            )
            frames.append(frame)
            stats.emitted_frames += 1
            if frame.rejected:
                stats.rejected_frames += 1
            if frame.one_sided:
                stats.one_sided_frames += 1
            if (
                next_grid % MINUTE_MS == 0
                and policy.first_cutoff_minute
                <= (next_grid - day_start) // MINUTE_MS
                <= policy.final_cutoff_minute
                and len(frames) == ChmtQConfig().micro_steps
            ):
                example = _build_example(
                    tuple(frames),
                    cutoff_ms=next_grid,
                    context=context,
                    trade_minutes=trade_minutes,
                    source=source,
                    l2_artifact=artifact,
                    policy=policy,
                )
                if example is not None:
                    stats.emitted_examples += 1
                    stats.first_cutoff_ms = (
                        next_grid if stats.first_cutoff_ms is None else stats.first_cutoff_ms
                    )
                    stats.last_cutoff_ms = next_grid
                    yield example
            next_grid += policy.grid_ms

    with tarfile.open(artifact.path, mode="r|gz") as archive:
        members = 0
        for member in archive:
            if not member.isfile():
                continue
            members += 1
            if member.name != expected_member:
                raise ValueError(f"unexpected L2 member in {artifact.path.name}: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"cannot read L2 member in {artifact.path.name}")
            for raw_line in stream:
                record = _json_loads(raw_line)
                if not isinstance(record, dict) or record.get("instId") != source.instrument_id:
                    raise ValueError(f"invalid L2 record in {artifact.path.name}")
                timestamp_ms = int(record["ts"])
                if not day_start <= timestamp_ms < day_start + DAY_MS:
                    raise ValueError(f"L2 timestamp falls outside {artifact.day}")
                if timestamp_ms > final_cutoff:
                    if current_timestamp is not None:
                        yield from emit_until(current_timestamp, inclusive=True)
                    yield from emit_until(final_cutoff, inclusive=True)
                    _consume(stream)
                    break
                if current_timestamp is None:
                    yield from emit_until(timestamp_ms, inclusive=False)
                    current_timestamp = timestamp_ms
                elif timestamp_ms < current_timestamp:
                    raise ValueError(f"out-of-order L2 timestamp in {artifact.path.name}")
                elif timestamp_ms != current_timestamp:
                    event_gap_ms = timestamp_ms - current_timestamp
                    stats.maximum_event_gap_ms = max(
                        stats.maximum_event_gap_ms,
                        event_gap_ms,
                    )
                    if (
                        event_gap_ms > policy.l2_state_reset_gap_ms
                        and record.get("action") != "snapshot"
                    ):
                        raise ValueError(
                            "L2 event gap exceeds the reset threshold without a new snapshot: "
                            f"{event_gap_ms}ms in {artifact.path.name}"
                        )
                    yield from emit_until(current_timestamp, inclusive=True)
                    yield from emit_until(timestamp_ms, inclusive=False)
                    current_timestamp = timestamp_ms
                if first_record and record.get("action") != "snapshot":
                    raise ValueError(f"first L2 record is not a snapshot in {artifact.path.name}")
                first_record = False
                book.apply(record, timestamp_ms)
                stats.parsed_rows += 1
                if record["action"] == "snapshot":
                    stats.snapshot_count += 1
                else:
                    stats.update_count += 1
            else:
                if current_timestamp is not None:
                    yield from emit_until(current_timestamp, inclusive=True)
                yield from emit_until(final_cutoff, inclusive=True)
        if members != 1 or first_record:
            raise ValueError(
                f"L2 archive must contain exactly one non-empty member: {artifact.path}"
            )


def _build_micro_frame(
    book: _OrderBook,
    *,
    grid_ms: int,
    previous_valid_l1: _L1Observation | None,
    previous_grid_l1_identity: tuple[int, int, int, int] | None,
    previous_grid_l2_revision: int | None,
    policy: OkxCompilerPolicy,
) -> tuple[
    _MicroFrame,
    _L1Observation | None,
    tuple[int, int, int, int] | None,
    int | None,
]:
    l1 = np.zeros(16, dtype=np.float32)
    l2 = np.zeros((2, 10, 12), dtype=np.float32)
    l2_mask = np.zeros((2, 10), dtype=np.bool_)
    bids = book.top("bid")
    asks = book.top("ask")
    one_sided = not bids or not asks
    age_ms = grid_ms - book.last_event_ms if book.last_event_ms is not None else None
    crossed = bool(bids and asks and bids[0][0] >= asks[0][0])
    valid_l1 = (
        age_ms is not None
        and 0 <= age_ms <= policy.l1_max_age_ms
        and not one_sided
        and not crossed
        and bids[0][1] > 0
        and asks[0][1] > 0
    )
    mid_ticks = 0.0
    next_previous_valid = previous_valid_l1
    next_grid_l1_identity: tuple[int, int, int, int] | None = None
    l1_carried = False
    if valid_l1:
        bid_tick, bid_units = bids[0]
        ask_tick, ask_units = asks[0]
        mid_ticks = (bid_tick + ask_tick) / 2.0
        bid_size = bid_units / QUANTITY_SCALE
        ask_size = ask_units / QUANTITY_SCALE
        size_sum = bid_size + ask_size
        if size_sum <= 0:
            valid_l1 = False
        else:
            assert book.last_event_ms is not None
            log_bid = math.log1p(bid_size)
            log_ask = math.log1p(ask_size)
            microprice_ticks = (ask_tick * bid_size + bid_tick * ask_size) / size_sum
            l1[0] = mid_ticks
            l1[2] = ask_tick - bid_tick
            l1[3] = log_bid
            l1[4] = log_ask
            l1[5] = (bid_size - ask_size) / size_sum
            l1[6] = microprice_ticks - mid_ticks
            if previous_valid_l1 is not None:
                l1[1] = mid_ticks - previous_valid_l1.mid_ticks
                l1[7] = bid_tick - previous_valid_l1.bid_tick
                l1[8] = ask_tick - previous_valid_l1.ask_tick
                l1[9] = log_bid - math.log1p(
                    previous_valid_l1.bid_units / QUANTITY_SCALE
                )
                l1[10] = log_ask - math.log1p(
                    previous_valid_l1.ask_units / QUANTITY_SCALE
                )
            current_observation = _L1Observation(
                mid_ticks=mid_ticks,
                bid_tick=bid_tick,
                ask_tick=ask_tick,
                bid_units=bid_units,
                ask_units=ask_units,
                event_ms=book.last_event_ms,
            )
            next_grid_l1_identity = current_observation.identity
            l1_carried = current_observation.identity == previous_grid_l1_identity
            l1[11] = age_ms / 1000.0
            l1[12] = 0.0
            l1[13] = policy.unavailable_sequence_gap_value
            l1[14] = float(l1_carried)
            l1[15] = 1.0
            next_previous_valid = current_observation

    reference_mid_ticks: float | None = mid_ticks if valid_l1 else None
    if (
        reference_mid_ticks is None
        and one_sided
        and not crossed
        and previous_valid_l1 is not None
        and 0 <= grid_ms - previous_valid_l1.event_ms <= policy.l1_max_age_ms
    ):
        reference_mid_ticks = previous_valid_l1.mid_ticks
    l2_valid = bool(
        reference_mid_ticks is not None
        and not crossed
        and age_ms is not None
        and 0 <= age_ms <= policy.l2_max_age_ms
        and (bids or asks)
    )
    l2_carried = l2_valid and book.revision == previous_grid_l2_revision
    if l2_valid:
        assert reference_mid_ticks is not None
        _fill_l2_features(
            l2,
            l2_mask,
            bids=bids,
            asks=asks,
            mid_ticks=reference_mid_ticks,
            age_seconds=age_ms / 1000.0,
            carried=l2_carried,
            sequence_gap=policy.unavailable_sequence_gap_value,
        )
        l2_valid = bool(l2_mask.any())
    rejected = crossed
    return (
        _MicroFrame(
            l1=l1,
            l1_valid=valid_l1,
            l1_mid_ticks=mid_ticks,
            l2=l2,
            l2_level_mask=l2_mask,
            l2_valid=l2_valid,
            l1_carried=l1_carried and valid_l1,
            l2_carried=l2_carried,
            approximate=valid_l1 or l2_valid,
            one_sided=one_sided,
            rejected=rejected,
        ),
        next_previous_valid,
        next_grid_l1_identity,
        book.revision if l2_valid else None,
    )


def _fill_l2_features(
    output: np.ndarray,
    mask: np.ndarray,
    *,
    bids: Sequence[tuple[int, int]],
    asks: Sequence[tuple[int, int]],
    mid_ticks: float,
    age_seconds: float,
    carried: bool,
    sequence_gap: float,
) -> None:
    sides = (bids[:10], asks[:10])
    totals = [sum(size for _, size in levels) for levels in sides]
    for side_index, levels in enumerate(sides):
        total = totals[side_index]
        if total <= 0:
            continue
        cumulative = 0
        for rank, (price_tick, size_units) in enumerate(levels):
            if size_units <= 0:
                continue
            cumulative += size_units
            paired = sides[1 - side_index]
            paired_units = paired[rank][1] if rank < len(paired) else 0
            pair_sum = size_units + paired_units
            better_gap = 0
            if rank > 0:
                better_tick = levels[rank - 1][0]
                better_gap = (
                    better_tick - price_tick
                    if side_index == 0
                    else price_tick - better_tick
                )
            row = output[side_index, rank]
            row[0] = price_tick - mid_ticks
            row[1] = better_gap
            row[2] = math.log1p(size_units / QUANTITY_SCALE)
            row[3] = math.log1p(cumulative / QUANTITY_SCALE)
            row[4] = size_units / total
            row[5] = cumulative / total
            if rank < len(paired) and pair_sum:
                bid_units = size_units if side_index == 0 else paired_units
                ask_units = paired_units if side_index == 0 else size_units
                row[6] = (bid_units - ask_units) / pair_sum
            else:
                row[6] = 0.0
            row[7] = age_seconds
            row[8] = 0.0
            row[9] = sequence_gap
            row[10] = float(carried)
            row[11] = 1.0
            mask[side_index, rank] = True


def _build_example(
    frames: tuple[_MicroFrame, ...],
    *,
    cutoff_ms: int,
    context: _SessionBars,
    trade_minutes: dict[int, MinuteTape],
    source: SourceContract,
    l2_artifact: SourceArtifact,
    policy: OkxCompilerPolicy,
) -> EngineeredExample | None:
    minute_index = (cutoff_ms - context.day_start_ms) // MINUTE_MS
    if minute_index <= 0 or minute_index > len(context.tokens):
        return None
    latest_bar = context.tokens[minute_index - 1]
    if (
        latest_bar.ema_after is None
        or latest_bar.valid_bar_count < policy.ema_min_valid_bars
    ):
        return None

    anchor_ticks: float | None = None
    anchor_source = ""
    if frames[-1].l1_valid:
        anchor_ticks = frames[-1].l1_mid_ticks
        anchor_source = "l1_mid_event_time_proxy"
    else:
        for offset in range(policy.bar_anchor_max_age_slots + 1):
            index = minute_index - 1 - offset
            if index >= 0 and context.tokens[index].source_valid:
                token = context.tokens[index]
                close_tick = token.close_tick
                age_ms = cutoff_ms - token.close_ms
                if (
                    close_tick is not None
                    and 0 <= age_ms <= policy.bar_anchor_max_age_slots * MINUTE_MS
                ):
                    anchor_ticks = float(close_tick)
                    anchor_source = "finalized_trade_bar_close"
                    break
    if anchor_ticks is None:
        return None

    bars = np.zeros((64, 16), dtype=np.float32)
    bars_mask = np.zeros(64, dtype=np.bool_)
    first_slot = minute_index - 64
    for slot in range(64):
        token_index = first_slot + slot
        if token_index < 0 or token_index >= len(context.tokens):
            continue
        token = context.tokens[token_index]
        if token.valid:
            bars[slot] = token.values
            bars[slot, 12] = max(0.0, (cutoff_ms - token.close_ms) / 1000.0)
            bars_mask[slot] = True

    l1 = np.stack([frame.l1 for frame in frames]).astype(np.float32, copy=True)
    l1_mask = np.asarray([frame.l1_valid for frame in frames], dtype=np.bool_)
    l1[l1_mask, 0] -= anchor_ticks
    l1[~l1_mask] = 0.0
    l2 = np.stack([frame.l2 for frame in frames]).astype(np.float32, copy=False)
    l2_level_mask = np.stack([frame.l2_level_mask for frame in frames])
    l2_snapshot_mask = np.asarray([frame.l2_valid for frame in frames], dtype=np.bool_)
    l2[~l2_level_mask] = 0.0

    quality = np.zeros(16, dtype=np.float32)
    quality[0] = bars_mask.mean()
    quality[1] = l1_mask.mean()
    quality[2] = l2_snapshot_mask.mean()
    quality[3] = l2_level_mask.mean()
    quality[4] = _fraction_true(
        frame.l1_carried for frame in frames if frame.l1_valid
    )
    quality[5] = _fraction_true(
        frame.l2_carried for frame in frames if frame.l2_valid
    )
    quality[6] = _fraction_true(
        frame.approximate for frame in frames if frame.l1_valid
    )
    quality[7] = _fraction_true(
        frame.approximate for frame in frames if frame.l2_valid
    )
    quality[8] = policy.unavailable_sequence_gap_value if l1_mask.any() else 0.0
    quality[9] = (
        policy.unavailable_sequence_gap_value if l2_snapshot_mask.any() else 0.0
    )
    quality[10] = 0.0
    quality[11] = 0.0
    quality[12] = _fraction_true(frame.one_sided for frame in frames)
    quality[13] = _fraction_true(frame.rejected for frame in frames)
    quality[14] = float(anchor_source.startswith("l1"))
    quality[15] = minute_index / (24 * 60)

    targets = np.zeros((5, 5), dtype=np.float32)
    target_mask = np.zeros((5, 5), dtype=np.bool_)
    presence = np.zeros((5, 2), dtype=np.float32)
    presence_mask = np.ones((5, 2), dtype=np.bool_)
    log_volume_reference = float(latest_bar.ema_after)
    for horizon in range(5):
        minute_start = cutoff_ms + horizon * MINUTE_MS
        summary = trade_minutes.get(minute_start)
        target_mask[horizon, 3] = True
        if summary is None:
            targets[horizon, 3] = -log_volume_reference
            continue
        volume = summary.total_units / QUANTITY_SCALE
        targets[horizon, 0] = summary.poc_tick - anchor_ticks
        targets[horizon, 3] = math.log1p(volume) - log_volume_reference
        targets[horizon, 4] = (summary.buy_units - summary.sell_units) / summary.total_units
        target_mask[horizon, 0] = True
        target_mask[horizon, 4] = summary.total_units > 0
        if summary.buy_poc_tick is not None:
            targets[horizon, 1] = summary.buy_poc_tick - anchor_ticks
            target_mask[horizon, 1] = True
            presence[horizon, 0] = 1.0
        if summary.sell_poc_tick is not None:
            targets[horizon, 2] = summary.sell_poc_tick - anchor_ticks
            target_mask[horizon, 2] = True
            presence[horizon, 1] = 1.0

    batch = ChmtQBatch(
        bars=torch.from_numpy(bars).unsqueeze(0),
        bars_mask=torch.from_numpy(bars_mask).unsqueeze(0),
        l1=torch.from_numpy(l1).unsqueeze(0),
        l1_mask=torch.from_numpy(l1_mask).unsqueeze(0),
        l2=torch.from_numpy(l2).unsqueeze(0),
        l2_level_mask=torch.from_numpy(l2_level_mask).unsqueeze(0),
        l2_snapshot_mask=torch.from_numpy(l2_snapshot_mask).unsqueeze(0),
        quality=torch.from_numpy(quality).unsqueeze(0),
        targets=torch.from_numpy(targets).unsqueeze(0),
        target_mask=torch.from_numpy(target_mask).unsqueeze(0),
        presence_labels=torch.from_numpy(presence).unsqueeze(0),
        presence_mask=torch.from_numpy(presence_mask).unsqueeze(0),
    )
    cutoff = datetime.fromtimestamp(cutoff_ms / 1000, tz=UTC)
    example_id = f"okx-btc-usdt-{cutoff.strftime('%Y%m%dT%H%M%SZ')}"
    provenance = _canonical_sha256(
        {
            "compiler_policy_sha256": policy.sha256,
            "cutoff_ms": cutoff_ms,
            "l2_archive_sha256": l2_artifact.sha256,
            "source_audit_sha256": source.source_audit_sha256,
            "source_manifest_sha256": source.source_manifest_sha256,
        }
    )
    metadata = ExampleMetadata(
        example_id=example_id,
        session_id=f"okx-btc-usdt-utc-{context.day.isoformat()}",
        feature_cutoff_ns=cutoff_ms * 1_000_000,
        decision_ns=(cutoff_ms + policy.decision_lag_ms) * 1_000_000,
        dependency_start_ns=context.day_start_ms * 1_000_000,
        dependency_end_ns=(cutoff_ms + 5 * MINUTE_MS + policy.label_grace_ms) * 1_000_000,
        instrument_id=source.instrument_id,
        anchor_source=anchor_source,
        anchor_price=anchor_ticks * TICK_SIZE,
        tick_size=TICK_SIZE,
        log_volume_reference=log_volume_reference,
        provenance_sha256=provenance,
    )
    return EngineeredExample(metadata=metadata, batch=batch)


def _parse_l2_level(
    level: Any,
    stats: L2CompileStats | None = None,
) -> tuple[int, int]:
    if not isinstance(level, list) or len(level) != 3:
        raise ValueError("L2 level must contain price, size, and order count")
    if not isinstance(level[0], str) or not isinstance(level[1], str):
        raise ValueError("L2 price and size must use decimal strings")
    price, price_error = _parse_l2_price(level[0])
    size, size_error = (
        (0, 0.0) if level[1] == "0" else _parse_source_fixed(level[1], 8)
    )
    if stats is not None and price_error:
        stats.price_quantization_count += 1
        stats.max_price_quantization_error_units = max(
            stats.max_price_quantization_error_units,
            price_error,
        )
    if stats is not None and size_error:
        stats.size_quantization_count += 1
        stats.max_size_quantization_error_units = max(
            stats.max_size_quantization_error_units,
            size_error,
        )
    order_count = int(level[2])
    if price <= 0 or size < 0 or order_count < 0:
        raise ValueError("L2 price/order count must be positive and size non-negative")
    if (size == 0) != (order_count == 0):
        raise ValueError("L2 size and order count zero states disagree")
    return price, size


@lru_cache(maxsize=32_768)
def _parse_l2_price(value: str) -> tuple[int, float]:
    return _parse_source_fixed(value, 1)


def _parse_fixed(value: bytes | str, decimals: int) -> int:
    units, _ = _parse_fixed_value(value, decimals, allow_source_quantization=False)
    return units


def _parse_source_fixed(value: bytes | str, decimals: int) -> tuple[int, float]:
    return _parse_fixed_value(value, decimals, allow_source_quantization=True)


def _parse_fixed_value(
    value: bytes | str,
    decimals: int,
    *,
    allow_source_quantization: bool,
) -> tuple[int, float]:
    if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 18:
        raise ValueError("decimals must be an integer between zero and eighteen")
    exact_units = _parse_common_exact_fixed(value, decimals)
    if exact_units is not None:
        return exact_units, 0.0
    raw = value if isinstance(value, bytes) else value.encode("ascii")
    raw = raw.strip()
    if not raw or len(raw) > 128:
        raise ValueError("empty decimal value")
    sign = -1 if raw.startswith(b"-") else 1
    if raw[:1] in {b"-", b"+"}:
        raw = raw[1:]
    exponent = 0
    lower = raw.lower()
    if b"e" in lower:
        mantissa, exponent_raw = lower.split(b"e", 1)
        exponent = int(exponent_raw)
        if not -100 <= exponent <= 100:
            raise ValueError("decimal exponent is outside the supported range")
    else:
        mantissa = lower
    if b"." in mantissa:
        whole, fraction = mantissa.split(b".", 1)
    else:
        whole, fraction = mantissa, b""
    if not whole:
        whole = b"0"
    digits_raw = whole + fraction
    if not digits_raw.isdigit():
        raise ValueError(f"invalid decimal value: {value!r}")
    digits = int(digits_raw)
    power = decimals + exponent - len(fraction)
    if not -100 <= power <= 100:
        raise ValueError("scaled decimal exponent is outside the supported range")
    if power >= 0:
        units = digits * POWERS_OF_TEN[power]
        error_units = 0.0
    else:
        divisor = POWERS_OF_TEN[-power]
        units, remainder = divmod(digits, divisor)
        if remainder:
            round_up = remainder * 2 >= divisor
            distance = divisor - remainder if round_up else remainder
            if (
                not allow_source_quantization
                or distance * SOURCE_UNIT_TOLERANCE_DENOMINATOR
                > divisor * SOURCE_UNIT_TOLERANCE_NUMERATOR
            ):
                raise ValueError(
                    f"decimal is not representable within source tolerance at scale 1e-{decimals}"
                )
            if round_up:
                units += 1
            error_units = distance / divisor
        else:
            error_units = 0.0
    return sign * units, error_units


def _parse_common_exact_fixed(value: bytes | str, decimals: int) -> int | None:
    if not isinstance(value, (bytes, str)) or not value:
        return None
    dot = b"." if isinstance(value, bytes) else "."
    plus = b"+" if isinstance(value, bytes) else "+"
    minus = b"-" if isinstance(value, bytes) else "-"
    raw = value
    sign = 1
    if raw[:1] in {plus, minus}:
        sign = -1 if raw[:1] == minus else 1
        raw = raw[1:]
    point = raw.find(dot)
    if point < 0:
        digits = raw
        fraction_length = 0
    else:
        if raw.find(dot, point + 1) >= 0:
            return None
        whole = raw[:point]
        fraction = raw[point + 1 :]
        if len(fraction) > decimals:
            return None
        digits = whole + fraction
        fraction_length = len(fraction)
    if not digits or not digits.isdigit():
        return None
    return sign * int(digits) * POWERS_OF_TEN[decimals - fraction_length]


def _poc_tick(volume_by_tick: dict[int, int]) -> int:
    if not volume_by_tick:
        raise ValueError("POC requires at least one price bucket")
    return max(volume_by_tick, key=lambda tick: (volume_by_tick[tick], tick))


def _fraction_true(values: Iterable[bool]) -> float:
    total = 0
    matched = 0
    for value in values:
        total += 1
        matched += int(bool(value))
    return matched / total if total else 0.0


def _json_loads(value: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(value)
    return json.loads(value)


def _consume(stream: BinaryIO) -> None:
    while stream.read(8 * 1024 * 1024):
        pass


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_relative(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("source relative path must be a non-empty string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("source relative path must not be absolute")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"source artifact path is unsafe: {relative}")
    return resolved


def _require_source_coverage(
    source: SourceContract,
    *,
    start: date,
    end: date,
) -> None:
    available = {(artifact.feed, artifact.day) for artifact in source.artifacts}
    required: list[tuple[str, date]] = []
    current = start
    while current <= end:
        required.append(("l2-400", current))
        required.append(("trades", current))
        current += timedelta(days=1)
    required.append(("trades", end + timedelta(days=1)))
    missing = [
        f"{feed}:{day.isoformat()}"
        for feed, day in required
        if (feed, day) not in available
    ]
    if missing:
        raise ValueError("source coverage is incomplete: " + ", ".join(missing))


def _validate_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _day_start_ms(value: date) -> int:
    return int(datetime.combine(value, datetime_time.min, tzinfo=UTC).timestamp() * 1000)


def _timestamp_ms_iso(value: int | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value / 1000, tz=UTC).isoformat().replace("+00:00", "Z")


def _trade_partition_start_ms(value: date) -> int:
    return _day_start_ms(value) - 8 * 60 * 60 * 1000


def _trade_partition_end_ms(value: date) -> int:
    return _trade_partition_start_ms(value) + DAY_MS


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an ISO date (YYYY-MM-DD)") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=_parse_date, required=True)
    parser.add_argument("--end", type=_parse_date, required=True)
    parser.add_argument("--rows-per-shard", type=int, default=256)
    parser.add_argument("--rows-per-record-batch", type=int, default=64)
    parser.add_argument("--compression", choices=("zstd", "lz4", "none"), default="zstd")
    parser.add_argument("--max-examples", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest, report = compile_okx_dataset(
            source_root=args.source,
            destination=args.output,
            start=args.start,
            end=args.end,
            rows_per_shard=args.rows_per_shard,
            rows_per_record_batch=args.rows_per_record_batch,
            compression=None if args.compression == "none" else args.compression,
            max_examples=args.max_examples,
        )
        print(
            json.dumps(
                {
                    "dataset_id": manifest.dataset_id,
                    "rows": manifest.row_count,
                    "status": report["status"],
                    "training_ready": manifest.capabilities.training_ready,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
