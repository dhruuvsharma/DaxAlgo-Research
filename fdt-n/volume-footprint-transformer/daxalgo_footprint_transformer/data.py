from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.error
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .config import (
    BAR_FEATURE_NAMES,
    ROW_FEATURE_NAMES,
    TARGET_NAMES,
    EncodingConfig,
)

DATASET_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FootprintRow:
    price_tick: int
    buy_volume: float
    sell_volume: float
    bid_imbalance: bool
    ask_imbalance: bool
    zero_bid: bool
    zero_ask: bool

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume


@dataclass(frozen=True)
class FootprintBar:
    start_ms: int
    end_ms: int
    rows: tuple[FootprintRow, ...]
    poc_tick: int
    buy_poc_tick: int
    sell_poc_tick: int
    volume_centroid_tick: float
    buy_centroid_tick: float
    sell_centroid_tick: float
    value_area_high_tick: int
    value_area_low_tick: int
    buy_volume: float
    sell_volume: float
    cumulative_delta: float
    stacked_buy: int
    stacked_sell: int
    quality: float = 1.0

    @property
    def total_volume(self) -> float:
        return self.buy_volume + self.sell_volume

    @property
    def delta(self) -> float:
        return self.buy_volume - self.sell_volume

    @property
    def high_tick(self) -> int:
        return self.rows[0].price_tick

    @property
    def low_tick(self) -> int:
        return self.rows[-1].price_tick


def _daterange(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_binance_trades(
    symbol: str,
    start: date,
    end: date,
    output_dir: Path,
    *,
    aggregate: bool = False,
) -> list[Path]:
    """Download official Binance Data Vision daily trade archives.

    The returned ZIP paths are chronological. Archives are streamed directly during compilation so
    the large CSV members never consume a second copy on disk. When Binance publishes a checksum
    sidecar it is verified before use. Individual trades are the default because they match the
    terminal's live ``@trade`` subscription; aggregate trades remain available for diagnostics.
    """

    symbol = symbol.upper()
    output_dir.mkdir(parents=True, exist_ok=True)
    archives: list[Path] = []
    dataset = "aggTrades" if aggregate else "trades"
    for day in _daterange(start, end):
        stem = f"{symbol}-{dataset}-{day.isoformat()}"
        base = f"https://data.binance.vision/data/spot/daily/{dataset}/{symbol}/{stem}.zip"
        archive = output_dir / f"{stem}.zip"
        checksum = output_dir / f"{stem}.zip.CHECKSUM"
        if not archive.exists():
            urllib.request.urlretrieve(base, archive)
        try:
            if not checksum.exists():
                urllib.request.urlretrieve(base + ".CHECKSUM", checksum)
            expected = checksum.read_text(encoding="utf-8").split()[0].lower()
            actual = _sha256(archive)
            if expected != actual:
                raise ValueError(f"checksum mismatch for {archive.name}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        archives.append(archive)
    return archives


def iter_binance_trades(
    source_paths: Sequence[Path], quantity_scale: float
) -> Iterator[tuple[int, float, float, bool]]:
    """Yield live-parity (timestamp, price, rounded size, buyer-is-maker) trade rows."""

    prior_ts = -1
    for path in source_paths:
        archive: zipfile.ZipFile | None = None
        text: io.TextIOWrapper | None = None
        stream: io.TextIOBase | None = None
        aggregate = "-aggTrades-" in path.name
        timestamp_index = 5 if aggregate else 4
        maker_index = 6 if aggregate else 5
        try:
            if path.suffix.lower() == ".zip":
                archive = zipfile.ZipFile(path)
                members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                if len(members) != 1:
                    raise ValueError(f"expected one CSV member in {path}")
                text = io.TextIOWrapper(archive.open(members[0]), encoding="utf-8", newline="")
                stream = text
            else:
                stream = path.open("r", encoding="utf-8", newline="")
            for fields in csv.reader(stream):
                if not fields or not fields[0].isdigit():
                    continue
                timestamp_ms = int(fields[timestamp_index])
                if timestamp_ms < prior_ts:
                    raise ValueError(f"timestamp reversal in {path}")
                prior_ts = timestamp_ms
                scaled_quantity = int(round(float(fields[2]) * quantity_scale))
                if scaled_quantity <= 0:
                    continue
                yield (
                    timestamp_ms,
                    float(fields[1]),
                    float(scaled_quantity),
                    (fields[maker_index].strip().lower() == "true"),
                )
        finally:
            if text is not None:
                text.close()
            elif stream is not None:
                stream.close()
            if archive is not None:
                archive.close()


def _longest_run(rows: Sequence[FootprintRow], attr: str) -> int:
    longest = current = 0
    for row in rows:
        if getattr(row, attr):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _value_area(rows: Sequence[FootprintRow], poc_index: int) -> tuple[int, int]:
    target = sum(row.total_volume for row in rows) * 0.70
    accumulated = rows[poc_index].total_volume
    high = low = poc_index
    while accumulated < target and (high > 0 or low < len(rows) - 1):
        above = rows[high - 1].total_volume if high > 0 else -1.0
        below = rows[low + 1].total_volume if low < len(rows) - 1 else -1.0
        if below > above:
            low += 1
            accumulated += rows[low].total_volume
        else:
            high -= 1
            accumulated += rows[high].total_volume
    return rows[high].price_tick, rows[low].price_tick


def _finalize_bar(
    start_ms: int,
    interval_ms: int,
    volumes: dict[int, list[float]],
    cumulative_delta_before: float,
    imbalance_ratio: float,
) -> FootprintBar | None:
    if not volumes:
        return None
    ticks = sorted(volumes, reverse=True)
    raw_rows: list[FootprintRow] = []
    for index, tick in enumerate(ticks):
        buy, sell = volumes[tick]
        buy_above = volumes[ticks[index - 1]][0] if index > 0 else 0.0
        sell_below = volumes[ticks[index + 1]][1] if index < len(ticks) - 1 else 0.0
        raw_rows.append(
            FootprintRow(
                price_tick=tick,
                buy_volume=buy,
                sell_volume=sell,
                bid_imbalance=sell > 0 and sell >= buy_above * imbalance_ratio,
                ask_imbalance=buy > 0 and buy >= sell_below * imbalance_ratio,
                zero_bid=sell == 0,
                zero_ask=buy == 0,
            )
        )
    rows = tuple(raw_rows)
    total = sum(row.total_volume for row in rows)
    buy_total = sum(row.buy_volume for row in rows)
    sell_total = sum(row.sell_volume for row in rows)
    poc_index = max(range(len(rows)), key=lambda i: rows[i].total_volume)
    buy_index = max(range(len(rows)), key=lambda i: rows[i].buy_volume)
    sell_index = max(range(len(rows)), key=lambda i: rows[i].sell_volume)
    value_high, value_low = _value_area(rows, poc_index)
    centroid = sum(row.price_tick * row.total_volume for row in rows) / total
    buy_centroid = (
        sum(row.price_tick * row.buy_volume for row in rows) / buy_total
        if buy_total
        else float(rows[poc_index].price_tick)
    )
    sell_centroid = (
        sum(row.price_tick * row.sell_volume for row in rows) / sell_total
        if sell_total
        else float(rows[poc_index].price_tick)
    )
    delta = buy_total - sell_total
    return FootprintBar(
        start_ms=start_ms,
        end_ms=start_ms + interval_ms,
        rows=rows,
        poc_tick=rows[poc_index].price_tick,
        buy_poc_tick=rows[buy_index].price_tick,
        sell_poc_tick=rows[sell_index].price_tick,
        volume_centroid_tick=centroid,
        buy_centroid_tick=buy_centroid,
        sell_centroid_tick=sell_centroid,
        value_area_high_tick=value_high,
        value_area_low_tick=value_low,
        buy_volume=buy_total,
        sell_volume=sell_total,
        cumulative_delta=cumulative_delta_before + delta,
        stacked_buy=_longest_run(rows, "ask_imbalance"),
        stacked_sell=_longest_run(rows, "bid_imbalance"),
    )


def build_footprint_bars(
    trades: Iterable[tuple[int, float, float, bool]], config: EncodingConfig
) -> list[FootprintBar]:
    """Aggregate chronological Binance trades into the same sealed-bar information the chart has."""

    config.validate()
    interval_ms = config.interval_seconds * 1000
    current_start: int | None = None
    volumes: dict[int, list[float]] = {}
    cumulative_delta = 0.0
    bars: list[FootprintBar] = []
    for timestamp_ms, price, quantity, buyer_is_maker in trades:
        start_ms = timestamp_ms - timestamp_ms % interval_ms
        if current_start is None:
            current_start = start_ms
        if start_ms != current_start:
            sealed = _finalize_bar(
                current_start,
                interval_ms,
                volumes,
                cumulative_delta,
                config.imbalance_ratio,
            )
            if sealed is not None:
                bars.append(sealed)
                cumulative_delta = sealed.cumulative_delta
            current_start = start_ms
            volumes = {}
        price_tick = int(math.floor(price / config.row_size + 0.5))
        side = volumes.setdefault(price_tick, [0.0, 0.0])
        if buyer_is_maker:
            side[1] += quantity
        else:
            side[0] += quantity
    if current_start is not None:
        sealed = _finalize_bar(
            current_start,
            interval_ms,
            volumes,
            cumulative_delta,
            config.imbalance_ratio,
        )
        if sealed is not None:
            bars.append(sealed)
    return bars


def resample_footprint_bars(
    bars: Sequence[FootprintBar],
    interval_seconds: int,
    *,
    imbalance_ratio: float = 3.0,
) -> list[FootprintBar]:
    """Merge complete aligned footprint bars without replaying the source trades."""

    if not bars:
        return []
    if interval_seconds < 1 or imbalance_ratio <= 0:
        raise ValueError("resample interval and imbalance ratio must be positive")
    base_intervals = {bar.end_ms - bar.start_ms for bar in bars}
    if len(base_intervals) != 1:
        raise ValueError("source footprint bars must use one interval")
    base_interval_ms = base_intervals.pop()
    target_interval_ms = interval_seconds * 1_000
    if target_interval_ms < base_interval_ms or target_interval_ms % base_interval_ms:
        raise ValueError("target interval must be an integer multiple of the source interval")
    expected_bars = target_interval_ms // base_interval_ms
    output: list[FootprintBar] = []
    group: list[FootprintBar] = []
    group_start: int | None = None

    def flush() -> None:
        if not group or group_start is None:
            return
        complete = (
            len(group) == expected_bars
            and group[0].start_ms == group_start
            and group[-1].end_ms == group_start + target_interval_ms
            and all(
                right.start_ms == left.end_ms for left, right in zip(group, group[1:], strict=False)
            )
        )
        if not complete:
            return
        volumes: dict[int, list[float]] = {}
        group_delta = 0.0
        for bar in group:
            group_delta += bar.delta
            for row in bar.rows:
                side = volumes.setdefault(row.price_tick, [0.0, 0.0])
                side[0] += row.buy_volume
                side[1] += row.sell_volume
        sealed = _finalize_bar(
            group_start,
            target_interval_ms,
            volumes,
            group[-1].cumulative_delta - group_delta,
            imbalance_ratio,
        )
        if sealed is not None:
            output.append(replace(sealed, quality=min(bar.quality for bar in group)))

    for bar in bars:
        target_start = bar.start_ms - bar.start_ms % target_interval_ms
        if group_start is None:
            group_start = target_start
        if target_start != group_start:
            flush()
            group = []
            group_start = target_start
        group.append(bar)
    flush()
    return output


@dataclass
class EncodedFootprintSeries:
    config: EncodingConfig
    timestamps_ms: np.ndarray
    row_features: np.ndarray
    row_offsets: np.ndarray
    bar_values: np.ndarray

    BAR_VALUE_NAMES = (
        "poc_tick",
        "buy_poc_tick",
        "sell_poc_tick",
        "volume_centroid_tick",
        "buy_centroid_tick",
        "sell_centroid_tick",
        "low_tick",
        "high_tick",
        "value_area_low_tick",
        "value_area_high_tick",
        "log_volume",
        "buy_fraction",
        "sell_fraction",
        "delta_fraction",
        "cumulative_delta",
        "stacked_buy",
        "stacked_sell",
        "row_count",
        "quality",
    )

    @classmethod
    def from_bars(
        cls, bars: Sequence[FootprintBar], config: EncodingConfig
    ) -> EncodedFootprintSeries:
        config.validate()
        if len(bars) < config.lookback_bars + config.horizon_bars:
            raise ValueError("not enough bars for one complete window")
        count = len(bars)
        encoded_rows: list[np.ndarray] = []
        row_offsets = np.zeros(count + 1, dtype=np.int64)
        bar_values = np.zeros((count, len(cls.BAR_VALUE_NAMES)), dtype=np.float64)
        timestamps = np.empty(count, dtype=np.int64)
        for i, bar in enumerate(bars):
            timestamps[i] = bar.start_ms
            if not bar.rows:
                raise ValueError(f"bar at {bar.start_ms} has no observed price rows")
            if len(bar.rows) > config.max_rows:
                raise ValueError(
                    f"bar at {bar.start_ms} has {len(bar.rows)} rows, exceeding the explicit "
                    f"safety bound {config.max_rows}"
                )
            total = max(bar.total_volume, 1.0)
            log_total = max(math.log1p(total), 1e-6)
            for row in bar.rows:
                row_total = row.total_volume
                encoded_rows.append(
                    np.asarray(
                        (
                            np.clip(
                                (row.price_tick - bar.poc_tick) / config.gap_scale_ticks,
                                -2.0,
                                2.0,
                            ),
                            row.buy_volume / total,
                            row.sell_volume / total,
                            row_total / total,
                            row.delta / total,
                            math.log1p(row.buy_volume) / log_total,
                            math.log1p(row.sell_volume) / log_total,
                            float(row.bid_imbalance),
                            float(row.ask_imbalance),
                            float(row.zero_bid),
                            float(row.zero_ask),
                        ),
                        dtype=np.float32,
                    )
                )
            row_offsets[i + 1] = len(encoded_rows)
            buy_fraction = bar.buy_volume / total
            sell_fraction = bar.sell_volume / total
            bar_values[i] = np.asarray(
                (
                    bar.poc_tick,
                    bar.buy_poc_tick,
                    bar.sell_poc_tick,
                    bar.volume_centroid_tick,
                    bar.buy_centroid_tick,
                    bar.sell_centroid_tick,
                    bar.low_tick,
                    bar.high_tick,
                    bar.value_area_low_tick,
                    bar.value_area_high_tick,
                    math.log1p(total),
                    buy_fraction,
                    sell_fraction,
                    bar.delta / total,
                    bar.cumulative_delta,
                    bar.stacked_buy,
                    bar.stacked_sell,
                    len(bar.rows),
                    bar.quality,
                ),
                dtype=np.float64,
            )
        row_features = np.stack(encoded_rows).astype(np.float32, copy=False)
        series = cls(config, timestamps, row_features, row_offsets, bar_values)
        series.validate()
        return series

    def __len__(self) -> int:
        return int(self.timestamps_ms.shape[0])

    def validate(self) -> None:
        if self.timestamps_ms.ndim != 1 or len(self.timestamps_ms) != self.bar_values.shape[0]:
            raise ValueError(
                "timestamps and bar values must describe the same one-dimensional series"
            )
        if self.bar_values.shape[1:] != (len(self.BAR_VALUE_NAMES),):
            raise ValueError("dataset bar-value schema does not match this encoder")
        if self.row_features.ndim != 2 or self.row_features.shape[1] != len(ROW_FEATURE_NAMES):
            raise ValueError("dataset row-feature schema does not match this encoder")
        if self.row_offsets.shape != (len(self) + 1,) or self.row_offsets[0] != 0:
            raise ValueError("row offsets must contain one zero-based boundary per bar")
        if np.any(np.diff(self.row_offsets) <= 0):
            raise ValueError("every footprint bar must retain at least one observed price row")
        if int(self.row_offsets[-1]) != self.row_features.shape[0]:
            raise ValueError("row offsets do not cover the exact flattened row tensor")
        if np.any(np.diff(self.timestamps_ms) <= 0):
            raise ValueError("footprint bars must be strictly chronological")
        if not np.isfinite(self.row_features).all() or not np.isfinite(self.bar_values).all():
            raise ValueError("encoded footprint values must be finite")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            schema_version=np.asarray(DATASET_SCHEMA_VERSION, dtype=np.int64),
            config=json.dumps(self.config.to_dict(), sort_keys=True),
            timestamps_ms=self.timestamps_ms,
            row_features=self.row_features,
            row_offsets=self.row_offsets,
            bar_values=self.bar_values,
        )

    @classmethod
    def load(cls, path: Path) -> EncodedFootprintSeries:
        with np.load(path, allow_pickle=False) as bundle:
            if "schema_version" not in bundle:
                raise ValueError(
                    "legacy footprint dataset must be rebuilt with the exact-row encoder"
                )
            schema_version = int(bundle["schema_version"])
            if schema_version != DATASET_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported footprint dataset schema {schema_version}; "
                    f"expected {DATASET_SCHEMA_VERSION}"
                )
            config = EncodingConfig(**json.loads(str(bundle["config"])))
            series = cls(
                config=config,
                timestamps_ms=bundle["timestamps_ms"],
                row_features=bundle["row_features"],
                row_offsets=bundle["row_offsets"],
                bar_values=bundle["bar_values"],
            )
        series.validate()
        return series

    def valid_window_indices(self) -> np.ndarray:
        cfg = self.config
        first = cfg.lookback_bars - 1
        last = len(self) - cfg.horizon_bars
        interval_ms = cfg.interval_seconds * 1000
        valid: list[int] = []
        for context_end in range(first, last):
            start = context_end - cfg.lookback_bars + 1
            stop = context_end + cfg.horizon_bars + 1
            if np.all(np.diff(self.timestamps_ms[start:stop]) == interval_ms):
                valid.append(context_end)
        return np.asarray(valid, dtype=np.int64)

    def make_window(
        self, context_end: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        start = context_end - cfg.lookback_bars + 1
        stop = context_end + 1
        if start < 0 or context_end + cfg.horizon_bars >= len(self):
            raise IndexError("window exceeds encoded series")
        timeline = self.timestamps_ms[start : context_end + cfg.horizon_bars + 1]
        if not np.all(np.diff(timeline) == cfg.interval_seconds * 1000):
            raise ValueError("window crosses a missing or irregular footprint interval")
        values = self.bar_values[start:stop]
        anchor_poc = self.bar_values[context_end, 0]
        reference_log_volume = float(np.median(values[-min(16, len(values)) :, 10]))
        reference_volume = math.expm1(reference_log_volume)
        base_cumulative = values[0, 14]
        bar = np.zeros((cfg.lookback_bars, len(BAR_FEATURE_NAMES)), dtype=np.float32)
        for j, v in enumerate(values):
            timestamp = int(self.timestamps_ms[start + j])
            minute = (timestamp // 60_000) % 1_440
            weekday = (timestamp // 86_400_000 + 3) % 7  # Unix epoch was a Thursday.
            row_count = max(v[17], 1.0)
            bar[j] = np.asarray(
                (
                    np.clip((v[0] - anchor_poc) / cfg.price_scale_ticks, -2.0, 2.0),
                    np.clip((v[1] - v[0]) / cfg.gap_scale_ticks, -2.0, 2.0),
                    np.clip((v[2] - v[0]) / cfg.gap_scale_ticks, -2.0, 2.0),
                    np.clip((v[3] - v[0]) / cfg.gap_scale_ticks, -2.0, 2.0),
                    np.clip((v[4] - v[0]) / cfg.gap_scale_ticks, -2.0, 2.0),
                    np.clip((v[5] - v[0]) / cfg.gap_scale_ticks, -2.0, 2.0),
                    min((v[0] - v[6]) / cfg.gap_scale_ticks, 2.0),
                    min((v[7] - v[0]) / cfg.gap_scale_ticks, 2.0),
                    min((v[0] - v[8]) / cfg.gap_scale_ticks, 2.0),
                    min((v[9] - v[0]) / cfg.gap_scale_ticks, 2.0),
                    np.clip((v[10] - reference_log_volume) / cfg.log_volume_scale, -2.0, 2.0),
                    v[11],
                    v[12],
                    v[13],
                    np.clip(
                        (v[14] - base_cumulative)
                        / max(reference_volume * cfg.lookback_bars, 1.0)
                        / cfg.cumulative_delta_scale,
                        -2.0,
                        2.0,
                    ),
                    min(v[15] / row_count, 1.0),
                    min(v[16] / row_count, 1.0),
                    min(math.log1p(v[17]) / math.log1p(cfg.max_rows), 1.0),
                    v[18],
                    math.sin(2.0 * math.pi * minute / 1_440.0),
                    math.cos(2.0 * math.pi * minute / 1_440.0),
                    math.sin(2.0 * math.pi * weekday / 7.0),
                    math.cos(2.0 * math.pi * weekday / 7.0),
                ),
                dtype=np.float32,
            )

        targets = np.zeros((cfg.horizon_bars, len(TARGET_NAMES)), dtype=np.float32)
        target_mask = np.ones_like(targets, dtype=np.bool_)
        for h in range(cfg.horizon_bars):
            v = self.bar_values[context_end + h + 1]
            targets[h] = np.asarray(
                (
                    np.clip((v[0] - anchor_poc) / cfg.price_scale_ticks, -2.0, 2.0),
                    np.clip((v[6] - anchor_poc) / cfg.price_scale_ticks, -2.0, 2.0),
                    np.clip((v[7] - anchor_poc) / cfg.price_scale_ticks, -2.0, 2.0),
                    np.clip((v[1] - anchor_poc) / cfg.price_scale_ticks, -2.0, 2.0),
                    np.clip((v[2] - anchor_poc) / cfg.price_scale_ticks, -2.0, 2.0),
                    np.clip((v[10] - reference_log_volume) / cfg.log_volume_scale, -2.0, 2.0),
                    np.clip(v[13], -1.0, 1.0),
                ),
                dtype=np.float32,
            )
            if v[11] <= 0:
                target_mask[h, 3] = False
            if v[12] <= 0:
                target_mask[h, 4] = False
        counts = self.row_offsets[start + 1 : stop + 1] - self.row_offsets[start:stop]
        row_width = int(counts.max())
        rows = np.zeros((cfg.lookback_bars, row_width, len(ROW_FEATURE_NAMES)), dtype=np.float32)
        row_mask = np.zeros((cfg.lookback_bars, row_width), dtype=np.bool_)
        for index, count in enumerate(counts):
            offset = int(self.row_offsets[start + index])
            width = int(count)
            rows[index, :width] = self.row_features[offset : offset + width]
            row_mask[index, :width] = True
        return rows, row_mask, bar, targets, target_mask


def write_dataset_manifest(
    path: Path,
    csv_paths: Sequence[Path],
    series: EncodedFootprintSeries,
    source_name: str = "Binance Data Vision spot daily trades",
) -> None:
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source": source_name,
        "source_files": [
            {"path": str(csv_path), "bytes": csv_path.stat().st_size, "sha256": _sha256(csv_path)}
            for csv_path in csv_paths
        ],
        "encoding": series.config.to_dict(),
        "row_feature_names": list(ROW_FEATURE_NAMES),
        "bar_feature_names": list(BAR_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "bars": len(series),
        "encoded_price_rows": int(series.row_offsets[-1]),
        "maximum_rows_per_bar": int(np.diff(series.row_offsets).max()),
        "valid_windows": int(len(series.valid_window_indices())),
        "first_timestamp_ms": int(series.timestamps_ms[0]),
        "last_timestamp_ms": int(series.timestamps_ms[-1]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
