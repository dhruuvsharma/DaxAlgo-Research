"""Acquire real Binance BTCUSDT trades for the frozen test period and encode to footprint series.

Downloads the last 3 days of the frozen 14-day study window (2024-01-12..14) using the
package's own downloader, builds footprint bars via the exact sealed-bar pipeline, and saves:

  test-series.npz      EncodedFootprintSeries (their schema, loadable with EncodedFootprintSeries.load)
  test-windows.npz     ~259 sliding windows: inputs, targets, actuals, timestamps
  acquisition.json     provenance: urls, sizes, sha256 of each archive, bar/row counts

The frozen study trained on 2024-01-01..11 and tested on the 2024-01-12..14 tail, so
these re-downloaded windows are out-of-sample for the frozen weights -- same regime as
the frozen test split. Real Binance data, no synthetic fallback needed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
import os
import tempfile
from pathlib import Path

import numpy as np

from daxalgo_footprint_transformer.config import EncodingConfig
from daxalgo_footprint_transformer.data import (
    EncodedFootprintSeries,
    build_footprint_bars,
    download_binance_trades,
    iter_binance_trades,
)

VIS = Path(__file__).resolve().parent
HEAVY = Path(os.environ.get("FDTN_VIS_DATA", Path(tempfile.gettempdir()) / "fdtn-visualizer-data"))
HEAVY.mkdir(parents=True, exist_ok=True)
RAW = HEAVY / "binance-raw"
RAW.mkdir(parents=True, exist_ok=True)

cfg = EncodingConfig()
start, end = date(2024, 1, 12), date(2024, 1, 14)
print(f"downloading Binance BTCUSDT trades {start} .. {end} ...")
archives = download_binance_trades("BTCUSDT", start, end, RAW)
print("archives:", [a.name for a in archives])

print("building sealed footprint bars (exact pipeline) ...")
bars = build_footprint_bars(iter_binance_trades(archives, cfg.quantity_scale), cfg)
print(f"bars: {len(bars)}, total rows: {sum(len(b.rows) for b in bars)}")
if len(bars) < cfg.lookback_bars + cfg.horizon_bars:
    raise RuntimeError("not enough bars for even one window")

series = EncodedFootprintSeries.from_bars(bars, cfg)
print(f"series: {len(series)} bars, {len(series.row_offsets) - 1} encoded")

series_path = HEAVY / "test-series.npz"
series.save(series_path)

# ---- sliding windows ------------------------------------------------------------
indices = series.valid_window_indices()
print(f"valid window context_end indices: {len(indices)} first={indices[0]} last={indices[-1]}")

rows_list, row_masks, bars_arr, targets_arr, t_masks, ts_list, actuals = [], [], [], [], [], [], []
for ce in indices:
    r, rm, b, t, tm = series.make_window(int(ce))
    rows_list.append(r)
    row_masks.append(rm)
    bars_arr.append(b)
    targets_arr.append(t)
    t_masks.append(tm)
    ts_list.append(series.timestamps_ms[ce])

row_width = max(r.shape[1] for r in rows_list)
rows = np.zeros((len(rows_list), cfg.lookback_bars, row_width, 11), dtype=np.float32)
row_mask = np.zeros((len(rows_list), cfg.lookback_bars, row_width), dtype=bool)
for i, (r, rm) in enumerate(zip(rows_list, row_masks)):
    rows[i, :, : r.shape[1]] = r
    row_mask[i, :, : r.shape[1]] = rm

np.savez_compressed(
    HEAVY / "test-windows.npz",
    row_features=rows,
    row_mask=row_mask,
    bar_features=np.asarray(bars_arr, dtype=np.float32),
    targets=np.asarray(targets_arr, dtype=np.float32),
    target_masks=np.asarray(t_masks, dtype=bool),
    context_timestamps=np.asarray(ts_list, dtype=np.int64),
    context_ends=indices.astype(np.int64),
)

manifest = {
    "source": "Binance Data Vision spot daily trades (re-download of frozen study period)",
    "symbol": "BTCUSDT",
    "days": [f"2024-01-{d:02d}" for d in (12, 13, 14)],
    "archives": [
        {
            "name": a.name,
            "bytes": a.stat().st_size,
            "sha256": hashlib.sha256(a.read_bytes()).hexdigest(),
        }
        for a in archives
    ],
    "encoding": cfg.to_dict(),
    "bars": len(series),
    "encoded_price_rows": int(series.row_offsets[-1]),
    "windows": int(len(indices)),
    "role": "out-of-sample for frozen seed-17 weights (same tail as frozen test split)",
    "research_only": True,
    "creates_order_signals": False,
}
(VIS / "acquisition.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print("windows saved:", len(indices))
print("OK ->", HEAVY / "test-series.npz", ",", HEAVY / "test-windows.npz")

# ---- fixture-window actuals ---------------------------------------------------------
# The frozen parity fixture is the real window at context_end=14113 of the original
# 14-day series (2024-01-01 00:00 start) => Jan 10 19:13 UTC. Its 8 realized future
# bars lie inside Jan 10, so re-download Jan 10..11 and re-encode the same window.
# Raw archives are staged in the temp data directory; everything raw is deleted after encoding.
TMP_RAW = HEAVY / "binance-raw-fixture"
TMP_RAW.mkdir(parents=True, exist_ok=True)
start2, end2 = date(2024, 1, 10), date(2024, 1, 11)
print(f"downloading Binance BTCUSDT trades {start2} .. {end2} (fixture window) ...")
archives2 = download_binance_trades("BTCUSDT", start2, end2, TMP_RAW)
bars2 = build_footprint_bars(iter_binance_trades(archives2, cfg.quantity_scale), cfg)
series2 = EncodedFootprintSeries.from_bars(bars2, cfg)
series2.save(HEAVY / "fixture-series.npz")
for a in archives2:  # free space immediately
    a.unlink(missing_ok=True)
    Path(str(a) + ".CHECKSUM").unlink(missing_ok=True)
local_ce = 14113 - 9 * 1440  # Jan 10 00:00 UTC is minute 12960 of the original series
print("fixture local context_end:", local_ce, "bars in fixture series:", len(series2))
r2, rm2, b2, t2, tm2 = series2.make_window(local_ce)

fixture = json.loads((VIS.parent / "doc" / "model" / "parity-fixture.json").read_text())
dr = np.abs(np.asarray(r2, dtype=np.float32) - np.asarray(fixture["row_features"], dtype=np.float32))
width2 = r2.shape[1]
fr = np.asarray(fixture["row_features"], dtype=np.float32)
dw = dr[:, : fr.shape[1]]
# fixture rows are a subset width (187); compare overlapping region and masks
db = np.abs(b2 - np.asarray(fixture["bar_features"], dtype=np.float32))
print(f"fixture re-encode check: row max diff {dw.max():.3e} (width {fr.shape[1]} vs {width2}), "
      f"bar max diff {db.max():.3e}")
row_counts = np.diff(series2.row_offsets)[local_ce - 63 : local_ce + 1]
np.savez_compressed(
    HEAVY / "fixture-window.npz",
    targets=t2,
    target_masks=tm2,
    row_counts=row_counts.astype(np.int64),
    context_timestamps=series2.timestamps_ms[local_ce - 63 : local_ce + 1],
    future_timestamps=series2.timestamps_ms[local_ce + 1 : local_ce + 9],
    anchor_poc_tick=series2.bar_values[local_ce, 0],
    row_reencode_max_diff=dw.max(),
    bar_reencode_max_diff=db.max(),
)
print("OK ->", HEAVY / "fixture-window.npz")

# free space: encoded artifacts are saved; raw archives no longer needed
for a in RAW.glob("*.zip*"):
    a.unlink()
print("raw archives removed")

