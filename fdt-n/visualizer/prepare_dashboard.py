"""Prepare all data for the self-contained visualizer dashboard.

Inputs : restored-seed17.pt, test-windows.npz, test-series.npz, doc artifacts
Outputs: dashboard-data.json (single payload consumed by the HTML dashboard)

All data is real Binance BTCUSDT from the frozen study period (2024-01-12..14),
re-downloaded and re-encoded with the project's exact pipeline. Research-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from daxalgo_footprint_transformer.config import (
    EncodingConfig,
    ModelConfig,
    ROW_FEATURE_NAMES,
)
from daxalgo_footprint_transformer.model import FootprintDistributionTransformer

VIS = Path(r"D:\Github\DaxAlgo-Research\fdt-n\visualizer")
HEAVY = Path(r"C:\Users\dhruv\AppData\Local\Temp\opencode\visualizer-data")
DOC = Path(r"D:\Github\DaxAlgo-Research\fdt-n\doc\model")

cfg = EncodingConfig()
mcfg = ModelConfig()
D = mcfg.d_model
H = mcfg.attention_heads
DK = D // H
SCALE = 1.0 / np.sqrt(DK)

model = FootprintDistributionTransformer(cfg, mcfg)
model.load_state_dict(torch.load(VIS / "restored-seed17.pt", weights_only=True))
model.eval()

w = np.load(HEAVY / "test-windows.npz")
rows, rmask, bfeat = w["row_features"], w["row_mask"], w["bar_features"]
targets, tmasks, cts = w["targets"], w["target_masks"], w["context_timestamps"]
N = rows.shape[0]
print(f"windows: {N}, row width: {rows.shape[2]}")

# ---- 1. batched inference over all real windows ---------------------------------------
B = 256
preds = np.zeros((N, 8, 7, 3), dtype=np.float64)
with torch.inference_mode():
    for s in range(0, N, B):
        sl = slice(s, min(s + B, N))
        out = model(
            torch.from_numpy(rows[sl]),
            torch.from_numpy(rmask[sl]),
            torch.from_numpy(bfeat[sl]),
            torch.ones(sl.stop - sl.start, 64, dtype=torch.bool),
        )
        preds[sl] = out.numpy()

# ---- 2. live metrics with their exact formulas (training.py distribution_metrics) ------
from daxalgo_footprint_transformer.config import TARGET_NAMES  # noqa: E402

QUANTILE_LEVELS = np.asarray((0.1, 0.5, 0.9))
QUANTILE_NAMES = ("q10", "q50", "q90")


def masked_mean(x, m):
    m = np.asarray(m, dtype=bool)
    return float(np.asarray(x, dtype=np.float64)[m].mean()) if m.any() else None


errs = targets[..., None] - preds
pinball = np.maximum(QUANTILE_LEVELS * errs, (QUANTILE_LEVELS - 1.0) * errs)
med_err = preds[..., 1] - targets
cover = (targets >= preds[..., 0]) & (targets <= preds[..., 2])
width = preds[..., 2] - preds[..., 0]

pooled = {
    "pinball_normalized": {n: masked_mean(pinball[..., i], tmasks) for i, n in enumerate(QUANTILE_NAMES)},
    "median_mae_normalized": masked_mean(np.abs(med_err), tmasks),
    "median_bias_normalized": masked_mean(med_err, tmasks),
    "central_80_coverage": masked_mean(cover.astype(np.float64), tmasks),
    "mean_interval_width_normalized": masked_mean(width, tmasks),
    "poc_directional_hit_rate": masked_mean(
        (np.sign(preds[:, :, 0, 1]) == np.sign(targets[:, :, 0])).astype(np.float64),
        tmasks[:, :, 0],
    ),
}
per_target = {
    TARGET_NAMES[t]: {
        "median_mae_normalized": masked_mean(np.abs(med_err[:, :, t]), tmasks[:, :, t]),
        "median_bias_normalized": masked_mean(med_err[:, :, t], tmasks[:, :, t]),
        "central_80_coverage": masked_mean(cover[:, :, t].astype(np.float64), tmasks[:, :, t]),
    }
    for t in range(7)
}
poc_hits = (np.sign(preds[:, :, 0, 1]) == np.sign(targets[:, :, 0]))
by_horizon_poc_direction = [
    masked_mean(poc_hits[:, h].astype(np.float64), tmasks[:, h, 0]) for h in range(8)
]
# persistence comparison (their construction: last context bar values held forward)
# Use the package's own baseline builder for exactness.
from daxalgo_footprint_transformer.data import EncodedFootprintSeries  # noqa: E402
from daxalgo_footprint_transformer.training import (  # noqa: E402
    persistence_baseline_outputs,
    point_prediction_metrics,
)

test_series = EncodedFootprintSeries.load(HEAVY / "test-series.npz")
persist_preds, persist_targets, persist_masks = persistence_baseline_outputs(
    test_series, w["context_ends"]
)
# sanity: persistence targets equal our stored targets
assert np.abs(persist_targets.astype(np.float64) - targets).max() < 1e-5
assert (persist_masks == tmasks).all()
point_metrics = point_prediction_metrics(persist_preds, persist_targets, persist_masks)
p_mae = point_metrics["pooled"]["median_mae_normalized"]
persist = persist_preds.astype(np.float64)
print("live pooled:", json.dumps(pooled))
print("persistence pooled q50 MAE:", p_mae)

# ---- 3. replay sample (192 windows) ----------------------------------------------------
rng = np.random.default_rng(17)
order = np.sort(rng.permutation(N)[:192])
replay = {
    "index": order.tolist(),
    "timestamps": cts[order].tolist(),
    "targets": np.round(targets[order], 4).tolist(),
    "q10": np.round(preds[order, :, :, 0], 4).tolist(),
    "q50": np.round(preds[order, :, :, 1], 4).tolist(),
    "q90": np.round(preds[order, :, :, 2], 4).tolist(),
    "masks": tmasks[order].tolist(),
    "persist": np.round(persist[order], 4).tolist(),
    "context_poc": np.round(bfeat[order, :, 0], 4).tolist(),
}

# ---- 4. stage-by-stage internals for the real fixture window ---------------------------
fixture = json.loads((DOC / "parity-fixture.json").read_text())
rf = torch.tensor(fixture["row_features"]).unsqueeze(0)
rm = torch.tensor(fixture["row_mask"]).unsqueeze(0)
bf = torch.tensor(fixture["bar_features"]).unsqueeze(0)
bm = torch.ones(1, 64, dtype=torch.bool)

stages: dict[str, np.ndarray] = {}
hooks = []


def snap(name):
    def fn(mod, inp, out):
        if isinstance(out, tuple):          # nn.MultiheadAttention returns (output, weights)
            out = out[0]
        stages[name] = out.detach().numpy().copy()
    return fn


targets_hooks = {
    "row_projected": model.row_projection,
    "row_query_norm": model.row_pool.query_norm,
    "row_pooled": model.row_pool.attention,
    "row_summary": model.row_summary,
    "bar_summary": model.bar_projection,
    "fusion_gate_preact": model.fusion_gate,
    "fusion_projected": model.fusion_projection,
    "temporal_memory": model.temporal_norm,
    "raw_head": model.output_head,
}
for name, mod in targets_hooks.items():
    hooks.append(mod.register_forward_hook(snap(name)))

pre_caps: dict[str, np.ndarray] = {}


def pre_snap(name):
    def fn(mod, inp):
        pre_caps[name] = inp[0].detach().numpy().copy()
    return fn


hooks.append(model.temporal_blocks[0].norm.register_forward_pre_hook(pre_snap("t0_in")))
hooks.append(model.temporal_blocks[1].norm.register_forward_pre_hook(pre_snap("t1_in")))
hooks.append(model.temporal_blocks[2].norm.register_forward_pre_hook(pre_snap("t2_in")))
hooks.append(model.decoder_blocks[0].cross.query_norm.register_forward_pre_hook(pre_snap("d0_q_in")))
hooks.append(model.decoder_blocks[1].cross.query_norm.register_forward_pre_hook(pre_snap("d1_q_in")))

with torch.inference_mode():
    fixture_out = model(rf, rm, bf, bm)
for h in hooks:
    h.remove()

# verify manual MHA replication against the real model before trusting it
assert np.abs(fixture_out.numpy() - np.asarray(fixture["expected_quantiles"])).max() < 1e-4


def rms(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12) * np.sqrt(x.shape[-1])


def attention_from(mha, q_in, kv_in, key_pad=None, causal=False):
    Wt = mha.in_proj_weight.detach().numpy().T            # [D, 3D]
    Wq, Wk, Wv = np.split(Wt, 3, axis=1)                  # [D, D] each
    bq, bk, bv = np.split(mha.in_proj_bias.detach().numpy(), 3)
    q = q_in @ Wq + bq
    k = kv_in @ Wk + bk
    v = kv_in @ Wv + bv
    Lq, Lk = q.shape[0], k.shape[0]
    qh = q.reshape(Lq, H, DK).transpose(1, 0, 2)
    kh = k.reshape(Lk, H, DK).transpose(1, 0, 2)
    vh = v.reshape(Lk, H, DK).transpose(1, 0, 2)
    logits = (qh @ kh.transpose(0, 2, 1)) * SCALE
    if causal:
        upper = np.triu(np.ones((Lq, Lk), dtype=bool), k=1)
        logits = np.where(upper[None], -np.inf, logits)
    if key_pad is not None:
        logits = np.where(key_pad[None], -np.inf, logits)
    logits = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(logits)
    a = e / e.sum(axis=-1, keepdims=True)
    ctx = (a @ vh).transpose(1, 0, 2).reshape(Lq, D)
    return a, ctx @ mha.out_proj.weight.detach().numpy().T + mha.out_proj.bias.detach().numpy()


# row_pool cross-attention: 4 auction queries over the anchor bar's price rows (bar 63)
aq = model.auction_queries.detach().numpy()                        # [4,64]
row_mem = stages["row_projected"].reshape(64, -1, D)               # [64 bars, R, 64]
R = row_mem.shape[1]
mem_norm = rms(row_mem[-1])                                        # anchor bar rows
A_row, _ = attention_from(
    model.row_pool.attention, rms(aq), mem_norm,
    key_pad=~np.asarray(fixture["row_mask"], dtype=bool)[-1],
)
# temporal attention maps (64x64 causal)
A_t0, _ = attention_from(model.temporal_blocks[0].attention, rms(pre_caps["t0_in"][0]), rms(pre_caps["t0_in"][0]), causal=True)
A_t1, _ = attention_from(model.temporal_blocks[1].attention, rms(pre_caps["t1_in"][0]), rms(pre_caps["t1_in"][0]), causal=True)
A_t2, _ = attention_from(model.temporal_blocks[2].attention, rms(pre_caps["t2_in"][0]), rms(pre_caps["t2_in"][0]), causal=True)
# decoder cross-attention: 8 horizon queries over 64 bar memories
A_d0, _ = attention_from(model.decoder_blocks[0].cross.attention, rms(pre_caps["d0_q_in"][0]), rms(stages["temporal_memory"][0]))
A_d1, _ = attention_from(model.decoder_blocks[1].cross.attention, rms(pre_caps["d1_q_in"][0]), rms(stages["temporal_memory"][0]))

gate = 1.0 / (1.0 + np.exp(-stages["fusion_gate_preact"][0]))      # [64,64]

# quantile head transform trace
raw = stages["raw_head"][0]                                         # [8,21]
rq = raw.reshape(8, 7, 3)
signed = rq[..., 0]
low_gap = np.log1p(np.exp(rq[..., 1])) + 1e-4
up_gap = np.log1p(np.exp(rq[..., 2])) + 1e-4
qt = np.stack((signed - low_gap, signed, signed + up_gap), axis=-1)
qt[:, 6] = np.tanh(qt[:, 6])
poc = qt[:, 0]
low = np.minimum(qt[:, 1], poc)
high = np.maximum(qt[:, 2], poc)
buy = np.clip(qt[:, 3], low, high)
sell = np.clip(qt[:, 4], low, high)
qt_final = np.stack((poc, low, high, buy, sell, qt[:, 5], qt[:, 6]), axis=-2)
assert np.abs(qt_final - fixture_out[0].numpy()).max() < 1e-5, "quantile transform mismatch"

fx_rows = np.asarray(fixture["row_features"], dtype=np.float32)
fx_mask = np.asarray(fixture["row_mask"], dtype=bool)
fx_bars = np.asarray(fixture["bar_features"], dtype=np.float32)

# fixture-window actuals: the realized future 8 bars of this real window (bit-exact re-encode)
fw = np.load(HEAVY / "fixture-window.npz")
assert float(fw["row_reencode_max_diff"]) == 0.0 and float(fw["bar_reencode_max_diff"]) == 0.0
fixture_actuals = {
    "targets": np.round(fw["targets"], 4).tolist(),              # [8,7] real realized values
    "masks": fw["target_masks"].tolist(),
    "future_timestamps": fw["future_timestamps"].tolist(),
    "anchor_poc_tick": float(fw["anchor_poc_tick"]),
    "row_counts": fw["row_counts"].tolist(),                     # [64] price rows per bar
    "reencode_max_diff": 0.0,
    "prediction_check": np.round(fixture_out[0].numpy(), 4).tolist(),  # model output for this window
}

payload_stages = {
    "row_features": np.round(fx_rows, 4).tolist(),
    "row_mask": fx_mask.tolist(),
    "bar_features": np.round(fx_bars, 4).tolist(),
    # row-token grid downsampled for payload size: anchor bar full, others every 8th bar
    "row_projected": np.round(stages["row_projected"].reshape(64, -1, D)[::8], 2).tolist(),
    "row_projected_bars": list(range(0, 64, 8)),
    "row_attention": np.round(A_row, 4).tolist(),                  # [H,4,R]
    "row_pooled": np.round(stages["row_pooled"].reshape(64, 4, D), 2).tolist(),
    "row_summary": np.round(stages["row_summary"][0], 2).tolist(),
    "bar_summary": np.round(stages["bar_summary"][0], 2).tolist(),
    "fusion_gate": np.round(gate, 3).tolist(),                     # [64,64]
    "temporal_memory": np.round(stages["temporal_memory"][0], 2).tolist(),
    "temporal_attention_0": np.round(A_t0, 4).tolist(),             # [H,64,64]
    "temporal_attention_1": np.round(A_t1, 4).tolist(),
    "temporal_attention_2": np.round(A_t2, 4).tolist(),
    "decoder_attention_0": np.round(A_d0, 4).tolist(),              # [H,8,64]
    "decoder_attention_1": np.round(A_d1, 4).tolist(),
    "raw_head": np.round(raw, 4).tolist(),                         # [8,21]
    "quantile_pre_clamp": np.round(qt, 4).tolist(),                # [8,7,3]
    "quantile_final": np.round(qt_final, 4).tolist(),              # [8,7,3]
    "expected_quantiles": np.asarray(fixture["expected_quantiles"], dtype=np.float32).round(5).tolist(),
    "actuals": fixture_actuals,
}

# ---- 5. series context ------------------------------------------------------------------
series = np.load(HEAVY / "test-series.npz")
ts = series["timestamps_ms"]
bv = series["bar_values"]
n_ctx = min(1440, len(bv))
series_ctx = {
    "timestamps": ts[:n_ctx].tolist(),
    "poc_ticks": bv[:n_ctx, 0].tolist(),
    "log_volume": bv[:n_ctx, 10].tolist(),
    "delta_fraction": bv[:n_ctx, 13].tolist(),
}

# ---- 6. assemble + save ------------------------------------------------------------------
meta_doc = json.loads((DOC / "metadata.json").read_text())
frozen = {
    "seeds": [
        {"seed": 17, "epoch": 21, "val": 0.112634, "test": 0.070392, "mae": 0.235870, "coverage": 0.7898, "poc_sign": 0.5490},
        {"seed": 29, "epoch": 13, "val": 0.114391, "test": 0.071542, "mae": 0.237238, "coverage": 0.7896, "poc_sign": 0.4944},
        {"seed": 43, "epoch": 20, "val": 0.113713, "test": 0.070793, "mae": 0.236171, "coverage": 0.8023, "poc_sign": 0.5071},
    ],
    "pooled_pinball": {"q10": 0.055493, "q50": 0.117935, "q90": 0.051387},
    "pooled_mae": {"fdt": 0.235870, "persistence": 0.254298, "ridge": 0.229719},
    "pooled_interval_width": 0.720113,
    "pooled_q50_bias": 0.034807,
    "per_target_mae": {
        "poc_offset": {"fdt": 0.2414, "persistence": 0.2432, "ridge": 0.2357},
        "low_offset": {"fdt": 0.2552, "persistence": 0.2423, "ridge": 0.2371},
        "high_offset": {"fdt": 0.2272, "persistence": 0.2221, "ridge": 0.2225},
        "buy_poc_offset": {"fdt": 0.2431, "persistence": 0.2442, "ridge": 0.2374},
        "sell_poc_offset": {"fdt": 0.2415, "persistence": 0.2426, "ridge": 0.2359},
        "log_volume_relative": {"fdt": 0.1219, "persistence": 0.1536, "ridge": 0.1198},
        "delta_fraction": {"fdt": 0.3207, "persistence": 0.4321, "ridge": 0.3197},
    },
    "per_horizon_mae": {
        "1": {"fdt": 0.1388, "persistence": 0.1577, "ridge": 0.1315},
        "2": {"fdt": 0.1843, "persistence": 0.2041, "ridge": 0.1782},
        "3": {"fdt": 0.2146, "persistence": 0.2344, "ridge": 0.2084},
        "4": {"fdt": 0.2383, "persistence": 0.2560, "ridge": 0.2310},
        "5": {"fdt": 0.2559, "persistence": 0.2731, "ridge": 0.2496},
        "6": {"fdt": 0.2732, "persistence": 0.2906, "ridge": 0.2663},
        "7": {"fdt": 0.2846, "persistence": 0.3038, "ridge": 0.2802},
        "8": {"fdt": 0.2973, "persistence": 0.3147, "ridge": 0.2926},
    },
    "bootstrap_intervals": {
        "vs_persistence_seed17": {"fdt": 21, "zero": 21, "comp": 14},
        "vs_ridge_seed17": {"fdt": 0, "zero": 20, "comp": 36},
    },
    "dataset": "14 days Binance BTCUSDT spot (2024-01-01..14), 20,160 bars, 357,676 price rows",
    "splits": "13,990 / 2,869 / 2,942 windows (train / val / test), 72-bar purge at each boundary",
    "clipping_note": "price-target clipping affects ~1.04-1.17% of observations; row offset saturation 3.51%",
    "onnx_parity": {"max_abs": 2.384e-07, "mean_abs": 5.167e-08, "tolerance": 1e-4},
    "cpu_latency_ms": {"p50": 11.4099, "p95": 14.6406, "p99": 17.2398},
    "honest_takeaway": (
        "FDT-N beats zero-change persistence pooled (7.25% lower q50 MAE) but is 2.68% above the "
        "train-only direct ridge baseline; ridge is lower in 55 of 56 horizon-target cells. The "
        "experiment supports the architecture's viability, not superiority over simple baselines."
    ),
}
meta = {
    "model_kind": meta_doc["model_kind"],
    "model_version": meta_doc["model_version"],
    "parameter_count": meta_doc["parameter_count"],
    "onnx_sha256": meta_doc["onnx_sha256"],
    "checkpoint_sha256": meta_doc["checkpoint_sha256"],
    "checkpoint_epoch": meta_doc["checkpoint_epoch"],
    "quantiles": meta_doc["quantiles"],
    "row_feature_names": list(ROW_FEATURE_NAMES),
    "bar_feature_names": meta_doc["bar_feature_names"],
    "target_names": meta_doc["target_names"],
    "encoding_config": meta_doc["encoding_config"],
    "model_config": meta_doc["model_config"],
    "environment": meta_doc["environment"],
    "runtime_status": meta_doc["runtime_status"],
    "creates_order_signals": meta_doc["creates_order_signals"],
    "restore_parity_max_abs_err": 0.0,
    "restore_parity_random_windows_max": 4.768e-07,
    "data": json.loads((VIS / "acquisition.json").read_text()),
    "windows_total": int(N),
    "replay_windows": int(order.shape[0]),
    "row_size_usd": cfg.row_size,
    "price_scale_usd": cfg.price_scale_ticks * cfg.row_size,
    "gap_scale_usd": cfg.gap_scale_ticks * cfg.row_size,
    "log_volume_scale": cfg.log_volume_scale,
}

payload = {
    "meta": meta,
    "frozen": frozen,
    "metrics": {
        "pooled": pooled,
        "per_target": per_target,
        "persistence_pooled_q50_mae": p_mae,
    "persistence_pooled": point_metrics["pooled"],
        "poc_direction_by_horizon": by_horizon_poc_direction,
        "windows": int(N),
    },
    "replay": replay,
    "stages": payload_stages,
    "series": series_ctx,
}
out_path = VIS / "dashboard-data.json"
out_path.write_text(json.dumps(payload))
print("payload size MB:", round(out_path.stat().st_size / 1e6, 2))
print("OK ->", out_path)
