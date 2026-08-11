from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .config import BAR_FEATURE_NAMES, ROW_FEATURE_NAMES, TARGET_NAMES, EncodingConfig, ModelConfig
from .data import EncodedFootprintSeries
from .model import FootprintDistributionTransformer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> FootprintDistributionTransformer:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    encoding = EncodingConfig(**saved["encoding_config"])
    model_config = ModelConfig(**saved["model_config"])
    model = FootprintDistributionTransformer(encoding, model_config)
    model.load_state_dict(saved["model_state"])
    model.eval()
    return model


def _representative_window(series: EncodedFootprintSeries) -> tuple[int, dict[str, object]]:
    valid = series.valid_window_indices()
    row_counts = np.diff(series.row_offsets)
    lookback = series.config.lookback_bars
    widths = np.asarray(
        [int(row_counts[int(index) - lookback + 1 : int(index) + 1].max()) for index in valid],
        dtype=np.int64,
    )
    percentile = float(np.percentile(widths, 95))
    target = int(np.ceil(percentile))
    candidates = np.flatnonzero(widths >= target)
    selected_position = int(candidates[np.argmin(widths[candidates])])
    return int(valid[selected_position]), {
        "selection": "earliest_window_with_minimum_width_at_or_above_p95",
        "window_max_rows": int(widths[selected_position]),
        "p95_window_max_rows": percentile,
        "valid_windows": int(len(valid)),
    }


def export_onnx(
    checkpoint: Path,
    output_dir: Path,
    series: EncodedFootprintSeries | None = None,
    *,
    instrument_key: str | None = None,
    source: str | None = None,
    dataset_path: Path | None = None,
    dataset_manifest: Path | None = None,
    training_results: Path | None = None,
) -> dict[str, object]:
    model = load_checkpoint(checkpoint)
    cfg = model.encoding
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "fdt-n.onnx"
    sample_profile: dict[str, object] | None = None
    if series is not None:
        sample_index, sample_profile = _representative_window(series)
        sample_rows, sample_row_mask, sample_bar, _, _ = series.make_window(sample_index)
        row = torch.from_numpy(sample_rows).unsqueeze(0)
        row_mask = torch.from_numpy(sample_row_mask).unsqueeze(0)
        bar = torch.from_numpy(sample_bar).unsqueeze(0)
    else:
        row_width = min(cfg.max_rows, 64)
        row = torch.zeros(1, cfg.lookback_bars, row_width, len(ROW_FEATURE_NAMES))
        row_mask = torch.ones(1, cfg.lookback_bars, row_width, dtype=torch.bool)
        bar = torch.zeros(1, cfg.lookback_bars, len(BAR_FEATURE_NAMES))
    bar_mask = torch.ones(1, cfg.lookback_bars, dtype=torch.bool)
    torch.onnx.export(
        model,
        (row, row_mask, bar, bar_mask),
        onnx_path,
        input_names=["row_features", "row_mask", "bar_features", "bar_mask"],
        output_names=["quantiles"],
        dynamic_axes={
            "row_features": {0: "batch", 2: "rows"},
            "row_mask": {0: "batch", 2: "rows"},
            "bar_features": {0: "batch"},
            "bar_mask": {0: "batch"},
            "quantiles": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
    )
    parity: dict[str, object] | None = None
    if series is not None:
        index = sample_index
        row_np, row_mask_np, bar_np, _, _ = series.make_window(index)
        with torch.inference_mode():
            reference = model(
                torch.from_numpy(row_np).unsqueeze(0),
                torch.from_numpy(row_mask_np).unsqueeze(0),
                torch.from_numpy(bar_np).unsqueeze(0),
                torch.ones(1, cfg.lookback_bars, dtype=torch.bool),
            ).numpy()
        row_batch = row_np[None, ...]
        row_mask_batch = row_mask_np[None, ...]
        bar_batch = bar_np[None, ...]
        bar_mask_batch = np.ones((1, cfg.lookback_bars), dtype=np.bool_)
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exception:
            raise RuntimeError(
                "ONNX export parity requires the optional 'onnx' dependencies"
            ) from exception
        onnx.checker.check_model(onnx.load(onnx_path))
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        feed = {
            "row_features": row_batch,
            "row_mask": row_mask_batch,
            "bar_features": bar_batch,
            "bar_mask": bar_mask_batch,
        }
        runtime_output = session.run(["quantiles"], feed)[0]
        absolute_error = np.abs(runtime_output - reference)
        maximum_error = float(absolute_error.max())
        if maximum_error > 1e-4:
            raise RuntimeError(f"ONNX parity failed: maximum absolute error {maximum_error:.8g}")
        for _ in range(5):
            session.run(["quantiles"], feed)
        timings_ms: list[float] = []
        for _ in range(50):
            started = time.perf_counter()
            session.run(["quantiles"], feed)
            timings_ms.append((time.perf_counter() - started) * 1_000.0)
        fixture = {
            "context_end": index,
            "row_features": row_np.tolist(),
            "row_mask": row_mask_np.astype(np.int8).tolist(),
            "bar_features": bar_np.tolist(),
            "bar_mask": bar_mask_batch[0].astype(np.int8).tolist(),
            "expected_quantiles": reference.tolist(),
        }
        (output_dir / "parity-fixture.json").write_text(
            json.dumps(fixture, separators=(",", ":")), encoding="utf-8"
        )
        parity = {
            "provider": session.get_providers()[0],
            "sample": sample_profile,
            "maximum_absolute_error": maximum_error,
            "mean_absolute_error": float(absolute_error.mean()),
            "tolerance": 1e-4,
            "passed": True,
            "cpu_latency_ms": {
                "runs": len(timings_ms),
                "p50": float(np.percentile(timings_ms, 50)),
                "p95": float(np.percentile(timings_ms, 95)),
                "p99": float(np.percentile(timings_ms, 99)),
            },
            "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__,
        }
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    lineage: dict[str, object] = {}
    for name, path in (
        ("dataset", dataset_path),
        ("dataset_manifest", dataset_manifest),
        ("training_results", training_results),
    ):
        if path is not None:
            lineage[name] = {"file": path.name, "sha256": _sha256(path)}
    metadata = {
        "schema_version": 2,
        "model_kind": "fdt-n-footprint-distribution-transformer",
        "model_version": _sha256(checkpoint)[:16],
        "coordinate": {
            "instrument_key": instrument_key,
            "source": source,
            "interval_seconds": cfg.interval_seconds,
            "row_size": cfg.row_size,
        },
        "minimum_history_bars": cfg.lookback_bars,
        "maximum_horizon_bars": cfg.horizon_bars,
        "encoding_config": cfg.to_dict(),
        "model_config": model.config.to_dict(),
        "row_feature_names": list(ROW_FEATURE_NAMES),
        "bar_feature_names": list(BAR_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "quantiles": [0.1, 0.5, 0.9],
        "parameter_count": model.parameter_count,
        "checkpoint_sha256": _sha256(checkpoint),
        "onnx_sha256": _sha256(onnx_path),
        "checkpoint_epoch": saved.get("epoch"),
        "checkpoint_validation_loss": saved.get("validation_loss"),
        "lineage": lineage,
        "parity": parity,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "runtime_status": "shadow-only-research",
        "creates_order_signals": False,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata
