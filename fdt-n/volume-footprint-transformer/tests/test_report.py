from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from daxalgo_footprint_transformer.cli import main
from daxalgo_footprint_transformer.config import (
    BAR_FEATURE_NAMES,
    ROW_FEATURE_NAMES,
    TARGET_NAMES,
    EncodingConfig,
    ModelConfig,
)
from daxalgo_footprint_transformer.report import (
    PendingResultsError,
    generate_pdf_report,
    load_report_bundle,
)


def _distribution(examples: int, loss: float, offset: float) -> dict[str, object]:
    cells: dict[str, object] = {}
    for horizon in range(1, 9):
        targets: dict[str, object] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            base = offset + horizon * 0.002 + target_index * 0.003
            metric: dict[str, object] = {
                "examples": examples,
                "pinball_normalized": {
                    "q10": base + 0.010,
                    "q50": base + 0.012,
                    "q90": base + 0.011,
                },
                "median_mae_normalized": base + 0.050,
                "median_bias_normalized": base - 0.020,
                "central_80_coverage": 0.78 + target_index * 0.005,
                "mean_interval_width_normalized": 0.30 + horizon * 0.012,
            }
            if target_name == "poc_offset":
                metric["poc_directional_hit_rate"] = 0.51 + horizon * 0.001
            targets[target_name] = metric
        cells[str(horizon)] = targets
    return {
        "loss": loss,
        "examples": examples,
        "pooled": {
            "pinball_normalized": {"q10": 0.030, "q50": 0.032, "q90": 0.031},
            "median_mae_normalized": 0.080 + offset,
            "median_bias_normalized": 0.002,
            "central_80_coverage": 0.80,
            "mean_interval_width_normalized": 0.36,
            "poc_directional_hit_rate": 0.52,
        },
        "median_mae_normalized": {
            target: 0.070 + index * 0.005 + offset for index, target in enumerate(TARGET_NAMES)
        },
        "central_80_coverage": {
            target: 0.78 + index * 0.005 for index, target in enumerate(TARGET_NAMES)
        },
        "by_horizon_target": cells,
    }


def _point_metrics(examples: int, offset: float) -> dict[str, object]:
    cells: dict[str, object] = {}
    for horizon in range(1, 9):
        targets: dict[str, object] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            metric: dict[str, object] = {
                "examples": examples,
                "median_mae_normalized": offset + 0.080 + target_index * 0.004,
                "median_bias_normalized": -0.010 + target_index * 0.002,
            }
            if target_name == "poc_offset":
                metric["poc_directional_hit_rate"] = 0.50
            targets[target_name] = metric
        cells[str(horizon)] = targets
    return {
        "examples": examples,
        "pooled": {
            "median_mae_normalized": 0.10 + offset,
            "median_bias_normalized": 0.0,
            "poc_directional_hit_rate": 0.50,
        },
        "median_mae_normalized": {
            target: offset + 0.080 + index * 0.004 for index, target in enumerate(TARGET_NAMES)
        },
        "median_bias_normalized": {
            target: -0.010 + index * 0.002 for index, target in enumerate(TARGET_NAMES)
        },
        "by_horizon_target": cells,
    }


def _comparison(offset: float) -> dict[str, object]:
    cells: dict[str, object] = {}
    for horizon in range(1, 9):
        targets: dict[str, object] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            point = offset + (horizon - 4) * 0.002 + target_index * 0.001
            targets[target_name] = {
                "mae_difference_normalized": point,
                "ci_95_lower": point - 0.020,
                "ci_95_upper": point + 0.020,
            }
        cells[str(horizon)] = targets
    return {
        "difference": "model_median_mae_minus_baseline_mae",
        "confidence_level": 0.95,
        "resamples": 2_000,
        "seed": 17,
        "blocks": 5,
        "block_definition": "contiguous_240_example_blocks",
        "by_horizon_target": cells,
    }


def _write_bundle(
    tmp_path: Path,
    runs: tuple[tuple[int, float, float], ...] = ((17, 0.040, 0.052), (23, 0.038, 0.061)),
) -> dict[str, object]:
    root = tmp_path / "repo"
    paper = root / "docs/research/volume-footprint-transformer/research-paper.md"
    ledger = root / "docs/research/volume-footprint-transformer/evidence-ledger.md"
    package_marker = root / "tools/volume-footprint-transformer"
    paper.parent.mkdir(parents=True)
    package_marker.mkdir(parents=True)
    paper.write_text("# FDT-N: Synthetic layout source\n", encoding="utf-8")
    ledger.write_text("# FDT-N evidence and claim ledger\n", encoding="utf-8")

    config = EncodingConfig().to_dict()
    bars = 1_200
    counts = np.full(bars, 2, dtype=np.int64)
    offsets = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    row_features = np.zeros((int(offsets[-1]), len(ROW_FEATURE_NAMES)), dtype=np.float32)
    row_features[0, 0] = -2.0
    row_features[1, 0] = 2.0
    bar_values = np.zeros((bars, 19), dtype=np.float64)
    bar_values[:, 10] = np.log1p(1_000.0)
    bar_values[:, 11:13] = 0.5
    bar_values[:, 17] = 2.0
    bar_values[:, 18] = 1.0
    dataset = root / "tmp/volume-footprint-transformer/frozen.npz"
    dataset.parent.mkdir(parents=True)
    np.savez_compressed(
        dataset,
        schema_version=np.asarray(2, dtype=np.int64),
        config=json.dumps(config, sort_keys=True),
        timestamps_ms=np.arange(bars, dtype=np.int64) * 60_000,
        row_features=row_features,
        row_offsets=offsets,
        bar_values=bar_values,
    )
    valid_windows = bars - config["lookback_bars"] - config["horizon_bars"] + 1
    archive_sizes = [1] * 13 + [374_843_570 - 13]
    source_files = [
        {
            "path": f"BTCUSDT-trades-2024-01-{index + 1:02d}.zip",
            "bytes": size,
            "sha256": hashlib.sha256(f"archive-{index}".encode()).hexdigest(),
        }
        for index, size in enumerate(archive_sizes)
    ]
    manifest_data = {
        "schema_version": 2,
        "source": "Binance Data Vision spot daily individual trades",
        "source_files": source_files,
        "encoding": config,
        "row_feature_names": list(ROW_FEATURE_NAMES),
        "bar_feature_names": list(BAR_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "bars": bars,
        "encoded_price_rows": int(offsets[-1]),
        "maximum_rows_per_bar": 2,
        "valid_windows": valid_windows,
        "first_timestamp_ms": 0,
        "last_timestamp_ms": (bars - 1) * 60_000,
    }
    manifest = dataset.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(manifest_data, sort_keys=True), encoding="utf-8")

    purge = config["lookback_bars"] + config["horizon_bars"]
    valid = np.arange(config["lookback_bars"] - 1, bars - config["horizon_bars"])
    train_stop = int(valid_windows * 0.70)
    validation_stop = int(valid_windows * 0.85)
    split_arrays = {
        "train": valid[: train_stop - purge],
        "validation": valid[train_stop + purge : validation_stop - purge],
        "test": valid[validation_stop + purge :],
    }
    split_counts = {
        "train_examples": len(split_arrays["train"]),
        "validation_examples": len(split_arrays["validation"]),
        "test_examples": len(split_arrays["test"]),
        "purge_bars": purge,
    }

    baseline_persistence = _point_metrics(split_counts["test_examples"], 0.020)
    baseline_ridge = _point_metrics(split_counts["test_examples"], 0.010)
    paths: list[Path] = []
    for seed, validation_loss, test_loss in runs:
        run_dir = root / f"tmp/volume-footprint-transformer/seed-{seed}"
        run_dir.mkdir(parents=True)
        checkpoint = run_dir / "best.pt"
        checkpoint.write_bytes(f"checkpoint-{seed}".encode())
        ridge = run_dir / "direct_ridge_baseline.npz"
        np.savez_compressed(ridge, coefficients=np.ones((1,), dtype=np.float32))
        np.savez_compressed(run_dir / "splits.npz", **split_arrays)
        history = [
            {
                "epoch": epoch,
                "train_loss": 0.090 / epoch + seed * 1e-6,
                "validation_loss": validation_loss + (3 - epoch) * 0.004,
                "learning_rate": 3e-4 / epoch,
            }
            for epoch in range(1, 4)
        ]
        result = {
            "schema_version": 2,
            "architecture": "FDT-N",
            "parameter_count": 371_861,
            "device": "cpu",
            "elapsed_seconds": 12.5,
            "best_epoch": 3,
            "epochs_completed": 3,
            "global_steps": 30,
            "splits": split_counts,
            "encoding_config": config,
            "model_config": ModelConfig().to_dict(),
            "train_config": {
                "epochs": 3,
                "batch_size": 64,
                "learning_rate": 3e-4,
                "weight_decay": 1e-3,
                "warmup_fraction": 0.08,
                "gradient_clip": 1.0,
                "patience": 8,
                "seed": seed,
                "workers": 0,
                "row_token_reference": 64,
                "ridge_alpha": 1.0,
                "bootstrap_resamples": 2_000,
            },
            "validation": _distribution(
                split_counts["validation_examples"], validation_loss, seed * 1e-5
            ),
            "test": _distribution(split_counts["test_examples"], test_loss, seed * 1e-5),
            "baselines": {
                "persistence": {
                    "target_space": "encoded_normalized_targets",
                    "test": baseline_persistence,
                },
                "direct_ridge": {
                    "target_space": "encoded_normalized_targets",
                    "feature_strategy": "latest_mean_trend_bar_aggregates",
                    "feature_count": 69,
                    "alpha": 1.0,
                    "training_examples": split_counts["train_examples"],
                    "artifact": str(ridge),
                    "test": baseline_ridge,
                },
            },
            "comparisons": {
                "model_vs_persistence": _comparison(-0.010 + seed * 1e-5),
                "model_vs_direct_ridge": _comparison(-0.003 + seed * 1e-5),
            },
            "history": history,
            "checkpoint": str(checkpoint),
        }
        result_path = run_dir / "results.json"
        result_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
        paths.append(result_path)

    primary_seed, primary_validation_loss, _ = min(runs, key=lambda item: (item[1], item[0]))
    primary_result = root / f"tmp/volume-footprint-transformer/seed-{primary_seed}/results.json"
    primary_checkpoint = primary_result.parent / "best.pt"
    deployment_dir = root / "tmp/volume-footprint-transformer/deployment"
    deployment_dir.mkdir(parents=True)
    onnx_path = deployment_dir / "fdt-n.onnx"
    onnx_path.write_bytes(b"synthetic-onnx-layout-fixture")

    def file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    metadata = {
        "schema_version": 2,
        "model_kind": "fdt-n-footprint-distribution-transformer",
        "model_version": file_hash(primary_checkpoint)[:16],
        "coordinate": {
            "instrument_key": "BTCUSDT",
            "source": "Binance",
            "interval_seconds": config["interval_seconds"],
            "row_size": config["row_size"],
        },
        "minimum_history_bars": config["lookback_bars"],
        "maximum_horizon_bars": config["horizon_bars"],
        "encoding_config": config,
        "model_config": ModelConfig().to_dict(),
        "row_feature_names": list(ROW_FEATURE_NAMES),
        "bar_feature_names": list(BAR_FEATURE_NAMES),
        "target_names": list(TARGET_NAMES),
        "quantiles": [0.1, 0.5, 0.9],
        "parameter_count": 371_861,
        "checkpoint_sha256": file_hash(primary_checkpoint),
        "onnx_sha256": file_hash(onnx_path),
        "checkpoint_epoch": 3,
        "checkpoint_validation_loss": primary_validation_loss,
        "lineage": {
            "dataset": {"file": dataset.name, "sha256": file_hash(dataset)},
            "dataset_manifest": {"file": manifest.name, "sha256": file_hash(manifest)},
            "training_results": {
                "file": primary_result.name,
                "sha256": file_hash(primary_result),
            },
        },
        "parity": {
            "provider": "CPUExecutionProvider",
            "sample": {
                "selection": "earliest_window_with_minimum_width_at_or_above_p95",
                "window_max_rows": 2,
                "p95_window_max_rows": 2.0,
                "valid_windows": valid_windows,
            },
            "maximum_absolute_error": 1e-6,
            "mean_absolute_error": 1e-7,
            "tolerance": 1e-4,
            "passed": True,
            "cpu_latency_ms": {"runs": 50, "p50": 1.0, "p95": 1.2, "p99": 1.4},
            "onnx_version": "synthetic",
            "onnxruntime_version": "synthetic",
        },
        "environment": {
            "python": "synthetic",
            "platform": "synthetic",
            "torch": "synthetic",
            "numpy": "synthetic",
        },
        "runtime_status": "shadow-only-research",
        "creates_order_signals": False,
    }
    metadata_path = deployment_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")

    return {
        "root": root,
        "paper": paper,
        "ledger": ledger,
        "dataset": dataset,
        "manifest": manifest,
        "results": paths,
        "metadata": metadata_path,
    }


def test_missing_results_fail_as_pending_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "output.pdf"
    with pytest.raises(PendingResultsError, match="metrics are pending"):
        generate_pdf_report(
            [tmp_path / "missing-results.json"],
            tmp_path / "missing-manifest.json",
            tmp_path / "missing.npz",
            tmp_path / "paper.md",
            tmp_path / "ledger.md",
            tmp_path / "metadata.json",
            output,
            tmp_path / "work",
            enforce_repo_paths=False,
        )
    assert not output.exists()


def test_legacy_results_fail_as_pending(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, ((17, 0.040, 0.052),))
    result_path = bundle["results"][0]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["schema_version"] = 1
    result_path.write_text(json.dumps(result), encoding="utf-8")

    with pytest.raises(PendingResultsError, match="schema-v2"):
        load_report_bundle(
            bundle["results"],
            bundle["manifest"],
            bundle["dataset"],
            bundle["paper"],
            bundle["ledger"],
            bundle["metadata"],
        )


def test_primary_selection_uses_validation_not_test(tmp_path: Path) -> None:
    bundle = _write_bundle(
        tmp_path,
        (
            (17, 0.040, 0.030),
            (23, 0.035, 0.090),
        ),
    )
    loaded = load_report_bundle(
        bundle["results"],
        bundle["manifest"],
        bundle["dataset"],
        bundle["paper"],
        bundle["ledger"],
        bundle["metadata"],
    )

    assert loaded.primary.seed == 23
    assert loaded.primary.data["test"]["loss"] == 0.090
    assert {run.seed for run in loaded.runs} == {17, 23}


def test_clipping_audit_covers_inputs_and_targets(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path, ((17, 0.040, 0.052),))
    loaded = load_report_bundle(
        bundle["results"],
        bundle["manifest"],
        bundle["dataset"],
        bundle["paper"],
        bundle["ledger"],
        bundle["metadata"],
    )

    row = loaded.corpus.clipping_audit[0]
    assert row.method == "encoded_boundary_saturation"
    assert (row.lower_hits, row.upper_hits, row.observations) == (1, 1, 2_400)
    targets = [item for item in loaded.corpus.clipping_audit if item.scope == "target"]
    assert {item.feature for item in targets} == set(TARGET_NAMES)
    assert all(item.lower_hits == 0 and item.upper_hits == 0 for item in targets)


def test_seed_stability_rejects_mixed_training_budgets(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    changed_path = bundle["results"][0]
    changed = json.loads(changed_path.read_text(encoding="utf-8"))
    changed["train_config"]["batch_size"] = 32
    changed_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="non-seed training configuration"):
        load_report_bundle(
            bundle["results"],
            bundle["manifest"],
            bundle["dataset"],
            bundle["paper"],
            bundle["ledger"],
            bundle["metadata"],
        )


def test_metadata_must_bind_to_validation_selected_results(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    metadata = json.loads(bundle["metadata"].read_text(encoding="utf-8"))
    metadata["lineage"]["training_results"]["sha256"] = "0" * 64
    bundle["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(PendingResultsError, match="validation-selected run"):
        load_report_bundle(
            bundle["results"],
            bundle["manifest"],
            bundle["dataset"],
            bundle["paper"],
            bundle["ledger"],
            bundle["metadata"],
        )


def test_unwatermarked_final_paper_requires_three_seeds(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    with pytest.raises(PendingResultsError, match="at least three declared seeds"):
        generate_pdf_report(
            bundle["results"],
            bundle["manifest"],
            bundle["dataset"],
            bundle["paper"],
            bundle["ledger"],
            bundle["metadata"],
            tmp_path / "should-not-exist.pdf",
            tmp_path / "figures",
            enforce_repo_paths=False,
        )


def test_rendered_layout_reports_every_seed_without_placeholders(tmp_path: Path) -> None:
    bundle = _write_bundle(tmp_path)
    output = tmp_path / "fdt-n-layout-test.pdf"
    metadata = generate_pdf_report(
        bundle["results"],
        bundle["manifest"],
        bundle["dataset"],
        bundle["paper"],
        bundle["ledger"],
        bundle["metadata"],
        output,
        tmp_path / "figures",
        enforce_repo_paths=False,
        watermark="SYNTHETIC LAYOUT TEST - NOT A RESULT",
    )

    reader = PdfReader(str(output))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert metadata["primary_seed"] == 23
    assert metadata["seed_count"] == 2
    assert metadata["pages"] == len(reader.pages)
    assert len(reader.pages) >= 40
    assert "Abstract and evidence grammar" in text
    assert "Volume footprints from first principles" in text
    assert "Worked footprint and forecast example" in text
    assert "Prior negative evidence - CHMT-Q" in text
    assert "Input clipping and row-bound saturation" in text
    assert "Exact row pre-clip loss is explicitly unimplemented" in text
    assert "Selected model artifact and runtime evidence" in text
    assert "Seed 17 held-out cells" in text
    assert "[S04]" in text and "[S28]" in text
    assert "Held-out scorecard - seed 17" in text
    assert "Held-out scorecard - seed 23" in text
    assert "minimum validation loss" in text
    assert "SYNTHETIC LAYOUT TEST - NOT A RESULT" in text
    assert "[PENDING]" not in text


def test_cli_returns_two_when_results_are_pending(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "report",
            "--results",
            str(tmp_path / "missing.json"),
            "--dataset-manifest",
            str(tmp_path / "missing.manifest.json"),
            "--dataset",
            str(tmp_path / "missing.npz"),
            "--model-metadata",
            str(tmp_path / "missing-metadata.json"),
        ]
    )

    assert code == 2
    assert "metrics are pending" in capsys.readouterr().err
