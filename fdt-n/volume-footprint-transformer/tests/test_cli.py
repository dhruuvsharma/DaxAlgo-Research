from __future__ import annotations

from pathlib import Path

import pytest

from daxalgo_footprint_transformer import cli
from daxalgo_footprint_transformer.config import ModelConfig
from daxalgo_footprint_transformer.training import TrainConfig


def test_train_cli_passes_exact_tiny_preset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "dataset.npz"
    run_dir = tmp_path / "run"
    sentinel_series = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(cli.EncodedFootprintSeries, "load", lambda path: sentinel_series)

    def fake_train_model(series, output, train_config, model_config, device):
        captured.update(
            series=series,
            output=output,
            train_config=train_config,
            model_config=model_config,
            device=device,
        )
        return {"best_epoch": 1}

    monkeypatch.setattr(cli, "train_model", fake_train_model)

    assert (
        cli.main(
            [
                "train",
                "--dataset",
                str(dataset),
                "--run-dir",
                str(run_dir),
                "--model-preset",
                "fdt-n-tiny",
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert captured == {
        "series": sentinel_series,
        "output": run_dir,
        "train_config": TrainConfig(),
        "model_config": ModelConfig.from_preset("fdt-n-tiny"),
        "device": "cuda",
    }


def test_train_cli_rejects_preset_with_width_override(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        cli.main(
            [
                "train",
                "--dataset",
                str(tmp_path / "dataset.npz"),
                "--run-dir",
                str(tmp_path / "run"),
                "--model-preset",
                "fdt-n-tiny",
                "--d-model",
                "64",
            ]
        )


def test_backtest_cli_passes_frozen_policy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_study(**kwargs):
        captured.update(kwargs)
        return {
            "financial_metrics": {
                "fdt_n_tiny_ensemble_primary": {"net_return": 0.01, "trades": 3}
            }
        }

    monkeypatch.setattr(cli, "run_profitability_study", fake_study)
    checkpoints = [tmp_path / f"seed-{seed}.pt" for seed in (17, 29, 43)]
    results = [tmp_path / f"seed-{seed}.json" for seed in (17, 29, 43)]
    arguments = [
        "backtest",
        "--preregistration",
        str(tmp_path / "preregistration.json"),
        "--development-dataset",
        str(tmp_path / "development.npz"),
        "--dataset",
        str(tmp_path / "holdout.npz"),
        "--dataset-manifest",
        str(tmp_path / "holdout.manifest.json"),
        "--development-manifest",
        str(tmp_path / "development.manifest.json"),
        "--checkpoints",
        *(str(path) for path in checkpoints),
        "--training-results",
        *(str(path) for path in results),
        "--ridge",
        str(tmp_path / "ridge.npz"),
        "--run-dir",
        str(tmp_path / "run"),
        "--device",
        "cuda",
    ]

    assert cli.main(arguments) == 0
    assert captured["checkpoint_paths"] == checkpoints
    assert captured["training_results_paths"] == results
    assert captured["preregistration_path"] == tmp_path / "preregistration.json"
    assert captured["development_dataset_path"] == tmp_path / "development.npz"
    assert captured["device_name"] == "cuda"


def test_prepare_screen_cli_forwards_declared_candidates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_prepare(protocol, manifests, output, candidates):
        captured.update(
            protocol=protocol,
            manifests=manifests,
            output=output,
            candidates=candidates,
        )
        return {"candidates": []}

    monkeypatch.setattr(cli, "prepare_screen_datasets", fake_prepare)
    manifests = [tmp_path / "development.json", tmp_path / "holdout.json"]
    assert (
        cli.main(
            [
                "prepare-screen",
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--source-manifests",
                *(str(path) for path in manifests),
                "--output-dir",
                str(tmp_path / "datasets"),
                "--candidates",
                "5m-l48-h6",
                "15m-l16-h2",
            ]
        )
        == 0
    )
    assert captured["manifests"] == manifests
    assert captured["candidates"] == ["5m-l48-h6", "15m-l16-h2"]


def test_screen_candidate_cli_reports_gate_status(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_screen(*args, **kwargs):
        captured.update(args=args, kwargs=kwargs)
        return {
            "status": "complete_but_not_profitable",
            "financial_metrics": {"primary": {"net_return": -0.02, "trades": 40}},
        }

    monkeypatch.setattr(cli, "evaluate_screen_candidate", fake_screen)
    assert (
        cli.main(
            [
                "screen-candidate",
                "--protocol",
                str(tmp_path / "protocol.json"),
                "--candidate",
                "5m-l48-h6",
                "--dataset",
                str(tmp_path / "dataset.npz"),
                "--dataset-manifest",
                str(tmp_path / "dataset.manifest.json"),
                "--checkpoint",
                str(tmp_path / "best.pt"),
                "--selection-result",
                str(tmp_path / "selection-results.json"),
                "--run-dir",
                str(tmp_path / "screen"),
                "--device",
                "cuda",
            ]
        )
        == 0
    )
    assert captured["args"][1] == "5m-l48-h6"
    assert captured["kwargs"]["device_name"] == "cuda"
