from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from daxalgo_footprint_transformer.config import TARGET_NAMES, EncodingConfig, ModelConfig
from daxalgo_footprint_transformer.data import EncodedFootprintSeries, build_footprint_bars
from daxalgo_footprint_transformer.training import (
    RowTokenBatchSampler,
    TrainConfig,
    direct_ridge_baseline_outputs,
    distribution_metrics,
    fit_direct_ridge_baseline,
    paired_block_bootstrap_mae_difference,
    persistence_baseline_outputs,
    point_prediction_metrics,
    train_model,
)


def _series(count: int = 180, *, horizon: int = 3) -> EncodedFootprintSeries:
    config = EncodingConfig(
        lookback_bars=8,
        horizon_bars=horizon,
        max_rows=8,
        row_size=1.0,
        price_scale_ticks=16.0,
        gap_scale_ticks=8.0,
    )
    trades: list[tuple[int, float, float, bool]] = []
    for minute in range(count):
        anchor = 100.0 + 0.15 * minute + 0.5 * np.sin(minute / 7.0)
        timestamp = minute * 60_000
        buy_size = 8.0 + minute % 5
        sell_size = 5.0 + minute % 3
        trades.extend(
            [
                (timestamp + 1_000, anchor - 1.0, sell_size, True),
                (timestamp + 2_000, anchor, buy_size, False),
                (timestamp + 3_000, anchor, buy_size / 2.0, False),
                (timestamp + 4_000, anchor + 1.0, buy_size / 3.0, False),
            ]
        )
    return EncodedFootprintSeries.from_bars(build_footprint_bars(trades, config), config)


def test_distribution_metrics_report_every_horizon_target_and_quantile() -> None:
    targets = np.zeros((2, 2, len(TARGET_NAMES)), dtype=np.float64)
    predictions = np.empty(targets.shape + (3,), dtype=np.float64)
    predictions[..., 0] = -1.0
    predictions[..., 1] = 0.5
    predictions[..., 2] = 1.0
    mask = np.ones_like(targets, dtype=np.bool_)

    metrics = distribution_metrics(predictions, targets, mask, loss=0.25)
    poc = metrics["by_horizon_target"]["1"]["poc_offset"]

    assert metrics["loss"] == 0.25
    assert set(metrics["by_horizon_target"]) == {"1", "2"}
    assert set(metrics["by_horizon_target"]["1"]) == set(TARGET_NAMES)
    assert poc["pinball_normalized"] == pytest.approx({"q10": 0.1, "q50": 0.25, "q90": 0.1})
    assert poc["median_mae_normalized"] == 0.5
    assert poc["median_bias_normalized"] == 0.5
    assert poc["central_80_coverage"] == 1.0
    assert poc["mean_interval_width_normalized"] == 2.0
    assert poc["poc_directional_hit_rate"] == 0.0
    assert metrics["pooled"]["median_mae_normalized"] == 0.5


def test_row_token_batches_shrink_around_wide_exact_row_windows() -> None:
    widths = np.asarray([5, 7, 8, 9, 100, 110, 600], dtype=np.int64)
    sampler = RowTokenBatchSampler(
        widths,
        maximum_items=4,
        maximum_row_tokens=128,
        sort_by_width=True,
        shuffle_batches=False,
        seed=17,
    )
    batches = list(sampler)

    assert sorted(index for batch in batches for index in batch) == list(range(len(widths)))
    assert all(len(batch) <= 4 for batch in batches)
    assert all(len(batch) == 1 or len(batch) * int(widths[batch].max()) <= 128 for batch in batches)
    assert any(batch == [6] for batch in batches)


def test_persistence_uses_anchor_relative_price_scale_and_encoded_units() -> None:
    series = _series(count=80, horizon=2)
    indices = series.valid_window_indices()[10:13]
    prediction, target, mask = persistence_baseline_outputs(series, indices)

    context = int(indices[0])
    values = series.bar_values[context]
    expected = {
        "poc_offset": 0.0,
        "low_offset": (values[6] - values[0]) / series.config.price_scale_ticks,
        "high_offset": (values[7] - values[0]) / series.config.price_scale_ticks,
        "buy_poc_offset": (values[1] - values[0]) / series.config.price_scale_ticks,
        "sell_poc_offset": (values[2] - values[0]) / series.config.price_scale_ticks,
        "delta_fraction": values[13],
    }
    for target_name, value in expected.items():
        target_index = TARGET_NAMES.index(target_name)
        np.testing.assert_allclose(prediction[0, :, target_index], value)

    assert prediction.shape == target.shape == mask.shape
    metrics = point_prediction_metrics(prediction, target, mask)
    assert set(metrics["by_horizon_target"]["1"]) == set(TARGET_NAMES)
    assert "poc_directional_hit_rate" in metrics["by_horizon_target"]["2"]["poc_offset"]


def test_direct_ridge_fits_only_supplied_training_windows_and_persists(tmp_path: Path) -> None:
    series = _series(count=150, horizon=2)
    valid = series.valid_window_indices()
    train_indices = valid[:60]
    evaluation_indices = valid[-8:]

    baseline = fit_direct_ridge_baseline(series, train_indices, alpha=0.75)
    original_values = series.bar_values.copy()
    untouched_start = int(train_indices[-1]) + series.config.horizon_bars + 1
    series.bar_values[untouched_start:] += 10_000.0
    refit = fit_direct_ridge_baseline(series, train_indices, alpha=0.75)
    np.testing.assert_allclose(refit.coefficients, baseline.coefficients)
    np.testing.assert_allclose(refit.feature_mean, baseline.feature_mean)
    series.bar_values[:] = original_values

    prediction, target, mask = direct_ridge_baseline_outputs(baseline, series, evaluation_indices)
    assert prediction.shape == target.shape == mask.shape
    assert np.isfinite(prediction).all()
    assert np.max(np.abs(prediction[..., :6])) <= 2.0
    assert np.max(np.abs(prediction[..., 6])) <= 1.0

    artifact = tmp_path / "ridge.npz"
    baseline.save(artifact)
    with np.load(artifact, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["target_names"], np.asarray(TARGET_NAMES))
        assert float(saved["alpha"]) == 0.75
        assert str(saved["target_space"]) == "encoded_normalized_targets"


def test_paired_block_bootstrap_is_deterministic_and_paired() -> None:
    targets = np.zeros((6, 1, len(TARGET_NAMES)), dtype=np.float64)
    model = np.zeros_like(targets)
    baseline = np.ones_like(targets)
    model[2:4] = 1.0
    model[4:] = 2.0
    mask = np.ones_like(targets, dtype=np.bool_)
    blocks = np.asarray([0, 0, 1, 1, 2, 2])

    first = paired_block_bootstrap_mae_difference(
        model, baseline, targets, mask, blocks, resamples=200, seed=9
    )
    second = paired_block_bootstrap_mae_difference(
        model, baseline, targets, mask, blocks, resamples=200, seed=9
    )

    assert first == second
    assert first["difference"] == "model_median_mae_minus_baseline_mae"
    assert first["blocks"] == 3
    poc = first["by_horizon_target"]["1"]["poc_offset"]
    assert poc["mae_difference_normalized"] == 0.0
    assert poc["ci_95_lower"] <= 0.0 <= poc["ci_95_upper"]


def test_train_model_persists_rigorous_metrics_baselines_and_comparisons(
    tmp_path: Path,
) -> None:
    series = _series(count=180, horizon=2)
    result = train_model(
        series,
        tmp_path,
        TrainConfig(
            epochs=1,
            batch_size=32,
            patience=1,
            seed=3,
            ridge_alpha=0.5,
            bootstrap_resamples=32,
        ),
        ModelConfig(
            d_model=16,
            row_queries=2,
            attention_heads=4,
            temporal_layers=1,
            decoder_layers=1,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        "cpu",
    )

    assert result["schema_version"] == 2
    assert set(result["baselines"]) == {"persistence", "direct_ridge"}
    assert set(result["comparisons"]) == {
        "model_vs_persistence",
        "model_vs_direct_ridge",
    }
    assert set(result["test"]["by_horizon_target"]["1"]) == set(TARGET_NAMES)
    assert (
        result["baselines"]["direct_ridge"]["training_examples"]
        == result["splits"]["train_examples"]
    )
    assert (tmp_path / "direct_ridge_baseline.npz").is_file()
    persisted = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 2
    assert "model_vs_direct_ridge" in persisted["comparisons"]


def test_selection_only_training_never_writes_reserved_screen_metrics(tmp_path: Path) -> None:
    series = _series(count=400, horizon=2)
    result = train_model(
        series,
        tmp_path,
        TrainConfig(
            epochs=1,
            batch_size=32,
            patience=1,
            seed=5,
            bootstrap_resamples=16,
            train_fraction=0.60,
            validation_fraction=0.15,
            selection_only=True,
        ),
        ModelConfig(
            d_model=16,
            row_queries=2,
            attention_heads=4,
            temporal_layers=1,
            decoder_layers=1,
            feedforward_multiplier=2,
            dropout=0.0,
        ),
        "cpu",
    )

    assert result["status"] == "development_selection_only_no_screen_access"
    assert "test" not in result
    assert "reserved_screen_examples" in result["splits"]
    assert (tmp_path / "selection-results.json").is_file()
    assert not (tmp_path / "results.json").exists()
