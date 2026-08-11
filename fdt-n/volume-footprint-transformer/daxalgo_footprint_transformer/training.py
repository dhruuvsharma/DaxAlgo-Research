from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import TARGET_NAMES, ModelConfig
from .data import EncodedFootprintSeries
from .loss import quantile_loss
from .model import FootprintDistributionTransformer


class WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, series: EncodedFootprintSeries, indices: np.ndarray) -> None:
        self.series = series
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, item: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        row, row_mask, bar, target, target_mask = self.series.make_window(int(self.indices[item]))
        return (
            torch.from_numpy(row),
            torch.from_numpy(row_mask),
            torch.from_numpy(bar),
            torch.from_numpy(target),
            torch.from_numpy(target_mask),
        )


def collate_windows(
    items: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Pad only to the widest row set in this batch; no observed price row is compacted."""

    max_rows = max(item[0].shape[1] for item in items)
    batch = len(items)
    lookback = items[0][0].shape[0]
    row_features = items[0][0].shape[2]
    rows = torch.zeros(batch, lookback, max_rows, row_features, dtype=torch.float32)
    row_mask = torch.zeros(batch, lookback, max_rows, dtype=torch.bool)
    for index, item in enumerate(items):
        width = item[0].shape[1]
        rows[index, :, :width] = item[0]
        row_mask[index, :, :width] = item[1]
    return (
        rows,
        row_mask,
        torch.stack([item[2] for item in items]),
        torch.stack([item[3] for item in items]),
        torch.stack([item[4] for item in items]),
    )


class RowTokenBatchSampler(Sampler[list[int]]):
    """Bound padded row tokens while retaining every observed row in every selected window."""

    def __init__(
        self,
        widths: np.ndarray,
        *,
        maximum_items: int,
        maximum_row_tokens: int,
        sort_by_width: bool,
        shuffle_batches: bool,
        seed: int,
    ) -> None:
        self.widths = np.asarray(widths, dtype=np.int64)
        if self.widths.ndim != 1 or self.widths.size == 0 or np.any(self.widths < 1):
            raise ValueError("widths must be a non-empty positive vector")
        if maximum_items < 1 or maximum_row_tokens < 1:
            raise ValueError("batch bounds must be positive")
        self.maximum_items = maximum_items
        self.maximum_row_tokens = maximum_row_tokens
        self.sort_by_width = sort_by_width
        self.shuffle_batches = shuffle_batches
        self.seed = seed
        self._epoch = 0

    def _build_batches(self) -> list[list[int]]:
        order = (
            np.argsort(self.widths, kind="stable")
            if self.sort_by_width
            else np.arange(len(self.widths), dtype=np.int64)
        )
        batches: list[list[int]] = []
        current: list[int] = []
        current_width = 0
        for raw_index in order:
            index = int(raw_index)
            next_width = max(current_width, int(self.widths[index]))
            exceeds = len(current) >= self.maximum_items or (
                current and (len(current) + 1) * next_width > self.maximum_row_tokens
            )
            if exceeds:
                batches.append(current)
                current = []
                current_width = 0
                next_width = int(self.widths[index])
            current.append(index)
            current_width = next_width
        if current:
            batches.append(current)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self.shuffle_batches:
            np.random.default_rng(self.seed + self._epoch).shuffle(batches)
            self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches())


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    warmup_fraction: float = 0.08
    gradient_clip: float = 1.0
    patience: int = 8
    seed: int = 17
    workers: int = 0
    row_token_reference: int = 64
    ridge_alpha: float = 1.0
    bootstrap_resamples: int = 2_000
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    selection_only: bool = False


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)


def chronological_split(
    series: EncodedFootprintSeries,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if train_fraction <= 0 or validation_fraction <= 0:
        raise ValueError("split fractions must be positive")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train + validation fractions must leave a test partition")
    indices = series.valid_window_indices()
    n = len(indices)
    train_stop = int(n * train_fraction)
    validation_stop = int(n * (train_fraction + validation_fraction))
    purge = series.config.lookback_bars + series.config.horizon_bars
    train = indices[: max(0, train_stop - purge)]
    validation = indices[
        min(n, train_stop + purge) : max(train_stop + purge, validation_stop - purge)
    ]
    test = indices[min(n, validation_stop + purge) :]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("series is too short for purged chronological splits")
    return train, validation, test


def _to_device(
    batch: tuple[Tensor, Tensor, Tensor, Tensor, Tensor], device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(item.to(device, non_blocking=True) for item in batch)  # type: ignore[return-value]


def _window_row_widths(series: EncodedFootprintSeries, indices: np.ndarray) -> np.ndarray:
    row_counts = np.diff(series.row_offsets)
    lookback = series.config.lookback_bars
    return np.asarray(
        [int(row_counts[int(index) - lookback + 1 : int(index) + 1].max()) for index in indices],
        dtype=np.int64,
    )


def _window_loader(
    series: EncodedFootprintSeries,
    indices: np.ndarray,
    train_config: TrainConfig,
    *,
    training: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
    sampler = RowTokenBatchSampler(
        _window_row_widths(series, indices),
        maximum_items=train_config.batch_size,
        maximum_row_tokens=train_config.batch_size * train_config.row_token_reference,
        sort_by_width=training,
        shuffle_batches=training,
        seed=train_config.seed,
    )
    return DataLoader(
        WindowDataset(series, indices),
        batch_sampler=sampler,
        num_workers=train_config.workers,
        pin_memory=True,
        collate_fn=collate_windows,
    )


QUANTILE_LEVELS = np.asarray((0.1, 0.5, 0.9), dtype=np.float64)
QUANTILE_NAMES = ("q10", "q50", "q90")


def _mean_or_none(values: np.ndarray) -> float | None:
    return float(np.mean(values)) if values.size else None


def _masked_values(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=np.bool_)]


def distribution_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    *,
    loss: float | None = None,
) -> dict[str, object]:
    """Report calibrated-distribution diagnostics in the encoded target space."""

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    expected = targets.shape + (len(QUANTILE_LEVELS),)
    if predictions.shape != expected or target_mask.shape != targets.shape:
        raise ValueError("prediction, target, and target-mask shapes do not align")
    if targets.ndim != 3 or targets.shape[2] != len(TARGET_NAMES):
        raise ValueError("unexpected target shape")

    by_horizon_target: dict[str, object] = {}
    for horizon in range(targets.shape[1]):
        horizon_metrics: dict[str, object] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            valid = target_mask[:, horizon, target_index]
            realized = targets[valid, horizon, target_index]
            predicted = predictions[valid, horizon, target_index, :]
            errors = realized[:, None] - predicted
            pinball = np.maximum(
                QUANTILE_LEVELS * errors,
                (QUANTILE_LEVELS - 1.0) * errors,
            )
            metrics: dict[str, object] = {
                "examples": int(valid.sum()),
                "pinball_normalized": {
                    name: _mean_or_none(pinball[:, quantile_index])
                    for quantile_index, name in enumerate(QUANTILE_NAMES)
                },
                "median_mae_normalized": _mean_or_none(np.abs(predicted[:, 1] - realized)),
                "median_bias_normalized": _mean_or_none(predicted[:, 1] - realized),
                "central_80_coverage": _mean_or_none(
                    ((realized >= predicted[:, 0]) & (realized <= predicted[:, 2])).astype(
                        np.float64
                    )
                ),
                "mean_interval_width_normalized": _mean_or_none(predicted[:, 2] - predicted[:, 0]),
            }
            if target_index == 0:
                metrics["poc_directional_hit_rate"] = _mean_or_none(
                    (np.sign(predicted[:, 1]) == np.sign(realized)).astype(np.float64)
                )
            horizon_metrics[target_name] = metrics
        by_horizon_target[str(horizon + 1)] = horizon_metrics

    all_errors = targets[..., None] - predictions
    pooled_pinball = np.maximum(
        QUANTILE_LEVELS * all_errors,
        (QUANTILE_LEVELS - 1.0) * all_errors,
    )
    median_error = predictions[..., 1] - targets
    coverage = (targets >= predictions[..., 0]) & (targets <= predictions[..., 2])
    interval_width = predictions[..., 2] - predictions[..., 0]
    poc_mask = target_mask[:, :, 0]
    poc_hits = np.sign(predictions[:, :, 0, 1]) == np.sign(targets[:, :, 0])

    pooled = {
        "pinball_normalized": {
            name: _mean_or_none(_masked_values(pooled_pinball[..., index], target_mask))
            for index, name in enumerate(QUANTILE_NAMES)
        },
        "median_mae_normalized": _mean_or_none(_masked_values(np.abs(median_error), target_mask)),
        "median_bias_normalized": _mean_or_none(_masked_values(median_error, target_mask)),
        "central_80_coverage": _mean_or_none(
            _masked_values(coverage.astype(np.float64), target_mask)
        ),
        "mean_interval_width_normalized": _mean_or_none(
            _masked_values(interval_width, target_mask)
        ),
        "poc_directional_hit_rate": _mean_or_none(
            _masked_values(poc_hits.astype(np.float64), poc_mask)
        ),
    }
    per_target_mae: dict[str, float | None] = {}
    per_target_coverage: dict[str, float | None] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        valid = target_mask[:, :, target_index]
        per_target_mae[target_name] = _mean_or_none(
            _masked_values(np.abs(median_error[:, :, target_index]), valid)
        )
        per_target_coverage[target_name] = _mean_or_none(
            _masked_values(coverage[:, :, target_index].astype(np.float64), valid)
        )

    return {
        "loss": float(loss) if loss is not None else None,
        "examples": int(targets.shape[0]),
        "pooled": pooled,
        "median_mae_normalized": per_target_mae,
        "central_80_coverage": per_target_coverage,
        "by_horizon_target": by_horizon_target,
    }


def point_prediction_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, object]:
    """Metrics for point baselines, kept separate from distribution calibration claims."""

    predictions = np.asarray(predictions, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    if predictions.shape != targets.shape or target_mask.shape != targets.shape:
        raise ValueError("point prediction, target, and mask shapes do not align")
    if targets.ndim != 3 or targets.shape[2] != len(TARGET_NAMES):
        raise ValueError("unexpected target shape")

    errors = predictions - targets
    by_horizon_target: dict[str, object] = {}
    for horizon in range(targets.shape[1]):
        horizon_metrics: dict[str, object] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            valid = target_mask[:, horizon, target_index]
            selected = errors[valid, horizon, target_index]
            metrics: dict[str, object] = {
                "examples": int(valid.sum()),
                "median_mae_normalized": _mean_or_none(np.abs(selected)),
                "median_bias_normalized": _mean_or_none(selected),
            }
            if target_index == 0:
                metrics["poc_directional_hit_rate"] = _mean_or_none(
                    (
                        np.sign(predictions[valid, horizon, target_index])
                        == np.sign(targets[valid, horizon, target_index])
                    ).astype(np.float64)
                )
            horizon_metrics[target_name] = metrics
        by_horizon_target[str(horizon + 1)] = horizon_metrics

    per_target_mae: dict[str, float | None] = {}
    per_target_bias: dict[str, float | None] = {}
    for target_index, target_name in enumerate(TARGET_NAMES):
        valid = target_mask[:, :, target_index]
        selected = errors[:, :, target_index][valid]
        per_target_mae[target_name] = _mean_or_none(np.abs(selected))
        per_target_bias[target_name] = _mean_or_none(selected)

    poc_mask = target_mask[:, :, 0]
    poc_hits = np.sign(predictions[:, :, 0]) == np.sign(targets[:, :, 0])
    return {
        "examples": int(targets.shape[0]),
        "pooled": {
            "median_mae_normalized": _mean_or_none(_masked_values(np.abs(errors), target_mask)),
            "median_bias_normalized": _mean_or_none(_masked_values(errors, target_mask)),
            "poc_directional_hit_rate": _mean_or_none(
                _masked_values(poc_hits.astype(np.float64), poc_mask)
            ),
        },
        "median_mae_normalized": per_target_mae,
        "median_bias_normalized": per_target_bias,
        "by_horizon_target": by_horizon_target,
    }


@torch.inference_mode()
def _collect_model_outputs(
    model: FootprintDistributionTransformer,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []
    masks: list[Tensor] = []
    for batch in loader:
        row, row_mask, bar, target, target_mask = _to_device(batch, device)
        bar_mask = torch.ones(bar.shape[:2], dtype=torch.bool, device=device)
        predictions.append(model(row, row_mask, bar, bar_mask).cpu())
        targets.append(target.cpu())
        masks.append(target_mask.cpu())
    if not predictions:
        raise ValueError("evaluation loader is empty")
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    mask = torch.cat(masks)
    loss = float(quantile_loss(prediction, target, mask).item())
    return prediction.numpy(), target.numpy(), mask.numpy(), loss


def evaluate(
    model: FootprintDistributionTransformer,
    loader: DataLoader[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]],
    device: torch.device,
) -> dict[str, object]:
    prediction, target, mask, loss = _collect_model_outputs(model, loader, device)
    return distribution_metrics(prediction, target, mask, loss=loss)


def _persistence_target_values(
    series: EncodedFootprintSeries,
    context_index: int,
    encoded_bar: np.ndarray,
) -> dict[str, float]:
    values = series.bar_values[context_index]
    anchor_poc = values[0]
    scale = series.config.price_scale_ticks
    return {
        "poc_offset": 0.0,
        "low_offset": float(np.clip((values[6] - anchor_poc) / scale, -2.0, 2.0)),
        "high_offset": float(np.clip((values[7] - anchor_poc) / scale, -2.0, 2.0)),
        "buy_poc_offset": float(np.clip((values[1] - anchor_poc) / scale, -2.0, 2.0)),
        "sell_poc_offset": float(np.clip((values[2] - anchor_poc) / scale, -2.0, 2.0)),
        "log_volume_relative": float(encoded_bar[-1, 10]),
        "delta_fraction": float(np.clip(values[13], -1.0, 1.0)),
    }


def persistence_baseline_outputs(
    series: EncodedFootprintSeries,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in np.asarray(indices, dtype=np.int64):
        _, _, encoded_bar, target, target_mask = series.make_window(int(index))
        values = _persistence_target_values(series, int(index), encoded_bar)
        try:
            point = np.asarray([values[name] for name in TARGET_NAMES], dtype=np.float32)
        except KeyError as exception:
            message = f"unsupported target for persistence: {exception.args[0]}"
            raise ValueError(message) from exception
        predictions.append(np.broadcast_to(point, target.shape).copy())
        targets.append(target)
        masks.append(target_mask)
    if not predictions:
        raise ValueError("baseline indices are empty")
    return np.stack(predictions), np.stack(targets), np.stack(masks)


def persistence_baseline(
    series: EncodedFootprintSeries,
    indices: np.ndarray,
) -> dict[str, object]:
    prediction, target, mask = persistence_baseline_outputs(series, indices)
    return point_prediction_metrics(prediction, target, mask)


def _ridge_window_arrays(
    series: EncodedFootprintSeries,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for index in np.asarray(indices, dtype=np.int64):
        _, _, encoded_bar, target, target_mask = series.make_window(int(index))
        latest = encoded_bar[-1].astype(np.float64)
        mean = encoded_bar.mean(axis=0, dtype=np.float64)
        trend = latest - encoded_bar[0]
        features.append(np.concatenate((latest, mean, trend)))
        targets.append(target.astype(np.float64))
        masks.append(target_mask)
    if not features:
        raise ValueError("ridge indices are empty")
    return np.stack(features), np.stack(targets), np.stack(masks)


def _clip_target_space(predictions: np.ndarray) -> np.ndarray:
    clipped = np.asarray(predictions, dtype=np.float64).copy()
    for target_index, target_name in enumerate(TARGET_NAMES):
        if target_name == "delta_fraction":
            clipped[:, :, target_index] = np.clip(clipped[:, :, target_index], -1.0, 1.0)
        else:
            clipped[:, :, target_index] = np.clip(clipped[:, :, target_index], -2.0, 2.0)
    return clipped


@dataclass(frozen=True)
class DirectRidgeBaseline:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    alpha: float
    training_examples: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_mean.shape[0]:
            raise ValueError("unexpected ridge feature shape")
        standardized = (values - self.feature_mean) / self.feature_scale
        design = np.concatenate((standardized, np.ones((len(values), 1))), axis=1)
        prediction = np.einsum("nd,htd->nht", design, self.coefficients, optimize=True)
        return _clip_target_space(prediction)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            coefficients=self.coefficients,
            alpha=np.asarray(self.alpha),
            training_examples=np.asarray(self.training_examples),
            target_names=np.asarray(TARGET_NAMES, dtype="U"),
            feature_strategy=np.asarray("latest_mean_trend_bar_aggregates"),
            target_space=np.asarray("encoded_normalized_targets"),
        )

    @classmethod
    def load(cls, path: Path) -> DirectRidgeBaseline:
        with np.load(path, allow_pickle=False) as bundle:
            target_names = tuple(str(name) for name in bundle["target_names"])
            if target_names != TARGET_NAMES:
                raise ValueError("ridge artifact target schema does not match this evaluator")
            if str(bundle["feature_strategy"]) != "latest_mean_trend_bar_aggregates":
                raise ValueError("unsupported ridge feature strategy")
            if str(bundle["target_space"]) != "encoded_normalized_targets":
                raise ValueError("unsupported ridge target space")
            feature_mean = np.asarray(bundle["feature_mean"], dtype=np.float64)
            feature_scale = np.asarray(bundle["feature_scale"], dtype=np.float64)
            coefficients = np.asarray(bundle["coefficients"], dtype=np.float64)
            alpha = float(bundle["alpha"])
            training_examples = int(bundle["training_examples"])
        if (
            feature_mean.ndim != 1
            or feature_scale.shape != feature_mean.shape
            or coefficients.ndim != 3
            or coefficients.shape[1] != len(TARGET_NAMES)
            or coefficients.shape[2] != len(feature_mean) + 1
        ):
            raise ValueError("ridge artifact array schema is invalid")
        if (
            not np.isfinite(feature_mean).all()
            or not np.isfinite(feature_scale).all()
            or not np.isfinite(coefficients).all()
            or np.any(feature_scale <= 0)
            or not math.isfinite(alpha)
            or alpha <= 0
            or training_examples < 1
        ):
            raise ValueError("ridge artifact contains invalid numeric values")
        return cls(
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            coefficients=coefficients,
            alpha=alpha,
            training_examples=training_examples,
        )


def fit_direct_ridge_baseline(
    series: EncodedFootprintSeries,
    train_indices: np.ndarray,
    *,
    alpha: float = 1.0,
) -> DirectRidgeBaseline:
    """Fit independent direct horizon/target ridge models using training windows only."""

    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("ridge alpha must be finite and positive")
    features, targets, masks = _ridge_window_arrays(series, train_indices)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (features - feature_mean) / feature_scale
    design = np.concatenate((standardized, np.ones((len(features), 1))), axis=1)
    coefficients = np.zeros(
        (targets.shape[1], targets.shape[2], design.shape[1]),
        dtype=np.float64,
    )
    regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularizer[-1, -1] = 0.0
    for horizon in range(targets.shape[1]):
        for target_index in range(targets.shape[2]):
            valid = masks[:, horizon, target_index]
            if not np.any(valid):
                raise ValueError(
                    f"no training targets for horizon {horizon + 1}, {TARGET_NAMES[target_index]}"
                )
            selected = design[valid]
            response = targets[valid, horizon, target_index]
            gram = selected.T @ selected + regularizer
            coefficients[horizon, target_index] = np.linalg.solve(
                gram,
                selected.T @ response,
            )
    return DirectRidgeBaseline(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        coefficients=coefficients,
        alpha=alpha,
        training_examples=int(len(features)),
    )


def direct_ridge_baseline_outputs(
    baseline: DirectRidgeBaseline,
    series: EncodedFootprintSeries,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, targets, masks = _ridge_window_arrays(series, indices)
    return baseline.predict(features), targets, masks


def paired_block_bootstrap_mae_difference(
    model_median: np.ndarray,
    baseline_prediction: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    block_ids: np.ndarray,
    *,
    resamples: int = 2_000,
    seed: int = 17,
) -> dict[str, object]:
    """Paired 95% CI for model MAE minus baseline MAE via whole-block resampling."""

    model_median = np.asarray(model_median, dtype=np.float64)
    baseline_prediction = np.asarray(baseline_prediction, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    target_mask = np.asarray(target_mask, dtype=np.bool_)
    block_ids = np.asarray(block_ids)
    if not (model_median.shape == baseline_prediction.shape == targets.shape == target_mask.shape):
        raise ValueError("paired prediction, target, and mask shapes do not align")
    if targets.ndim != 3 or targets.shape[2] != len(TARGET_NAMES):
        raise ValueError("unexpected paired target shape")
    if block_ids.ndim != 1 or len(block_ids) != targets.shape[0]:
        raise ValueError("one block id is required per example")
    if resamples < 1:
        raise ValueError("bootstrap resamples must be positive")

    unique_blocks, inverse = np.unique(block_ids, return_inverse=True)
    if unique_blocks.size == 0:
        raise ValueError("at least one bootstrap block is required")
    paired_error = np.abs(model_median - targets) - np.abs(baseline_prediction - targets)
    block_sums = np.zeros((len(unique_blocks),) + targets.shape[1:], dtype=np.float64)
    block_counts = np.zeros_like(block_sums)
    for block_index in range(len(unique_blocks)):
        rows = inverse == block_index
        selected_mask = target_mask[rows]
        block_sums[block_index] = np.where(selected_mask, paired_error[rows], 0.0).sum(axis=0)
        block_counts[block_index] = selected_mask.sum(axis=0)

    total_count = block_counts.sum(axis=0)
    point = np.divide(
        block_sums.sum(axis=0),
        total_count,
        out=np.full_like(total_count, np.nan),
        where=total_count > 0,
    )
    rng = np.random.default_rng(seed)
    draws = np.full((resamples,) + targets.shape[1:], np.nan, dtype=np.float64)
    for sample in range(resamples):
        selected = rng.integers(0, len(unique_blocks), size=len(unique_blocks))
        selected_count = block_counts[selected].sum(axis=0)
        draws[sample] = np.divide(
            block_sums[selected].sum(axis=0),
            selected_count,
            out=np.full_like(selected_count, np.nan),
            where=selected_count > 0,
        )

    by_horizon_target: dict[str, object] = {}
    for horizon in range(targets.shape[1]):
        horizon_metrics: dict[str, object] = {}
        for target_index, target_name in enumerate(TARGET_NAMES):
            finite = draws[:, horizon, target_index]
            finite = finite[np.isfinite(finite)]
            horizon_metrics[target_name] = {
                "examples": int(total_count[horizon, target_index]),
                "mae_difference_normalized": (
                    float(point[horizon, target_index])
                    if np.isfinite(point[horizon, target_index])
                    else None
                ),
                "ci_95_lower": float(np.quantile(finite, 0.025)) if finite.size else None,
                "ci_95_upper": float(np.quantile(finite, 0.975)) if finite.size else None,
            }
        by_horizon_target[str(horizon + 1)] = horizon_metrics
    return {
        "difference": "model_median_mae_minus_baseline_mae",
        "confidence_level": 0.95,
        "resamples": int(resamples),
        "seed": int(seed),
        "blocks": int(len(unique_blocks)),
        "by_horizon_target": by_horizon_target,
    }


def _evaluation_block_ids(
    series: EncodedFootprintSeries,
    indices: np.ndarray,
) -> tuple[np.ndarray, str]:
    # Four-hour contiguous blocks are materially longer than the 64+8-bar dependency span while
    # providing more resampling units than calendar days. Adjacent fixed blocks are not strictly
    # independent at their boundary because neighboring windows overlap, so the resulting interval
    # is a descriptive dependence-aware sensitivity check rather than a formal generalization bound.
    minimum_independent_span = series.config.lookback_bars + series.config.horizon_bars
    four_hours = max(1, 4 * 3_600 // series.config.interval_seconds)
    block_length = max(minimum_independent_span, four_hours)
    return np.arange(len(indices), dtype=np.int64) // block_length, (
        f"contiguous_{block_length}_example_blocks"
    )


def train_model(
    series: EncodedFootprintSeries,
    run_dir: Path,
    train_config: TrainConfig | None = None,
    model_config: ModelConfig | None = None,
    device_name: str | None = None,
) -> dict[str, object]:
    train_config = train_config or TrainConfig()
    model_config = model_config or ModelConfig()
    if train_config.ridge_alpha <= 0 or not math.isfinite(train_config.ridge_alpha):
        raise ValueError("ridge_alpha must be finite and positive")
    if train_config.bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be positive")
    if train_config.batch_size < 1 or train_config.row_token_reference < 1:
        raise ValueError("batch size and row-token reference must be positive")
    seed_everything(train_config.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_indices, validation_indices, test_indices = chronological_split(
        series,
        train_fraction=train_config.train_fraction,
        validation_fraction=train_config.validation_fraction,
    )
    loaders = {
        "train": _window_loader(series, train_indices, train_config, training=True),
        "validation": _window_loader(series, validation_indices, train_config, training=False),
    }
    if not train_config.selection_only:
        loaders["test"] = _window_loader(series, test_indices, train_config, training=False)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FootprintDistributionTransformer(series.config, model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_steps = max(1, train_config.epochs * len(loaders["train"]))
    warmup_steps = max(1, int(total_steps * train_config.warmup_fraction))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    best_loss = math.inf
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, float]] = []
    checkpoint_path = run_dir / "best.pt"
    progress_path = run_dir / "progress.json"
    global_step = 0
    started = time.perf_counter()

    for epoch in range(1, train_config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        epoch_losses: list[float] = []
        for batch in loaders["train"]:
            row, row_mask, bar, target, target_mask = _to_device(batch, device)
            bar_mask = torch.ones(bar.shape[:2], dtype=torch.bool, device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(row, row_mask, bar, bar_mask)
            loss = quantile_loss(prediction, target, target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_losses.append(float(loss.item()))
        validation = evaluate(model, loaders["validation"], device)
        validation_loss = float(validation["loss"])
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(epoch_losses)),
            "validation_loss": validation_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "encoding_config": series.config.to_dict(),
                    "model_config": model_config.to_dict(),
                    "train_config": asdict(train_config),
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "training",
                    "seed": train_config.seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "global_steps": global_step,
                    "elapsed_seconds": time.perf_counter() - started,
                    "history": history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if stale_epochs >= train_config.patience:
            break

    saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    if train_config.selection_only:
        validation_prediction, validation_target, validation_mask, validation_loss = (
            _collect_model_outputs(model, loaders["validation"], device)
        )
        validation = distribution_metrics(
            validation_prediction,
            validation_target,
            validation_mask,
            loss=validation_loss,
        )
        persistence_prediction, persistence_target, persistence_mask = (
            persistence_baseline_outputs(series, validation_indices)
        )
        if not (
            np.array_equal(validation_mask, persistence_mask)
            and np.array_equal(validation_target, persistence_target)
        ):
            raise RuntimeError("persistence validation windows do not align with model windows")
        persistence_validation = point_prediction_metrics(
            persistence_prediction, persistence_target, persistence_mask
        )
        ridge = fit_direct_ridge_baseline(
            series,
            train_indices,
            alpha=train_config.ridge_alpha,
        )
        ridge_path = run_dir / "direct_ridge_baseline.npz"
        ridge.save(ridge_path)
        ridge_prediction, ridge_target, ridge_mask = direct_ridge_baseline_outputs(
            ridge, series, validation_indices
        )
        if not (
            np.array_equal(validation_mask, ridge_mask)
            and np.array_equal(validation_target, ridge_target)
        ):
            raise RuntimeError("ridge validation windows do not align with model windows")
        ridge_validation = point_prediction_metrics(ridge_prediction, ridge_target, ridge_mask)
        evaluation_blocks, block_definition = _evaluation_block_ids(series, validation_indices)
        comparisons = {
            "model_vs_persistence": paired_block_bootstrap_mae_difference(
                validation_prediction[..., 1],
                persistence_prediction,
                validation_target,
                validation_mask,
                evaluation_blocks,
                resamples=train_config.bootstrap_resamples,
                seed=train_config.seed,
            ),
            "model_vs_direct_ridge": paired_block_bootstrap_mae_difference(
                validation_prediction[..., 1],
                ridge_prediction,
                validation_target,
                validation_mask,
                evaluation_blocks,
                resamples=train_config.bootstrap_resamples,
                seed=train_config.seed,
            ),
        }
        for comparison in comparisons.values():
            comparison["block_definition"] = block_definition
        result = {
            "schema_version": 1,
            "status": "development_selection_only_no_screen_access",
            "architecture": "FDT-N",
            "parameter_count": model.parameter_count,
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "global_steps": global_step,
            "splits": {
                "train_examples": int(len(train_indices)),
                "validation_examples": int(len(validation_indices)),
                "reserved_screen_examples": int(len(test_indices)),
                "purge_bars": series.config.lookback_bars + series.config.horizon_bars,
            },
            "encoding_config": series.config.to_dict(),
            "model_config": model_config.to_dict(),
            "train_config": asdict(train_config),
            "validation": validation,
            "baselines": {
                "persistence": {
                    "target_space": "encoded_normalized_targets",
                    "validation": persistence_validation,
                },
                "direct_ridge": {
                    "target_space": "encoded_normalized_targets",
                    "feature_strategy": "latest_mean_trend_bar_aggregates",
                    "feature_count": int(ridge.feature_mean.shape[0]),
                    "alpha": ridge.alpha,
                    "training_examples": ridge.training_examples,
                    "artifact": str(ridge_path),
                    "validation": ridge_validation,
                },
            },
            "comparisons": comparisons,
            "history": history,
            "checkpoint": str(checkpoint_path),
        }
        (run_dir / "selection-results.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        progress_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "selection_complete_screen_untouched",
                    "seed": train_config.seed,
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_loss,
                    "global_steps": global_step,
                    "elapsed_seconds": result["elapsed_seconds"],
                    "history": history,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            run_dir / "splits.npz",
            train=train_indices,
            validation=validation_indices,
            reserved_screen=test_indices,
        )
        return result
    validation = evaluate(model, loaders["validation"], device)
    test_prediction, test_target, test_mask, test_loss = _collect_model_outputs(
        model, loaders["test"], device
    )
    test = distribution_metrics(test_prediction, test_target, test_mask, loss=test_loss)

    persistence_prediction, persistence_target, persistence_mask = persistence_baseline_outputs(
        series, test_indices
    )
    if not (
        np.array_equal(test_mask, persistence_mask)
        and np.array_equal(test_target, persistence_target)
    ):
        raise RuntimeError("persistence evaluation windows do not align with model test windows")
    persistence_test = point_prediction_metrics(
        persistence_prediction, persistence_target, persistence_mask
    )

    ridge = fit_direct_ridge_baseline(
        series,
        train_indices,
        alpha=train_config.ridge_alpha,
    )
    ridge_path = run_dir / "direct_ridge_baseline.npz"
    ridge.save(ridge_path)
    ridge_prediction, ridge_target, ridge_mask = direct_ridge_baseline_outputs(
        ridge, series, test_indices
    )
    if not (np.array_equal(test_mask, ridge_mask) and np.array_equal(test_target, ridge_target)):
        raise RuntimeError("ridge evaluation windows do not align with model test windows")
    ridge_test = point_prediction_metrics(ridge_prediction, ridge_target, ridge_mask)

    evaluation_blocks, block_definition = _evaluation_block_ids(series, test_indices)
    comparisons = {
        "model_vs_persistence": paired_block_bootstrap_mae_difference(
            test_prediction[..., 1],
            persistence_prediction,
            test_target,
            test_mask,
            evaluation_blocks,
            resamples=train_config.bootstrap_resamples,
            seed=train_config.seed,
        ),
        "model_vs_direct_ridge": paired_block_bootstrap_mae_difference(
            test_prediction[..., 1],
            ridge_prediction,
            test_target,
            test_mask,
            evaluation_blocks,
            resamples=train_config.bootstrap_resamples,
            seed=train_config.seed,
        ),
    }
    for comparison in comparisons.values():
        comparison["block_definition"] = block_definition
    result = {
        "schema_version": 2,
        "architecture": "FDT-N",
        "parameter_count": model.parameter_count,
        "device": str(device),
        "elapsed_seconds": time.perf_counter() - started,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "global_steps": global_step,
        "splits": {
            "train_examples": int(len(train_indices)),
            "validation_examples": int(len(validation_indices)),
            "test_examples": int(len(test_indices)),
            "purge_bars": series.config.lookback_bars + series.config.horizon_bars,
        },
        "encoding_config": series.config.to_dict(),
        "model_config": model_config.to_dict(),
        "train_config": asdict(train_config),
        "validation": validation,
        "test": test,
        "baselines": {
            "persistence": {
                "target_space": "encoded_normalized_targets",
                "test": persistence_test,
            },
            "direct_ridge": {
                "target_space": "encoded_normalized_targets",
                "feature_strategy": "latest_mean_trend_bar_aggregates",
                "feature_count": int(ridge.feature_mean.shape[0]),
                "alpha": ridge.alpha,
                "training_examples": ridge.training_examples,
                "artifact": str(ridge_path),
                "test": ridge_test,
            },
        },
        "comparisons": comparisons,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    progress_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "seed": train_config.seed,
                "best_epoch": best_epoch,
                "best_validation_loss": best_loss,
                "global_steps": global_step,
                "elapsed_seconds": result["elapsed_seconds"],
                "history": history,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        run_dir / "splits.npz",
        train=train_indices,
        validation=validation_indices,
        test=test_indices,
    )
    return result
