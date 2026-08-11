"""Additive evaluation metrics for CHMT-Q probabilistic forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ChmtQConfig
from .output import ChmtQOutput


@dataclass(frozen=True, slots=True)
class ChmtQMetricReport:
    """Per-horizon/target metrics with explicit zero-count cells."""

    forecast_counts: Tensor
    pinball: Tensor
    median_mae: Tensor
    median_rmse: Tensor
    median_bias: Tensor
    interval_coverage: Tensor
    interval_width: Tensor
    presence_counts: Tensor
    presence_log_loss: Tensor
    presence_brier: Tensor
    weighted_primary_metric: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "forecast_counts": self.forecast_counts.tolist(),
            "pinball": self.pinball.tolist(),
            "median_mae": self.median_mae.tolist(),
            "median_rmse": self.median_rmse.tolist(),
            "median_bias": self.median_bias.tolist(),
            "interval_coverage": self.interval_coverage.tolist(),
            "interval_width": self.interval_width.tolist(),
            "presence_counts": self.presence_counts.tolist(),
            "presence_log_loss": self.presence_log_loss.tolist(),
            "presence_brier": self.presence_brier.tolist(),
            "weighted_primary_metric": self.weighted_primary_metric,
        }


class ChmtQMetricsAccumulator:
    """Accumulate metric numerators in float64 on CPU across arbitrary batches."""

    def __init__(self, config: ChmtQConfig | None = None) -> None:
        self.config = config or ChmtQConfig()
        forecast_shape = (self.config.horizons, self.config.targets)
        presence_shape = (self.config.horizons, self.config.presence_targets)
        self.forecast_counts = torch.zeros(forecast_shape, dtype=torch.int64)
        self.pinball_sum = torch.zeros(forecast_shape, dtype=torch.float64)
        self.absolute_error_sum = torch.zeros(forecast_shape, dtype=torch.float64)
        self.squared_error_sum = torch.zeros(forecast_shape, dtype=torch.float64)
        self.signed_error_sum = torch.zeros(forecast_shape, dtype=torch.float64)
        self.covered_sum = torch.zeros(forecast_shape, dtype=torch.float64)
        self.width_sum = torch.zeros(forecast_shape, dtype=torch.float64)
        self.presence_counts = torch.zeros(presence_shape, dtype=torch.int64)
        self.presence_log_loss_sum = torch.zeros(presence_shape, dtype=torch.float64)
        self.presence_brier_sum = torch.zeros(presence_shape, dtype=torch.float64)

    def update(
        self,
        output: ChmtQOutput,
        *,
        targets: Tensor | None,
        target_mask: Tensor | None,
        presence_labels: Tensor | None,
        presence_mask: Tensor | None,
    ) -> ChmtQMetricsAccumulator:
        output.validate(self.config)
        self._update_forecast(output, targets, target_mask)
        self._update_presence(output, presence_labels, presence_mask)
        return self

    def finalize(self) -> ChmtQMetricReport:
        forecast_denominator = self.forecast_counts.to(torch.float64)
        pinball_denominator = forecast_denominator * self.config.quantiles
        pinball = _safe_divide(self.pinball_sum, pinball_denominator)
        median_mae = _safe_divide(self.absolute_error_sum, forecast_denominator)
        median_rmse = torch.sqrt(
            _safe_divide(self.squared_error_sum, forecast_denominator)
        )
        median_bias = _safe_divide(self.signed_error_sum, forecast_denominator)
        interval_coverage = _safe_divide(self.covered_sum, forecast_denominator)
        interval_width = _safe_divide(self.width_sum, forecast_denominator)
        presence_denominator = self.presence_counts.to(torch.float64)
        presence_log_loss = _safe_divide(
            self.presence_log_loss_sum, presence_denominator
        )
        presence_brier = _safe_divide(self.presence_brier_sum, presence_denominator)

        horizon_weights = torch.tensor(
            self.config.horizon_weights, dtype=torch.float64
        ).view(-1, 1)
        target_weights = torch.tensor(
            self.config.target_weights, dtype=torch.float64
        ).view(1, -1)
        primary_weights = horizon_weights * target_weights
        primary_numerator = (self.pinball_sum * primary_weights).sum()
        primary_denominator = (pinball_denominator * primary_weights).sum()
        weighted_primary = float(
            _safe_divide(primary_numerator, primary_denominator).item()
        )
        return ChmtQMetricReport(
            forecast_counts=self.forecast_counts.clone(),
            pinball=pinball,
            median_mae=median_mae,
            median_rmse=median_rmse,
            median_bias=median_bias,
            interval_coverage=interval_coverage,
            interval_width=interval_width,
            presence_counts=self.presence_counts.clone(),
            presence_log_loss=presence_log_loss,
            presence_brier=presence_brier,
            weighted_primary_metric=weighted_primary,
        )

    def _update_forecast(
        self,
        output: ChmtQOutput,
        targets: Tensor | None,
        mask: Tensor | None,
    ) -> None:
        present = targets is not None or mask is not None
        if not present:
            return
        if targets is None or mask is None:
            raise ValueError("targets and target_mask must both be set or both be None")
        expected = (
            output.quantiles.shape[0],
            self.config.horizons,
            self.config.targets,
        )
        _validate_labels(targets, mask, expected, output.quantiles.device, "target")
        target_values = targets.detach().to(device="cpu", dtype=torch.float64)
        valid = mask.detach().to(device="cpu")
        quantiles = output.quantiles.detach().to(device="cpu", dtype=torch.float64)
        safe_targets = torch.where(valid, target_values, quantiles[..., 1])
        error = safe_targets.unsqueeze(-1) - quantiles
        levels = torch.tensor(self.config.quantile_levels, dtype=torch.float64)
        pinball_cells = torch.maximum(levels * error, (levels - 1.0) * error)
        self.pinball_sum += (pinball_cells * valid.unsqueeze(-1)).sum(dim=(0, 3))

        median_error = quantiles[..., 1] - safe_targets
        valid_float = valid.to(torch.float64)
        self.absolute_error_sum += (median_error.abs() * valid_float).sum(dim=0)
        self.squared_error_sum += (median_error.square() * valid_float).sum(dim=0)
        self.signed_error_sum += (median_error * valid_float).sum(dim=0)
        covered = (quantiles[..., 0] <= safe_targets) & (
            safe_targets <= quantiles[..., 2]
        )
        self.covered_sum += (covered & valid).sum(dim=0).to(torch.float64)
        width = quantiles[..., 2] - quantiles[..., 0]
        self.width_sum += (width * valid_float).sum(dim=0)
        self.forecast_counts += valid.sum(dim=0, dtype=torch.int64)

    def _update_presence(
        self,
        output: ChmtQOutput,
        labels: Tensor | None,
        mask: Tensor | None,
    ) -> None:
        present = labels is not None or mask is not None
        if not present:
            return
        if labels is None or mask is None:
            raise ValueError(
                "presence_labels and presence_mask must both be set or both be None"
            )
        expected = (
            output.presence_logits.shape[0],
            self.config.horizons,
            self.config.presence_targets,
        )
        _validate_labels(
            labels, mask, expected, output.presence_logits.device, "presence"
        )
        label_values = labels.detach().to(device="cpu", dtype=torch.float64)
        valid = mask.detach().to(device="cpu")
        supervised = label_values[valid]
        if supervised.numel() and not bool(
            ((supervised == 0.0) | (supervised == 1.0)).all().item()
        ):
            raise ValueError("supervised presence labels must be binary")
        logits = output.presence_logits.detach().to(device="cpu", dtype=torch.float64)
        safe_labels = torch.where(valid, label_values, torch.zeros_like(label_values))
        log_loss = F.binary_cross_entropy_with_logits(
            logits, safe_labels, reduction="none"
        )
        valid_float = valid.to(torch.float64)
        self.presence_log_loss_sum += (log_loss * valid_float).sum(dim=0)
        self.presence_brier_sum += (
            (torch.sigmoid(logits) - safe_labels).square() * valid_float
        ).sum(dim=0)
        self.presence_counts += valid.sum(dim=0, dtype=torch.int64)


def _validate_labels(
    values: Tensor,
    mask: Tensor,
    expected_shape: tuple[int, ...],
    expected_device: torch.device,
    name: str,
) -> None:
    if not isinstance(values, Tensor) or not isinstance(mask, Tensor):
        raise TypeError(f"{name} labels and mask must be torch.Tensor instances")
    if tuple(values.shape) != expected_shape or tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} labels and mask must have shape {expected_shape}")
    if not values.is_floating_point() or mask.dtype != torch.bool:
        raise TypeError(f"{name} labels must be floating point and mask must be bool")
    if values.device != expected_device or mask.device != expected_device:
        raise ValueError(f"{name} labels, mask, and output must use one device")
    finite = torch.where(mask, torch.isfinite(values), torch.ones_like(mask))
    if not bool(finite.all().item()):
        raise ValueError(f"{name} labels contain NaN or infinity at a supervised cell")


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    return torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )
