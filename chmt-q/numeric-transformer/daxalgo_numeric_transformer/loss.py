"""Masked multi-horizon regression objective."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


class NoValidTargetsError(RuntimeError):
    """Signals that a training batch has no supervised cells and must be skipped."""


def masked_huber_loss(
    predictions: Tensor,
    targets: Tensor,
    target_mask: Tensor,
    *,
    delta: float = 1.0,
    horizon_weights: Sequence[float] | Tensor | None = None,
    head_weights: Sequence[float] | Tensor | None = None,
) -> Tensor:
    """Return the architecture-specified masked Huber mean.

    Each horizon/head cell is averaged over its valid batch rows before the
    horizon and head weights are applied. A wholly masked batch raises
    :class:`NoValidTargetsError` so the trainer cannot count it as a successful
    optimizer or scheduler step.
    """

    if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
        raise TypeError("predictions and targets must be torch.Tensor instances")
    if not isinstance(target_mask, Tensor):
        raise TypeError("target_mask must be a torch.Tensor")
    if predictions.shape != targets.shape or predictions.shape != target_mask.shape:
        raise ValueError(
            "predictions, targets, and target_mask must have identical [B,H,K] shapes"
        )
    if predictions.ndim != 3:
        raise ValueError("predictions must have shape [B,H,K]")
    if predictions.shape[1:] != (5, 5):
        raise ValueError(
            f"predictions must have fixed trailing shape (5, 5), got {predictions.shape[1:]}"
        )
    if not predictions.is_floating_point() or targets.dtype != predictions.dtype:
        raise TypeError("predictions and targets must share a floating-point dtype")
    if target_mask.dtype != torch.bool:
        raise TypeError("target_mask must have dtype torch.bool")
    if predictions.device != targets.device or predictions.device != target_mask.device:
        raise ValueError("loss tensors must be on the same device")
    if delta <= 0.0 or not math.isfinite(delta):
        raise ValueError("delta must be finite and positive")
    if not bool(torch.isfinite(predictions).all().item()):
        raise ValueError("predictions contain NaN or infinity")
    valid_targets_finite = torch.where(
        target_mask,
        torch.isfinite(targets),
        torch.ones_like(target_mask),
    )
    if not bool(valid_targets_finite.all().item()):
        raise ValueError("targets contain NaN or infinity at a valid target")

    horizons = predictions.shape[1]
    heads = predictions.shape[2]
    horizon_weight = _weight_tensor(
        horizon_weights,
        default=[1.0 / math.sqrt(index + 1.0) for index in range(horizons)],
        expected=horizons,
        reference=predictions,
        name="horizon_weights",
    )
    head_weight = _weight_tensor(
        head_weights,
        default=[1.0, 1.0, 1.0, 0.5, 0.5],
        expected=heads,
        reference=predictions,
        name="head_weights",
    )

    safe_targets = torch.where(target_mask, targets, predictions.detach())
    absolute_error = (predictions - safe_targets).abs()
    quadratic = 0.5 * absolute_error.square()
    linear = delta * (absolute_error - 0.5 * delta)
    element_loss = torch.where(absolute_error <= delta, quadratic, linear)
    element_loss = torch.where(target_mask, element_loss, torch.zeros_like(element_loss))

    counts = target_mask.sum(dim=0)
    valid_cells = counts > 0
    cell_means = element_loss.sum(dim=0) / counts.clamp_min(1).to(predictions.dtype)
    weights = horizon_weight[:, None] * head_weight[None, :]
    active_weights = torch.where(valid_cells, weights, torch.zeros_like(weights))
    numerator = (cell_means * active_weights).sum()
    denominator = active_weights.sum()
    if not bool((denominator > 0).item()):
        raise NoValidTargetsError(
            "training batch has no valid target cells with positive loss weight"
        )
    return numerator / denominator


def _weight_tensor(
    values: Sequence[float] | Tensor | None,
    *,
    default: Sequence[float],
    expected: int,
    reference: Tensor,
    name: str,
) -> Tensor:
    source: Sequence[float] | Tensor = default if values is None else values
    result = torch.as_tensor(source, dtype=reference.dtype, device=reference.device)
    if result.ndim != 1 or result.numel() != expected:
        raise ValueError(f"{name} must contain exactly {expected} values")
    if not bool(torch.isfinite(result).all().item()) or bool((result < 0).any().item()):
        raise ValueError(f"{name} must contain finite non-negative values")
    return result


class MaskedHuberLoss(nn.Module):
    """``nn.Module`` wrapper around :func:`masked_huber_loss`."""

    def __init__(
        self,
        *,
        delta: float = 1.0,
        horizon_weights: Sequence[float] | None = None,
        head_weights: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.horizon_weights = horizon_weights
        self.head_weights = head_weights

    def forward(self, predictions: Tensor, targets: Tensor, target_mask: Tensor) -> Tensor:
        return masked_huber_loss(
            predictions,
            targets,
            target_mask,
            delta=self.delta,
            horizon_weights=self.horizon_weights,
            head_weights=self.head_weights,
        )
