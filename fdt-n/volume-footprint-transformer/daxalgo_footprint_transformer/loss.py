from __future__ import annotations

import torch
from torch import Tensor

from .config import TARGET_NAMES

QUANTILES = (0.1, 0.5, 0.9)
TARGET_WEIGHTS = (1.0, 0.7, 0.7, 0.8, 0.8, 0.5, 0.7)


def quantile_loss(predictions: Tensor, targets: Tensor, target_mask: Tensor) -> Tensor:
    """Weighted direct multi-horizon pinball loss with explicit missing-target masks."""

    if predictions.shape[-2:] != (len(TARGET_NAMES), len(QUANTILES)):
        raise ValueError("unexpected prediction shape")
    errors = targets.unsqueeze(-1) - predictions
    quantiles = predictions.new_tensor(QUANTILES)
    pinball = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    horizon = torch.arange(
        1, predictions.shape[1] + 1, device=predictions.device, dtype=predictions.dtype
    )
    horizon_weight = torch.rsqrt(horizon).view(1, -1, 1, 1)
    target_weight = predictions.new_tensor(TARGET_WEIGHTS).view(1, 1, -1, 1)
    mask = target_mask.unsqueeze(-1).to(predictions.dtype)
    weighted = pinball * horizon_weight * target_weight * mask
    denominator = (horizon_weight * target_weight * mask).sum() * len(QUANTILES)
    return weighted.sum() / denominator.clamp_min(1.0)


def median_mae_by_target(predictions: Tensor, targets: Tensor, target_mask: Tensor) -> Tensor:
    absolute = (predictions[..., 1] - targets).abs()
    mask = target_mask.to(absolute.dtype)
    return (absolute * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1.0)


def central_coverage_by_target(predictions: Tensor, targets: Tensor, target_mask: Tensor) -> Tensor:
    inside = (targets >= predictions[..., 0]) & (targets <= predictions[..., 2])
    mask = target_mask.to(predictions.dtype)
    return (inside.to(predictions.dtype) * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(
        1.0
    )
