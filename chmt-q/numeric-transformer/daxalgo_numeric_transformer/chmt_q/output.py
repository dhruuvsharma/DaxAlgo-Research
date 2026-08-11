"""Monotone probabilistic output contract for CHMT-Q v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import ChmtQConfig


@dataclass(frozen=True, slots=True)
class ChmtQOutput:
    """Ordered marginal quantiles and independent buy/sell presence logits."""

    quantiles: Tensor
    presence_logits: Tensor

    @property
    def q10(self) -> Tensor:
        return self.quantiles[..., 0]

    @property
    def median(self) -> Tensor:
        return self.quantiles[..., 1]

    @property
    def q90(self) -> Tensor:
        return self.quantiles[..., 2]

    def validate(self, config: ChmtQConfig | None = None) -> ChmtQOutput:
        """Validate shapes, common dtype/device, finiteness, order, and delta bounds."""

        cfg = config or ChmtQConfig()
        if not isinstance(self.quantiles, Tensor) or not isinstance(
            self.presence_logits, Tensor
        ):
            raise TypeError("quantiles and presence_logits must be torch.Tensor instances")
        if self.quantiles.ndim != 4:
            raise ValueError("quantiles must have shape [B,5,5,3]")
        batch_size = self.quantiles.shape[0]
        if batch_size <= 0:
            raise ValueError("output batch size must be positive")
        expected_quantiles = (batch_size, cfg.horizons, cfg.targets, cfg.quantiles)
        expected_presence = (batch_size, cfg.horizons, cfg.presence_targets)
        if tuple(self.quantiles.shape) != expected_quantiles:
            raise ValueError(f"quantiles must have shape {expected_quantiles}")
        if tuple(self.presence_logits.shape) != expected_presence:
            raise ValueError(f"presence_logits must have shape {expected_presence}")
        if not self.quantiles.is_floating_point() or (
            self.presence_logits.dtype != self.quantiles.dtype
        ):
            raise TypeError("output tensors must share one floating-point dtype")
        if self.presence_logits.device != self.quantiles.device:
            raise ValueError("output tensors must use one device")
        if not bool(torch.isfinite(self.quantiles).all().item()) or not bool(
            torch.isfinite(self.presence_logits).all().item()
        ):
            raise ValueError("output contains NaN or infinity")
        if bool((self.q10 > self.median).any().item()) or bool(
            (self.median > self.q90).any().item()
        ):
            raise ValueError("quantiles must satisfy q10 <= q50 <= q90")
        delta = self.quantiles[..., cfg.delta_target_index, :]
        if bool(((delta < -1.0) | (delta > 1.0)).any().item()):
            raise ValueError("delta-fraction quantiles must remain in [-1, 1]")
        return self


def transform_raw_output(
    raw: Tensor, config: ChmtQConfig | None = None
) -> ChmtQOutput:
    """Convert ``[B,5,17]`` raw values to ordered quantiles and presence logits."""

    cfg = config or ChmtQConfig()
    if not isinstance(raw, Tensor):
        raise TypeError("raw output must be a torch.Tensor")
    if raw.ndim != 3:
        raise ValueError("raw output must have shape [B,5,17]")
    expected = (raw.shape[0], cfg.horizons, cfg.raw_output_features)
    if tuple(raw.shape) != expected:
        raise ValueError(f"raw output must have shape {expected}")
    if not raw.is_floating_point():
        raise TypeError("raw output must use a floating-point dtype")
    if not bool(torch.isfinite(raw).all().item()):
        raise ValueError("raw output contains NaN or infinity")

    transform_values = (
        raw.float() if raw.dtype in (torch.float16, torch.bfloat16) else raw
    )
    parameters = transform_values[..., : cfg.targets * cfg.quantiles].reshape(
        raw.shape[0], cfg.horizons, cfg.targets, cfg.quantiles
    )
    center = parameters[..., 0]
    lower = center - F.softplus(parameters[..., 1])
    upper = center + F.softplus(parameters[..., 2])
    ordered = torch.stack((lower, center, upper), dim=-1)

    before_delta = ordered[..., : cfg.delta_target_index, :]
    delta = torch.tanh(ordered[..., cfg.delta_target_index :, :])
    quantiles = torch.cat((before_delta, delta), dim=-2)
    presence_logits = transform_values[..., cfg.targets * cfg.quantiles :].to(
        dtype=quantiles.dtype
    )
    return ChmtQOutput(quantiles, presence_logits).validate(cfg)
