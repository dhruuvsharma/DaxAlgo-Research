"""Fail-closed host reconstruction of one fixed-batch CHMT-Q forecast."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ChmtQConfig
from .output import ChmtQOutput
from .scaling import RobustScalerArtifact


@dataclass(frozen=True, slots=True)
class ReconstructedForecast:
    """Validated physical primitive marginals plus presence and plug-in delta."""

    primitive_quantiles: Tensor
    presence_probabilities: Tensor
    presence_decisions: Tensor
    raw_delta_median: Tensor


@dataclass(frozen=True, slots=True)
class ReconstructionResult:
    forecast: ReconstructedForecast | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if (self.forecast is None) == (self.unavailable_reason is None):
            raise ValueError("exactly one of forecast or unavailable_reason must be set")


def reconstruct_forecast(
    output: ChmtQOutput,
    scaler: RobustScalerArtifact,
    *,
    anchor_price: float,
    log_volume_reference: float,
    tick_size: float,
    presence_thresholds: tuple[float, float] = (0.5, 0.5),
    maximum_absolute_offset_ticks: float = 1_000_000.0,
    maximum_volume: float = 1.0e15,
    config: ChmtQConfig | None = None,
) -> ReconstructionResult:
    """Reconstruct one forecast atomically; any invalid primitive rejects all horizons."""

    cfg = config or ChmtQConfig.reference()
    try:
        output.validate(cfg)
        if output.quantiles.shape[0] != 1:
            raise ValueError("host reconstruction requires fixed batch size one")
        for value, name in (
            (anchor_price, "anchor_price"),
            (log_volume_reference, "log_volume_reference"),
            (tick_size, "tick_size"),
            (maximum_absolute_offset_ticks, "maximum_absolute_offset_ticks"),
            (maximum_volume, "maximum_volume"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if anchor_price <= 0.0 or tick_size <= 0.0:
            raise ValueError("anchor_price and tick_size must be positive")
        if maximum_absolute_offset_ticks <= 0.0 or maximum_volume <= 0.0:
            raise ValueError("reconstruction bounds must be positive")
        if len(presence_thresholds) != cfg.presence_targets or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in presence_thresholds
        ):
            raise ValueError("presence thresholds must contain two values in [0, 1]")

        normalized = output.quantiles.permute(0, 3, 1, 2)
        physical = scaler.denormalize_targets(normalized).permute(0, 2, 3, 1)
        offsets = physical[..., :3, :]
        if bool((offsets.abs() > maximum_absolute_offset_ticks).any().item()):
            raise ValueError("price offset exceeds the declared reconstruction bound")
        price_quantiles = anchor_price + offsets * tick_size
        price_quantiles = torch.round(price_quantiles / tick_size) * tick_size
        if bool((price_quantiles <= 0.0).any().item()):
            raise ValueError("reconstructed price is not positive")

        log_volume = physical[..., 3:4, :] + log_volume_reference
        volume_quantiles = torch.expm1(log_volume).clamp_min(0.0)
        if bool((volume_quantiles > maximum_volume).any().item()):
            raise ValueError("reconstructed volume exceeds the declared bound")
        delta_quantiles = physical[..., 4:5, :]
        if bool(((delta_quantiles < -1.0) | (delta_quantiles > 1.0)).any().item()):
            raise ValueError("reconstructed delta fraction is outside [-1, 1]")
        primitives = torch.cat(
            (price_quantiles, volume_quantiles, delta_quantiles), dim=-2
        )
        if not bool(torch.isfinite(primitives).all().item()):
            raise ValueError("reconstructed primitive contains NaN or infinity")
        if bool((primitives[..., 0] > primitives[..., 1]).any().item()) or bool(
            (primitives[..., 1] > primitives[..., 2]).any().item()
        ):
            raise ValueError("reconstructed primitive quantiles are unordered")

        probabilities = torch.sigmoid(output.presence_logits)
        thresholds = probabilities.new_tensor(presence_thresholds).view(1, 1, -1)
        decisions = probabilities >= thresholds
        raw_delta = primitives[..., 3, 1] * primitives[..., 4, 1]
        if not bool(torch.isfinite(raw_delta).all().item()):
            raise ValueError("plug-in raw delta is not finite")
        return ReconstructionResult(
            forecast=ReconstructedForecast(
                primitive_quantiles=primitives,
                presence_probabilities=probabilities,
                presence_decisions=decisions,
                raw_delta_median=raw_delta,
            ),
            unavailable_reason=None,
        )
    except (OverflowError, RuntimeError, TypeError, ValueError) as error:
        return ReconstructionResult(forecast=None, unavailable_reason=str(error))
