"""Tensor and supervision contracts for CHMT-Q v1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ChmtQConfig


class NoAvailableModalitiesError(ValueError):
    """Raised when a sample has no usable historical modality."""


@dataclass(slots=True)
class ChmtQBatch:
    """Fixed CHMT-Q inputs with independent forecast and presence supervision."""

    bars: Tensor
    bars_mask: Tensor
    l1: Tensor
    l1_mask: Tensor
    l2: Tensor
    l2_level_mask: Tensor
    l2_snapshot_mask: Tensor
    quality: Tensor
    targets: Tensor | None = None
    target_mask: Tensor | None = None
    presence_labels: Tensor | None = None
    presence_mask: Tensor | None = None

    @property
    def batch_size(self) -> int:
        if not isinstance(self.bars, Tensor):
            raise TypeError("bars must be a torch.Tensor")
        if self.bars.ndim == 0:
            raise ValueError("bars must have a batch dimension")
        return int(self.bars.shape[0])

    @property
    def device(self) -> torch.device:
        return self.bars.device

    def validate(
        self,
        config: ChmtQConfig | None = None,
        *,
        require_targets: bool = False,
        require_presence: bool = False,
        require_supervision: bool = False,
        require_available: bool = True,
        require_feature_domains: bool = False,
    ) -> ChmtQBatch:
        """Validate fixed shapes, dtypes, devices, finiteness, and mask invariants."""

        cfg = config or ChmtQConfig()
        batch_size = self.batch_size
        if batch_size <= 0:
            raise ValueError("batch size must be positive")

        tensors = self._tensor_map()
        for name, value in tensors.items():
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")

        expected_shapes = {
            "bars": (batch_size, cfg.bar_steps, cfg.bar_features),
            "bars_mask": (batch_size, cfg.bar_steps),
            "l1": (batch_size, cfg.micro_steps, cfg.l1_features),
            "l1_mask": (batch_size, cfg.micro_steps),
            "l2": (
                batch_size,
                cfg.micro_steps,
                cfg.l2_sides,
                cfg.l2_levels,
                cfg.l2_features,
            ),
            "l2_level_mask": (
                batch_size,
                cfg.micro_steps,
                cfg.l2_sides,
                cfg.l2_levels,
            ),
            "l2_snapshot_mask": (batch_size, cfg.micro_steps),
            "quality": (batch_size, cfg.quality_features),
        }
        for name, shape in expected_shapes.items():
            actual = tuple(getattr(self, name).shape)
            if actual != shape:
                raise ValueError(f"{name} must have shape {shape}, got {actual}")

        for name in ("bars", "l1", "l2", "quality"):
            value = getattr(self, name)
            if value.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} contains NaN or infinity")
        if bool(((self.quality < 0.0) | (self.quality > 1.0)).any().item()):
            raise ValueError("quality rates, flags, and session progress must be in [0, 1]")
        if require_feature_domains:
            self._validate_feature_domains()
        for name in ("bars_mask", "l1_mask", "l2_level_mask", "l2_snapshot_mask"):
            if getattr(self, name).dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool")

        devices = {
            value.device for value in tensors.values() if isinstance(value, Tensor)
        }
        if len(devices) != 1:
            rendered = ", ".join(sorted(str(device) for device in devices))
            raise ValueError(f"all tensors must use one device, got: {rendered}")

        self._validate_forecast_supervision(cfg, require_targets)
        self._validate_presence_supervision(cfg, require_presence)

        effective_snapshot_mask = self.l2_level_mask.any(dim=(-1, -2))
        if not torch.equal(self.l2_snapshot_mask, effective_snapshot_mask):
            raise ValueError(
                "l2_snapshot_mask must exactly equal any(l2_level_mask, side/rank)"
            )

        if require_available:
            self.require_available()
        if require_supervision and not self.has_supervision():
            raise ValueError("at least one forecast or presence label must be supervised")
        return self

    def _validate_feature_domains(self) -> None:
        """Enforce unscaled bounded fields at the compiled-dataset boundary."""

        for index, lower, upper in (
            (9, 0.0, 1.0),
            (13, 0.0, 1.0),
            (14, -1.0, 1.0),
            (15, -1.0, 1.0),
        ):
            _validate_bounded_feature("bars", self.bars, self.bars_mask, index, lower, upper)
        for index, lower, upper in (
            (5, -1.0, 1.0),
            (13, 0.0, 1.0),
        ):
            _validate_bounded_feature("l1", self.l1, self.l1_mask, index, lower, upper)
        for index in (14, 15):
            _validate_binary_feature("l1", self.l1, self.l1_mask, index)
        for index, lower, upper in (
            (4, 0.0, 1.0),
            (5, 0.0, 1.0),
            (6, -1.0, 1.0),
            (9, 0.0, 1.0),
        ):
            _validate_bounded_feature(
                "l2", self.l2, self.l2_level_mask, index, lower, upper
            )
        for index in (10, 11):
            _validate_binary_feature("l2", self.l2, self.l2_level_mask, index)

    def _tensor_map(self) -> dict[str, Tensor | None]:
        return {
            "bars": self.bars,
            "bars_mask": self.bars_mask,
            "l1": self.l1,
            "l1_mask": self.l1_mask,
            "l2": self.l2,
            "l2_level_mask": self.l2_level_mask,
            "l2_snapshot_mask": self.l2_snapshot_mask,
            "quality": self.quality,
            "targets": self.targets,
            "target_mask": self.target_mask,
            "presence_labels": self.presence_labels,
            "presence_mask": self.presence_mask,
        }

    def _validate_forecast_supervision(
        self, config: ChmtQConfig, require_targets: bool
    ) -> None:
        present = self.targets is not None or self.target_mask is not None
        if present and (self.targets is None or self.target_mask is None):
            raise ValueError("targets and target_mask must either both be set or both be None")
        if require_targets and not present:
            raise ValueError("targets and target_mask are required")
        if self.targets is None or self.target_mask is None:
            return

        shape = (self.batch_size, config.horizons, config.targets)
        if tuple(self.targets.shape) != shape or tuple(self.target_mask.shape) != shape:
            raise ValueError(f"targets and target_mask must have shape {shape}")
        if self.targets.dtype != torch.float32:
            raise TypeError("targets must have dtype torch.float32")
        if self.target_mask.dtype != torch.bool:
            raise TypeError("target_mask must have dtype torch.bool")
        finite = torch.where(
            self.target_mask,
            torch.isfinite(self.targets),
            torch.ones_like(self.target_mask),
        )
        if not bool(finite.all().item()):
            raise ValueError("targets contain NaN or infinity at a supervised cell")
        delta_mask = self.target_mask[..., config.delta_target_index]
        valid_delta = self.targets[..., config.delta_target_index][delta_mask]
        if valid_delta.numel() and not bool(
            ((valid_delta >= -1.0) & (valid_delta <= 1.0)).all().item()
        ):
            raise ValueError("supervised delta-fraction targets must be in [-1, 1]")

    def _validate_presence_supervision(
        self, config: ChmtQConfig, require_presence: bool
    ) -> None:
        present = self.presence_labels is not None or self.presence_mask is not None
        if present and (self.presence_labels is None or self.presence_mask is None):
            raise ValueError(
                "presence_labels and presence_mask must either both be set or both be None"
            )
        if require_presence and not present:
            raise ValueError("presence_labels and presence_mask are required")
        if self.presence_labels is None or self.presence_mask is None:
            return

        shape = (self.batch_size, config.horizons, config.presence_targets)
        if (
            tuple(self.presence_labels.shape) != shape
            or tuple(self.presence_mask.shape) != shape
        ):
            raise ValueError(f"presence labels and mask must have shape {shape}")
        if self.presence_labels.dtype != torch.float32:
            raise TypeError("presence_labels must have dtype torch.float32")
        if self.presence_mask.dtype != torch.bool:
            raise TypeError("presence_mask must have dtype torch.bool")
        finite = torch.where(
            self.presence_mask,
            torch.isfinite(self.presence_labels),
            torch.ones_like(self.presence_mask),
        )
        if not bool(finite.all().item()):
            raise ValueError("presence labels contain NaN or infinity at a supervised cell")
        valid_labels = self.presence_labels[self.presence_mask]
        if valid_labels.numel() and not bool(
            ((valid_labels == 0.0) | (valid_labels == 1.0)).all().item()
        ):
            raise ValueError("supervised presence labels must be binary")

    def modality_availability(self) -> Tensor:
        """Return hard bars/L1/L2 availability for each sample as ``[B,3]``."""

        return torch.stack(
            (
                self.bars_mask.any(dim=1),
                self.l1_mask.any(dim=1),
                self.l2_snapshot_mask.any(dim=1),
            ),
            dim=1,
        )

    def require_available(self) -> ChmtQBatch:
        """Reject every sample that has no valid historical modality."""

        available = self.modality_availability().any(dim=1)
        if not bool(available.all().item()):
            missing = (~available).nonzero(as_tuple=False).flatten().tolist()
            raise NoAvailableModalitiesError(
                f"at least one modality is required per sample; missing={missing}"
            )
        return self

    def has_supervision(self) -> bool:
        """Return whether any forecast or presence cell is supervised."""

        target_valid = self.target_mask is not None and bool(self.target_mask.any().item())
        presence_valid = self.presence_mask is not None and bool(self.presence_mask.any().item())
        return target_valid or presence_valid

    def as_model_kwargs(self) -> dict[str, Tensor]:
        """Return the eight inference inputs in their stable external order."""

        return {
            "bars": self.bars,
            "bars_mask": self.bars_mask,
            "l1": self.l1,
            "l1_mask": self.l1_mask,
            "l2": self.l2,
            "l2_level_mask": self.l2_level_mask,
            "l2_snapshot_mask": self.l2_snapshot_mask,
            "quality": self.quality,
        }


def _validate_bounded_feature(
    name: str,
    values: Tensor,
    valid_mask: Tensor,
    index: int,
    lower: float,
    upper: float,
) -> None:
    selected = values[..., index][valid_mask]
    if selected.numel() and not bool(
        ((selected >= lower) & (selected <= upper)).all().item()
    ):
        raise ValueError(f"{name} feature {index} must be in [{lower:g}, {upper:g}]")


def _validate_binary_feature(
    name: str, values: Tensor, valid_mask: Tensor, index: int
) -> None:
    selected = values[..., index][valid_mask]
    if selected.numel() and not bool(
        ((selected == 0.0) | (selected == 1.0)).all().item()
    ):
        raise ValueError(f"{name} feature {index} must be binary")
