"""Tensor contracts shared by the model, trainer, and exporter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import ModelConfig


class NoValidModalitiesError(ValueError):
    """Signals that one or more samples have no historical modality to decode."""


def _is_exporting() -> bool:
    return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()


@dataclass(slots=True)
class MarketBatch:
    """One batch in the fixed numeric-market tensor contract."""

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

    @property
    def batch_size(self) -> int:
        if self.bars.ndim == 0:
            raise ValueError("bars must have a batch dimension")
        return int(self.bars.shape[0])

    @property
    def device(self) -> torch.device:
        return self.bars.device

    def validate(
        self,
        config: ModelConfig | None = None,
        *,
        require_targets: bool = False,
        check_finite: bool = True,
    ) -> MarketBatch:
        """Validate shapes, dtypes, devices, finiteness, and nested L2 masks."""

        cfg = config or ModelConfig.reference()
        tensors: dict[str, Tensor | None] = {
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
        }
        for name, value in tensors.items():
            if value is not None and not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor, got {type(value)!r}")

        batch_size = self.batch_size
        if batch_size <= 0:
            raise ValueError("batch size must be positive")

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

        numeric_names = ("bars", "l1", "l2", "quality")
        mask_names = (
            "bars_mask",
            "l1_mask",
            "l2_level_mask",
            "l2_snapshot_mask",
        )
        for name in numeric_names:
            tensor = getattr(self, name)
            if tensor.dtype != torch.float32:
                raise TypeError(f"{name} must have dtype torch.float32, got {tensor.dtype}")
        for name in mask_names:
            tensor = getattr(self, name)
            if tensor.dtype != torch.bool:
                raise TypeError(f"{name} must have dtype torch.bool, got {tensor.dtype}")

        has_targets = self.targets is not None or self.target_mask is not None
        if has_targets and (self.targets is None or self.target_mask is None):
            raise ValueError("targets and target_mask must either both be set or both be None")
        if require_targets and not has_targets:
            raise ValueError("targets and target_mask are required")
        if self.targets is not None and self.target_mask is not None:
            target_shape = (batch_size, cfg.horizons, cfg.output_features)
            if tuple(self.targets.shape) != target_shape:
                raise ValueError(
                    f"targets must have shape {target_shape}, got {tuple(self.targets.shape)}"
                )
            if tuple(self.target_mask.shape) != target_shape:
                raise ValueError(
                    "target_mask must have shape "
                    f"{target_shape}, got {tuple(self.target_mask.shape)}"
                )
            if self.targets.dtype != torch.float32:
                raise TypeError(
                    f"targets must have dtype torch.float32, got {self.targets.dtype}"
                )
            if self.target_mask.dtype != torch.bool:
                raise TypeError(
                    "target_mask must have dtype torch.bool, "
                    f"got {self.target_mask.dtype}"
                )

        devices = {
            value.device
            for value in tensors.values()
            if isinstance(value, Tensor)
        }
        if len(devices) != 1:
            rendered = ", ".join(sorted(str(device) for device in devices))
            raise ValueError(f"all tensors must use one device, got: {rendered}")

        if not _is_exporting():
            effective_snapshot_mask = self.l2_level_mask.any(dim=(-1, -2))
            if not torch.equal(self.l2_snapshot_mask, effective_snapshot_mask):
                raise ValueError(
                    "l2_snapshot_mask must exactly equal any(l2_level_mask, side/rank)"
                )

            if check_finite:
                for name in numeric_names:
                    if not bool(torch.isfinite(getattr(self, name)).all().item()):
                        raise ValueError(f"{name} contains NaN or infinity")
                if self.targets is not None and self.target_mask is not None:
                    valid_targets_finite = torch.where(
                        self.target_mask,
                        torch.isfinite(self.targets),
                        torch.ones_like(self.target_mask),
                    )
                    if not bool(valid_targets_finite.all().item()):
                        raise ValueError("targets contain NaN or infinity at a valid target")

        return self

    def as_model_kwargs(self) -> dict[str, Tensor]:
        """Return only the eight inference inputs, keyed by stable ONNX names."""

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

    def modality_availability(self) -> Tensor:
        """Return per-sample bars/L1/L2 availability as a ``[B,3]`` bool tensor."""

        return torch.stack(
            (
                self.bars_mask.any(dim=1),
                self.l1_mask.any(dim=1),
                self.l2_snapshot_mask.any(dim=1),
            ),
            dim=1,
        )

    def require_inference_available(self) -> MarketBatch:
        """Reject samples that must abstain before horizon decoding."""

        if _is_exporting():
            return self
        available = self.modality_availability().any(dim=1)
        if not bool(available.all().item()):
            missing = (~available).nonzero(as_tuple=False).flatten().tolist()
            raise NoValidModalitiesError(
                f"inference requires at least one valid modality per sample; missing={missing}"
            )
        return self

    def model_inputs(self) -> tuple[Tensor, ...]:
        """Return inference inputs in the stable positional export order."""

        return tuple(self.as_model_kwargs().values())

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> MarketBatch:
        """Return a batch with every present tensor moved to ``device``."""

        def move(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.to(device=device, non_blocking=non_blocking)

        return MarketBatch(
            bars=move(self.bars),  # type: ignore[arg-type]
            bars_mask=move(self.bars_mask),  # type: ignore[arg-type]
            l1=move(self.l1),  # type: ignore[arg-type]
            l1_mask=move(self.l1_mask),  # type: ignore[arg-type]
            l2=move(self.l2),  # type: ignore[arg-type]
            l2_level_mask=move(self.l2_level_mask),  # type: ignore[arg-type]
            l2_snapshot_mask=move(self.l2_snapshot_mask),  # type: ignore[arg-type]
            quality=move(self.quality),  # type: ignore[arg-type]
            targets=move(self.targets),
            target_mask=move(self.target_mask),
        )

    def pin_memory(self) -> MarketBatch:
        """Return a CPU batch whose present tensors use pinned memory."""

        def pin(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.pin_memory()

        return MarketBatch(
            bars=pin(self.bars),  # type: ignore[arg-type]
            bars_mask=pin(self.bars_mask),  # type: ignore[arg-type]
            l1=pin(self.l1),  # type: ignore[arg-type]
            l1_mask=pin(self.l1_mask),  # type: ignore[arg-type]
            l2=pin(self.l2),  # type: ignore[arg-type]
            l2_level_mask=pin(self.l2_level_mask),  # type: ignore[arg-type]
            l2_snapshot_mask=pin(self.l2_snapshot_mask),  # type: ignore[arg-type]
            quality=pin(self.quality),  # type: ignore[arg-type]
            targets=pin(self.targets),
            target_mask=pin(self.target_mask),
        )
