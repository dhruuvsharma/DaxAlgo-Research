"""Configuration for the Pro-private numeric market transformer."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Immutable architecture and fixed tensor-contract configuration."""

    # Fixed external tensor contract.
    bar_steps: int = 64
    bar_features: int = 16
    micro_steps: int = 240
    l1_features: int = 16
    l2_sides: int = 2
    l2_levels: int = 10
    l2_features: int = 12
    quality_features: int = 16
    horizons: int = 5
    output_features: int = 5

    # Reference model dimensions.
    d_model: int = 128
    d_level: int = 32
    num_heads: int = 4
    level_num_heads: int = 4
    ff_dim: int = 256
    level_ff_dim: int = 64
    bar_layers: int = 3
    l1_layers: int = 3
    l2_temporal_layers: int = 3
    l2_level_layers: int = 1
    decoder_layers: int = 2
    patch_size: int = 4

    dropout: float = 0.10
    time_hidden_dim: int = 32
    gate_hidden_dim: int = 64
    bar_time_scales: tuple[float, ...] = (
        60.0,
        120.0,
        240.0,
        480.0,
        960.0,
        1920.0,
        3840.0,
    )
    micro_time_scales: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        16.0,
        32.0,
        64.0,
    )
    bar_relative_time_scale: float = 60.0
    micro_relative_time_scale: float = 2.0
    gate_epsilon: float = 1.0e-6
    huber_delta: float = 1.0
    volume_loss_weight: float = 0.5
    delta_loss_weight: float = 0.5

    OUTPUT_ORDER: ClassVar[tuple[str, ...]] = (
        "poc_offset",
        "buy_poc_offset",
        "sell_poc_offset",
        "log_volume_residual",
        "delta_fraction",
    )
    _FIXED_CONTRACT: ClassVar[dict[str, int]] = {
        "bar_steps": 64,
        "bar_features": 16,
        "micro_steps": 240,
        "l1_features": 16,
        "l2_sides": 2,
        "l2_levels": 10,
        "l2_features": 12,
        "quality_features": 16,
        "horizons": 5,
        "output_features": 5,
        "patch_size": 4,
    }

    def __post_init__(self) -> None:
        for name, expected in self._FIXED_CONTRACT.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(
                    f"{name} is fixed by tensor contract at {expected}, got {actual}"
                )

        positive_ints = (
            "d_model",
            "d_level",
            "num_heads",
            "level_num_heads",
            "ff_dim",
            "level_ff_dim",
            "bar_layers",
            "l1_layers",
            "l2_temporal_layers",
            "l2_level_layers",
            "decoder_layers",
            "time_hidden_dim",
            "gate_hidden_dim",
        )
        for name in positive_ints:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.d_level % self.level_num_heads:
            raise ValueError("d_level must be divisible by level_num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.gate_epsilon <= 0.0:
            raise ValueError("gate_epsilon must be positive")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if self.volume_loss_weight < 0.0 or self.delta_loss_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        self._validate_scales("bar_time_scales", self.bar_time_scales)
        self._validate_scales("micro_time_scales", self.micro_time_scales)
        if self.bar_relative_time_scale <= 0.0:
            raise ValueError("bar_relative_time_scale must be positive")
        if self.micro_relative_time_scale <= 0.0:
            raise ValueError("micro_relative_time_scale must be positive")

    @staticmethod
    def _validate_scales(name: str, values: tuple[float, ...]) -> None:
        if not values:
            raise ValueError(f"{name} must not be empty")
        if any(value <= 0.0 for value in values):
            raise ValueError(f"{name} values must be positive")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError(f"{name} values must be strictly increasing")

    @classmethod
    def reference(cls, **overrides: Any) -> ModelConfig:
        """Return reference defaults; experimental overrides must recheck the parameter ceiling."""

        cls._reject_unknown_overrides(overrides)
        return cls(**overrides)

    @classmethod
    def tiny(cls, **overrides: Any) -> ModelConfig:
        """Return a deterministic small configuration for tests and smoke training."""

        cls._reject_unknown_overrides(overrides)
        values: dict[str, Any] = {
            "d_model": 32,
            "d_level": 16,
            "num_heads": 4,
            "level_num_heads": 4,
            "ff_dim": 64,
            "level_ff_dim": 32,
            "bar_layers": 1,
            "l1_layers": 1,
            "l2_temporal_layers": 1,
            "l2_level_layers": 1,
            "decoder_layers": 1,
            "dropout": 0.0,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def _reject_unknown_overrides(cls, overrides: dict[str, Any]) -> None:
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise TypeError(f"unknown ModelConfig option(s): {', '.join(unknown)}")

    @property
    def micro_patch_steps(self) -> int:
        return self.micro_steps // self.patch_size

    @property
    def memory_tokens(self) -> int:
        return self.bar_steps + 2 * self.micro_patch_steps


# A descriptive alias for callers that prefer the longer name.
NumericTransformerConfig = ModelConfig
