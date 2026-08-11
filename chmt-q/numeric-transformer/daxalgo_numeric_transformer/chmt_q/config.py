"""Immutable configuration for the selected CHMT-Q v1 contract."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class ChmtQConfig:
    """Fixed tensor contract and validated objective settings for CHMT-Q v1."""

    bar_steps: int = 64
    bar_features: int = 16
    micro_steps: int = 240
    l1_features: int = 16
    l2_sides: int = 2
    l2_levels: int = 10
    l2_features: int = 12
    quality_features: int = 16
    horizons: int = 5
    targets: int = 5
    quantiles: int = 3
    presence_targets: int = 2
    raw_output_features: int = 17
    # The corrected literal ledger includes 36 learned Q/K temperatures and
    # 36 independent learned relative-decay gammas. The earlier 2,089,518
    # figure accidentally omitted the temperatures; neither family is tied.
    reference_parameter_count: int = 2_089_554

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

    bar_slot_seconds: float = 60.0
    micro_frame_seconds: float = 0.5
    forecast_interval_seconds: float = 60.0
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
    rms_epsilon: float = 1.0e-6
    qk_epsilon: float = 1.0e-6
    gate_epsilon: float = 1.0e-6
    layer_scale_initial: float = 0.1

    quantile_levels: tuple[float, ...] = (0.1, 0.5, 0.9)
    horizon_weights: tuple[float, ...] = (
        1.0,
        1.0 / math.sqrt(2.0),
        1.0 / math.sqrt(3.0),
        0.5,
        1.0 / math.sqrt(5.0),
    )
    target_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 0.5, 0.5)
    huber_delta: float = 1.0
    median_huber_weight: float = 0.25
    presence_weight: float = 0.20
    auxiliary_initial_weight: float = 0.25
    auxiliary_final_weight: float = 0.05
    auxiliary_decay_start: float = 0.60
    auxiliary_decay_end: float = 0.90

    TARGET_ORDER: ClassVar[tuple[str, ...]] = (
        "poc_offset",
        "buy_poc_offset",
        "sell_poc_offset",
        "log_volume_residual",
        "delta_fraction",
    )
    PRESENCE_ORDER: ClassVar[tuple[str, ...]] = ("buy_poc", "sell_poc")
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
        "targets": 5,
        "quantiles": 3,
        "presence_targets": 2,
        "raw_output_features": 17,
        "reference_parameter_count": 2_089_554,
        "patch_size": 4,
    }
    _FIXED_PHYSICAL: ClassVar[dict[str, float | tuple[float, ...]]] = {
        "bar_slot_seconds": 60.0,
        "micro_frame_seconds": 0.5,
        "forecast_interval_seconds": 60.0,
        "bar_time_scales": (60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 3840.0),
        "micro_time_scales": (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        "bar_relative_time_scale": 60.0,
        "micro_relative_time_scale": 2.0,
    }
    _TINY_OVERRIDES: ClassVar[dict[str, int | float]] = {
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
        "time_hidden_dim": 16,
        "gate_hidden_dim": 16,
    }

    def __post_init__(self) -> None:
        for name, expected in self._FIXED_CONTRACT.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"{name} is fixed at {expected}, got {actual}")
        for name, expected in self._FIXED_PHYSICAL.items():
            actual = getattr(self, name)
            if actual != expected:
                raise ValueError(f"{name} is fixed at {expected}, got {actual}")

        if self.raw_output_features != self.targets * self.quantiles + self.presence_targets:
            raise ValueError("raw output width must equal targets * quantiles + presence targets")
        if self.quantile_levels != (0.1, 0.5, 0.9):
            raise ValueError("CHMT-Q v1 quantile levels are fixed at (0.1, 0.5, 0.9)")
        self._validate_weights("horizon_weights", self.horizon_weights, self.horizons)
        self._validate_weights("target_weights", self.target_weights, self.targets)

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
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.d_level % self.level_num_heads:
            raise ValueError("d_level must be divisible by level_num_heads")
        if self.micro_steps % self.patch_size:
            raise ValueError("micro_steps must be divisible by patch_size")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be finite and in [0, 1)")

        epsilons = {
            "rms_epsilon": self.rms_epsilon,
            "qk_epsilon": self.qk_epsilon,
            "gate_epsilon": self.gate_epsilon,
        }
        for name, value in epsilons.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.layer_scale_initial) or self.layer_scale_initial < 0.0:
            raise ValueError("layer_scale_initial must be finite and non-negative")

        positive = {
            "huber_delta": self.huber_delta,
            "median_huber_weight": self.median_huber_weight,
            "presence_weight": self.presence_weight,
            "auxiliary_initial_weight": self.auxiliary_initial_weight,
            "auxiliary_final_weight": self.auxiliary_final_weight,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if self.auxiliary_initial_weight < self.auxiliary_final_weight:
            raise ValueError("auxiliary weight must not increase during training")
        if not (
            0.0 <= self.auxiliary_decay_start < self.auxiliary_decay_end <= 1.0
        ):
            raise ValueError("auxiliary decay boundaries must satisfy 0 <= start < end <= 1")

    @staticmethod
    def _validate_weights(name: str, values: tuple[float, ...], expected: int) -> None:
        if len(values) != expected:
            raise ValueError(f"{name} must contain exactly {expected} entries")
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"{name} entries must be finite and positive")

    @property
    def delta_target_index(self) -> int:
        """Return the fixed target index whose ordered quantiles are bounded by tanh."""

        return 4

    @property
    def micro_patch_steps(self) -> int:
        return self.micro_steps // self.patch_size

    @property
    def memory_tokens(self) -> int:
        return self.bar_steps + 2 * self.micro_patch_steps

    @property
    def bar_ages_seconds(self) -> tuple[float, ...]:
        return tuple(
            index * self.bar_slot_seconds for index in range(self.bar_steps - 1, -1, -1)
        )

    @property
    def patch_ages_seconds(self) -> tuple[float, ...]:
        return tuple(
            index * self.patch_size * self.micro_frame_seconds
            for index in range(self.micro_patch_steps - 1, -1, -1)
        )

    @classmethod
    def reference(cls) -> ChmtQConfig:
        """Return the literal selected CHMT-Q v1 architecture."""

        return cls()

    @classmethod
    def tiny(cls, **overrides: Any) -> ChmtQConfig:
        """Return a small test architecture without permitting contract overrides."""

        known = {field.name for field in fields(cls)}
        allowed = set(cls._TINY_OVERRIDES)
        unknown = sorted(set(overrides) - known)
        forbidden = sorted(set(overrides) - allowed)
        if unknown:
            raise TypeError(f"unknown ChmtQConfig option(s): {', '.join(unknown)}")
        if forbidden:
            raise TypeError(
                "tiny() only permits architectural test overrides; forbidden: "
                + ", ".join(forbidden)
            )
        values = dict(cls._TINY_OVERRIDES)
        values.update(overrides)
        return cls(**values)
