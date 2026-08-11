"""Strict, immutable configuration for a CHMT-Q training run."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Reference optimizer and step-loop settings frozen before a run begins.

    Epoch limits, early stopping, fold selection, and sampler state belong to the
    still-deferred experiment orchestrator and are intentionally not inert fields here.
    """

    seed: int = 17
    learning_rate: float = 3.0e-4
    minimum_learning_rate: float = 3.0e-5
    beta1: float = 0.9
    beta2: float = 0.95
    adam_epsilon: float = 1.0e-8
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    micro_batch_size: int = 8
    accumulation_steps: int = 16
    mixed_precision: bool = False
    deterministic: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "micro_batch_size",
            "accumulation_steps",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        positive = (
            "learning_rate",
            "minimum_learning_rate",
            "adam_epsilon",
            "gradient_clip_norm",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_learning_rate > self.learning_rate:
            raise ValueError("minimum_learning_rate must not exceed learning_rate")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError("warmup_fraction must be in [0, 1)")
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not isinstance(self.mixed_precision, bool) or not isinstance(
            self.deterministic, bool
        ):
            raise TypeError("mixed_precision and deterministic must be bool")

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.accumulation_steps

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> TrainingConfig:
        """Build a strict config, rejecting misspelled or unknown fields."""

        if not isinstance(values, dict):
            raise TypeError("training configuration must be a mapping")
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError("unknown training option(s): " + ", ".join(unknown))
        return cls(**values)
