"""AdamW grouping, deterministic seeding, and the reference LR schedule."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Iterable

import numpy as np
import torch
from torch import nn

from .config import TrainingConfig

_NO_DECAY_NAME_PARTS = (
    "bias",
    "embedding",
    "horizon_queries",
    "layer_scale",
    "log_temperature",
    "relative_time_bias.gamma",
    "norm",
    "pool.query",
)
_DETERMINISTIC_CUBLAS_CONFIGS = frozenset({":16:8", ":4096:8"})


def validate_training_runtime(
    config: TrainingConfig, device: torch.device | str
) -> torch.device:
    """Fail before allocation when the runtime cannot honor the frozen config."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    resolved = torch.device(device)
    if resolved.type != "cuda":
        if config.mixed_precision:
            raise ValueError("mixed_precision currently requires a CUDA device")
        return resolved
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if resolved.index is not None and resolved.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device index is unavailable")
    if config.deterministic and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (
        _DETERMINISTIC_CUBLAS_CONFIGS
    ):
        raise ValueError(
            "deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8"
        )
    if config.mixed_precision and not _cuda_bf16_supported(resolved):
        raise ValueError(
            "CUDA native bf16 mixed precision was requested but is unsupported"
        )
    return resolved


def _cuda_bf16_supported(device: torch.device) -> bool:
    """Query native bf16 support on the selected GPU, not emulation."""

    if device.index is None:
        return _native_cuda_bf16_supported()
    with torch.cuda.device(device):
        return _native_cuda_bf16_supported()


def _native_cuda_bf16_supported() -> bool:
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:
        major, _minor = torch.cuda.get_device_capability()
        return major >= 8 and bool(torch.cuda.is_bf16_supported())


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, CPU, and CUDA without hiding unsupported kernels."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=False)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic


def build_parameter_groups(
    model: nn.Module, weight_decay: float
) -> list[dict[str, object]]:
    """Apply decay only to matrix weights that are not embeddings or learned scales."""

    if not math.isfinite(weight_decay) or weight_decay < 0.0:
        raise ValueError("weight_decay must be finite and non-negative")
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if identity in seen:
            raise ValueError(f"parameter {name!r} appears more than once")
        seen.add(identity)
        excluded = parameter.ndim < 2 or any(part in name for part in _NO_DECAY_NAME_PARTS)
        (no_decay if excluded else decay).append(parameter)
    if not decay and not no_decay:
        raise ValueError("model has no trainable parameters")
    return [
        {"params": decay, "weight_decay": weight_decay, "group_name": "matrix_decay"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "no_decay"},
    ]


def build_adamw(model: nn.Module, config: TrainingConfig) -> torch.optim.AdamW:
    """Construct the frozen reference AdamW optimizer."""

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    return torch.optim.AdamW(
        build_parameter_groups(model, config.weight_decay),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.adam_epsilon,
    )


def learning_rate_at_step(
    step: int, total_steps: int, config: TrainingConfig
) -> float:
    """Linear warm-up followed by cosine decay, indexed by completed optimizer step."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    bounded_step = min(step, total_steps)
    warmup_steps = int(math.ceil(total_steps * config.warmup_fraction))
    if warmup_steps and bounded_step < warmup_steps:
        return config.learning_rate * (bounded_step + 1) / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max((bounded_step - warmup_steps) / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.minimum_learning_rate + (
        config.learning_rate - config.minimum_learning_rate
    ) * cosine


def apply_learning_rate(
    optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    """Set one finite positive rate on every optimizer group."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def trainable_parameters(groups: Iterable[dict[str, object]]) -> list[nn.Parameter]:
    """Flatten optimizer groups for focused gradient checks."""

    parameters: list[nn.Parameter] = []
    for group in groups:
        values = group.get("params")
        if not isinstance(values, list):
            raise TypeError("optimizer group params must be a list")
        parameters.extend(values)
    return parameters
