"""Bounded synthetic qualification controls for the CHMT-Q trainer only.

These controls exercise optimizer and evaluation wiring.  They never establish
market-data validity, out-of-sample performance, or eligibility for promotion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch

from .config import ChmtQConfig
from .contracts import ChmtQBatch
from .model import ChmtQTransformer
from .synthetic import make_chmt_q_synthetic_batch
from .training.config import TrainingConfig
from .training.optim import seed_everything
from .training.trainer import ChmtQTrainer, evaluate_model

NON_PROMOTABLE_REASON = (
    "synthetic trainer qualification only; never eligible for model promotion"
)

_CONTROL_VERSION = "chmt-q-qualification-v1"
_CONTROL_SEED = 421
_OVERFIT_INPUT_SEED = _CONTROL_SEED + 1
_OVERFIT_STEPS = 12
_OVERFIT_MINIMUM_REDUCTION = 0.30
_RANDOM_TRAIN_INPUT_SEED = _CONTROL_SEED + 11
_RANDOM_TRAIN_LABEL_SEED = _CONTROL_SEED + 101
_RANDOM_VALIDATION_INPUT_SEED = _CONTROL_SEED + 12
_RANDOM_VALIDATION_LABEL_SEED = _CONTROL_SEED + 202
_RANDOM_VALIDATION_BATCH_SIZE = 8
_RANDOM_STEPS = 16
_RANDOM_MAXIMUM_VALIDATION_IMPROVEMENT = 0.10


class QualificationControlError(RuntimeError):
    """Raised when a canonical synthetic trainer control does not pass."""


@dataclass(frozen=True, slots=True)
class TinyOverfitControlResult:
    """Audit-friendly result from the fixed-dataset overfit control."""

    control_version: str
    seed: int
    input_seed: int
    optimizer_steps: int
    initial_training_loss: float
    final_training_loss: float
    relative_loss_reduction: float
    minimum_required_reduction: float
    minimum_gradient_norm: float
    maximum_gradient_norm: float
    passed: bool = field(default=True, init=False)
    disposition: Literal["non-promotable"] = field(
        default="non-promotable", init=False
    )
    non_promotable_reason: str = field(default=NON_PROMOTABLE_REASON, init=False)


@dataclass(frozen=True, slots=True)
class RandomTargetControlResult:
    """Audit-friendly result from the held-out random-target control."""

    control_version: str
    seed: int
    training_input_seed: int
    training_label_seed: int
    validation_input_seed: int
    validation_label_seed: int
    optimizer_steps: int
    validation_batch_size: int
    initial_training_loss: float
    final_training_loss: float
    initial_validation_loss: float
    best_validation_loss: float
    final_validation_loss: float
    claimed_validation_improvement: float
    maximum_allowed_validation_improvement: float
    validation_loss_trace: tuple[float, ...]
    passed: bool = field(default=True, init=False)
    disposition: Literal["non-promotable"] = field(
        default="non-promotable", init=False
    )
    non_promotable_reason: str = field(default=NON_PROMOTABLE_REASON, init=False)


def run_tiny_overfit_control() -> TinyOverfitControlResult:
    """Require substantial loss reduction on one canonical synthetic batch.

    The seed is applied before model construction.  Training uses the exact
    effective-batch update API, and the before/after values come from the exact
    full-set evaluation API.
    """

    seed_everything(_CONTROL_SEED, deterministic=True)
    config = _qualification_model_config()
    model = ChmtQTransformer(config)
    batch = make_chmt_q_synthetic_batch(
        config,
        batch_size=1,
        seed=_OVERFIT_INPUT_SEED,
        device="cpu",
    )
    trainer = ChmtQTrainer(
        model,
        total_optimizer_steps=_OVERFIT_STEPS,
        config=_training_config(learning_rate=2.0e-3),
        device="cpu",
    )

    initial_loss = evaluate_model(model, (batch,), device="cpu").loss
    gradient_norms: list[float] = []
    for _ in range(_OVERFIT_STEPS):
        update = trainer.train_accumulation_window((batch,))
        gradient_norms.append(update.gradient_norm)
    final_loss = evaluate_model(model, (batch,), device="cpu").loss

    _require_finite_positive(initial_loss, "initial overfit loss")
    _require_finite_non_negative(final_loss, "final overfit loss")
    if not gradient_norms or not all(math.isfinite(value) for value in gradient_norms):
        raise QualificationControlError(
            "tiny overfit control produced a missing or non-finite gradient norm"
        )
    _require_finite_parameters(model, "tiny overfit control")
    reduction = (initial_loss - final_loss) / initial_loss
    if reduction < _OVERFIT_MINIMUM_REDUCTION:
        raise QualificationControlError(
            "tiny overfit control did not substantially reduce training loss: "
            f"observed={reduction:.6f}, required={_OVERFIT_MINIMUM_REDUCTION:.6f}"
        )

    return TinyOverfitControlResult(
        control_version=_CONTROL_VERSION,
        seed=_CONTROL_SEED,
        input_seed=_OVERFIT_INPUT_SEED,
        optimizer_steps=_OVERFIT_STEPS,
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        relative_loss_reduction=reduction,
        minimum_required_reduction=_OVERFIT_MINIMUM_REDUCTION,
        minimum_gradient_norm=min(gradient_norms),
        maximum_gradient_norm=max(gradient_norms),
    )


def run_random_target_negative_control() -> RandomTargetControlResult:
    """Reject an implausible validation gain on independent random labels.

    The fixed output head starts as a neutral marginal predictor.  This keeps
    routine calibration toward the label distribution from being misreported as
    learned random-label signal.  Validation is evaluated after every bounded
    update, so the best checkpoint claim is tested rather than only the last one.
    """

    seed_everything(_CONTROL_SEED, deterministic=True)
    config = _qualification_model_config()
    model = ChmtQTransformer(config)
    _initialize_neutral_output_head(model)
    training_batch = _make_random_target_batch(
        config,
        batch_size=1,
        input_seed=_RANDOM_TRAIN_INPUT_SEED,
        label_seed=_RANDOM_TRAIN_LABEL_SEED,
    )
    validation_batch = _make_random_target_batch(
        config,
        batch_size=_RANDOM_VALIDATION_BATCH_SIZE,
        input_seed=_RANDOM_VALIDATION_INPUT_SEED,
        label_seed=_RANDOM_VALIDATION_LABEL_SEED,
    )
    trainer = ChmtQTrainer(
        model,
        total_optimizer_steps=_RANDOM_STEPS,
        config=_training_config(learning_rate=3.0e-3),
        device="cpu",
    )

    initial_training_loss = evaluate_model(
        model, (training_batch,), device="cpu"
    ).loss
    initial_validation_loss = evaluate_model(
        model, (validation_batch,), device="cpu"
    ).loss
    validation_trace = [initial_validation_loss]
    gradient_norms: list[float] = []
    for _ in range(_RANDOM_STEPS):
        update = trainer.train_accumulation_window((training_batch,))
        gradient_norms.append(update.gradient_norm)
        validation_trace.append(
            evaluate_model(model, (validation_batch,), device="cpu").loss
        )
    final_training_loss = evaluate_model(
        model, (training_batch,), device="cpu"
    ).loss

    values = (initial_training_loss, final_training_loss, *validation_trace)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise QualificationControlError(
            "random-target control produced a non-finite or negative loss"
        )
    if initial_validation_loss <= 0.0:
        raise QualificationControlError(
            "random-target control requires a positive initial validation loss"
        )
    if not gradient_norms or not all(math.isfinite(value) for value in gradient_norms):
        raise QualificationControlError(
            "random-target control produced a missing or non-finite gradient norm"
        )
    _require_finite_parameters(model, "random-target control")

    best_validation_loss = min(validation_trace)
    claimed_improvement = max(
        0.0,
        (initial_validation_loss - best_validation_loss) / initial_validation_loss,
    )
    if claimed_improvement > _RANDOM_MAXIMUM_VALIDATION_IMPROVEMENT:
        raise QualificationControlError(
            "random-target control claimed implausible held-out improvement: "
            f"observed={claimed_improvement:.6f}, "
            f"allowed={_RANDOM_MAXIMUM_VALIDATION_IMPROVEMENT:.6f}"
        )

    return RandomTargetControlResult(
        control_version=_CONTROL_VERSION,
        seed=_CONTROL_SEED,
        training_input_seed=_RANDOM_TRAIN_INPUT_SEED,
        training_label_seed=_RANDOM_TRAIN_LABEL_SEED,
        validation_input_seed=_RANDOM_VALIDATION_INPUT_SEED,
        validation_label_seed=_RANDOM_VALIDATION_LABEL_SEED,
        optimizer_steps=_RANDOM_STEPS,
        validation_batch_size=_RANDOM_VALIDATION_BATCH_SIZE,
        initial_training_loss=initial_training_loss,
        final_training_loss=final_training_loss,
        initial_validation_loss=initial_validation_loss,
        best_validation_loss=best_validation_loss,
        final_validation_loss=validation_trace[-1],
        claimed_validation_improvement=claimed_improvement,
        maximum_allowed_validation_improvement=(
            _RANDOM_MAXIMUM_VALIDATION_IMPROVEMENT
        ),
        validation_loss_trace=tuple(validation_trace),
    )


def _qualification_model_config() -> ChmtQConfig:
    return ChmtQConfig.tiny(
        d_model=8,
        d_level=4,
        num_heads=1,
        level_num_heads=1,
        ff_dim=16,
        level_ff_dim=8,
        time_hidden_dim=4,
        gate_hidden_dim=4,
    )


def _training_config(*, learning_rate: float) -> TrainingConfig:
    return TrainingConfig(
        seed=_CONTROL_SEED,
        learning_rate=learning_rate,
        minimum_learning_rate=learning_rate,
        weight_decay=0.0,
        warmup_fraction=0.0,
        gradient_clip_norm=5.0,
        micro_batch_size=1,
        accumulation_steps=1,
        mixed_precision=False,
        deterministic=True,
    )


def _initialize_neutral_output_head(model: ChmtQTransformer) -> None:
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.zero_()


def _make_random_target_batch(
    config: ChmtQConfig,
    *,
    batch_size: int,
    input_seed: int,
    label_seed: int,
) -> ChmtQBatch:
    inputs = make_chmt_q_synthetic_batch(
        config,
        batch_size=batch_size,
        seed=input_seed,
        device="cpu",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(label_seed)
    targets = torch.empty(
        (batch_size, config.horizons, config.targets), dtype=torch.float32
    )
    targets[..., : config.delta_target_index] = 0.75 * torch.randn(
        (batch_size, config.horizons, config.delta_target_index),
        generator=generator,
        dtype=torch.float32,
    )
    targets[..., config.delta_target_index] = (
        2.0
        * torch.rand(
            (batch_size, config.horizons),
            generator=generator,
            dtype=torch.float32,
        )
        - 1.0
    )
    presence_labels = (
        torch.rand(
            (batch_size, config.horizons, config.presence_targets),
            generator=generator,
            dtype=torch.float32,
        )
        >= 0.5
    ).to(torch.float32)
    return ChmtQBatch(
        bars=inputs.bars,
        bars_mask=inputs.bars_mask,
        l1=inputs.l1,
        l1_mask=inputs.l1_mask,
        l2=inputs.l2,
        l2_level_mask=inputs.l2_level_mask,
        l2_snapshot_mask=inputs.l2_snapshot_mask,
        quality=inputs.quality,
        targets=targets,
        target_mask=torch.ones_like(targets, dtype=torch.bool),
        presence_labels=presence_labels,
        presence_mask=torch.ones_like(presence_labels, dtype=torch.bool),
    ).validate(config, require_supervision=True)


def _require_finite_positive(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise QualificationControlError(f"{name} must be finite and positive")


def _require_finite_non_negative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise QualificationControlError(f"{name} must be finite and non-negative")


def _require_finite_parameters(model: ChmtQTransformer, control: str) -> None:
    if any(
        not bool(torch.isfinite(parameter).all().item())
        for parameter in model.parameters()
    ):
        raise QualificationControlError(f"{control} produced a non-finite parameter")

