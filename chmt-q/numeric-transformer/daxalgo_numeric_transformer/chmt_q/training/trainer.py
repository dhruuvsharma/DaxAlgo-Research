"""Finite, deterministic reference training loop for CHMT-Q review and development."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ..config import ChmtQConfig
from ..contracts import ChmtQBatch
from ..loss import (
    ChmtQLossStatistics,
    ChmtQLossStatisticsAccumulator,
    ChmtQLossTerms,
    aggregate_chmt_q_loss_statistics,
    chmt_q_loss,
    chmt_q_objective_from_statistics,
)
from ..metrics import ChmtQMetricReport, ChmtQMetricsAccumulator
from .config import TrainingConfig
from .optim import (
    apply_learning_rate,
    build_adamw,
    learning_rate_at_step,
    seed_everything,
    validate_training_runtime,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from ..model import ChmtQTransformer
    from ..scaling import RobustScalerArtifact


_TRAINER_STATE_VERSION = 1


@dataclass(frozen=True, slots=True)
class TrainingStepResult:
    """One micro-batch result with explicit optimizer-step state."""

    micro_step: int
    optimizer_step: int
    optimizer_updated: bool
    learning_rate: float
    loss: float
    pinball: float
    median_huber: float
    presence: float
    auxiliary_scale: float
    gradient_norm: float | None


@dataclass(frozen=True, slots=True)
class TrainingWindowResult:
    """One exact effective-batch update and its constituent micro-batch results."""

    micro_batches: tuple[TrainingStepResult, ...]
    optimizer_step: int
    learning_rate: float
    loss: float
    pinball: float
    median_huber: float
    presence: float
    auxiliary_scale: float
    gradient_norm: float
    update_norm: float


@dataclass(frozen=True, slots=True)
class ChmtQEvaluationResult:
    """Full-set additive objective and detailed probabilistic metrics."""

    loss: float
    pinball: float
    median_huber: float
    presence: float
    statistics: ChmtQLossStatistics
    metrics: ChmtQMetricReport
    physical_metrics: ChmtQMetricReport | None = None


@dataclass(frozen=True, slots=True)
class _WindowDenominators:
    pinball: Tensor
    median_huber: Tensor
    presence: Tensor


class ChmtQTrainer:
    """Reference AdamW trainer with accumulation, finite checks, and gradient clipping."""

    def __init__(
        self,
        model: ChmtQTransformer,
        *,
        total_optimizer_steps: int,
        config: TrainingConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        from ..model import ChmtQTransformer

        if not isinstance(model, ChmtQTransformer):
            raise TypeError("model must be a ChmtQTransformer")
        if (
            isinstance(total_optimizer_steps, bool)
            or not isinstance(total_optimizer_steps, int)
            or total_optimizer_steps <= 0
        ):
            raise ValueError("total_optimizer_steps must be a positive integer")
        self.training_config = config or TrainingConfig()
        self.device = validate_training_runtime(self.training_config, device)
        seed_everything(
            self.training_config.seed,
            deterministic=self.training_config.deterministic,
        )
        self.model = model.to(device=self.device, dtype=torch.float32)
        self.optimizer = build_adamw(self.model, self.training_config)
        self.total_optimizer_steps = total_optimizer_steps
        self.micro_step = 0
        self.optimizer_step = 0
        self._accumulated = 0
        self.optimizer.zero_grad(set_to_none=True)
        self._scaler = torch.amp.GradScaler(
            "cuda", enabled=self.training_config.mixed_precision
        )

    def state_dict(self) -> dict[str, object]:
        """Return JSON-safe boundary state; sampler position remains caller-owned."""

        if self._accumulated != 0:
            raise RuntimeError("trainer state may only be captured at an accumulation boundary")
        return {
            "schema_version": _TRAINER_STATE_VERSION,
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "total_optimizer_steps": self.total_optimizer_steps,
            "accumulated_micro_batches": self._accumulated,
            "training_config": asdict(self.training_config),
            "grad_scaler": self._scaler.state_dict(),
            "sampler_position": "external",
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        """Restore validated boundary state; callers restore their sampler separately."""

        if not isinstance(state, dict):
            raise TypeError("trainer state must be a dictionary")
        expected_keys = {
            "schema_version",
            "micro_step",
            "optimizer_step",
            "total_optimizer_steps",
            "accumulated_micro_batches",
            "training_config",
            "grad_scaler",
            "sampler_position",
        }
        if set(state) != expected_keys:
            raise ValueError("trainer state field set does not match the supported schema")
        if state["schema_version"] != _TRAINER_STATE_VERSION:
            raise ValueError("unsupported trainer state schema version")
        if self._accumulated != 0:
            raise RuntimeError("trainer state may only be restored at an accumulation boundary")

        micro_step = _state_integer(state["micro_step"], "micro_step")
        optimizer_step = _state_integer(state["optimizer_step"], "optimizer_step")
        total_steps = _state_integer(
            state["total_optimizer_steps"], "total_optimizer_steps", positive=True
        )
        accumulated = _state_integer(
            state["accumulated_micro_batches"], "accumulated_micro_batches"
        )
        if accumulated != 0:
            raise ValueError("only accumulation-boundary trainer state can be restored")
        if total_steps != self.total_optimizer_steps:
            raise ValueError("trainer state total_optimizer_steps does not match this run")
        if optimizer_step > total_steps:
            raise ValueError("trainer state optimizer_step exceeds the configured budget")
        expected_micro_step = optimizer_step * self.training_config.accumulation_steps
        if micro_step != expected_micro_step:
            raise ValueError(
                "trainer state micro_step is inconsistent with accumulation boundaries"
            )
        if state["training_config"] != asdict(self.training_config):
            raise ValueError("trainer state training configuration does not match this run")
        if state["sampler_position"] != "external":
            raise ValueError("trainer state must leave sampler position externally managed")

        scaler_state = state["grad_scaler"]
        if not isinstance(scaler_state, dict):
            raise TypeError("trainer grad_scaler state must be a dictionary")
        if self._scaler.is_enabled() != bool(scaler_state):
            raise ValueError("trainer GradScaler state does not match mixed-precision mode")
        try:
            self._scaler.load_state_dict(scaler_state)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("trainer GradScaler state is invalid") from error

        self.micro_step = micro_step
        self.optimizer_step = optimizer_step
        self._accumulated = 0
        self.optimizer.zero_grad(set_to_none=True)

    def train_micro_batch(self, batch: ChmtQBatch) -> TrainingStepResult:
        """Train one batch when accumulation is disabled."""

        if self.training_config.accumulation_steps != 1:
            raise RuntimeError(
                "train_micro_batch requires accumulation_steps=1; "
                "use train_accumulation_window for exact masked accumulation"
            )
        return self.train_accumulation_window((batch,)).micro_batches[0]

    def train_accumulation_window(
        self, batches: Sequence[ChmtQBatch]
    ) -> TrainingWindowResult:
        """Update from one complete window using global component denominators."""

        if self.optimizer_step >= self.total_optimizer_steps:
            raise RuntimeError("the configured optimizer-step budget is already complete")
        if len(batches) != self.training_config.accumulation_steps:
            raise ValueError(
                "accumulation window size does not match TrainingConfig.accumulation_steps"
            )
        moved_batches: list[ChmtQBatch] = []
        for batch in batches:
            if batch.batch_size != self.training_config.micro_batch_size:
                raise ValueError(
                    "batch size does not match TrainingConfig.micro_batch_size: "
                    f"{batch.batch_size} != {self.training_config.micro_batch_size}"
                )
            moved = _move_batch(batch, self.device)
            moved.validate(self.model.config, require_supervision=True)
            moved_batches.append(moved)

        denominators = _window_denominators(moved_batches, self.model.config)
        self.model.train()
        terms_by_batch: list[ChmtQLossTerms] = []
        self._accumulated = 0
        try:
            for moved in moved_batches:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.training_config.mixed_precision,
                ):
                    prediction = self.model.forward_training(moved)
                terms = chmt_q_loss(
                    prediction.final,
                    targets=moved.targets,
                    target_mask=moved.target_mask,
                    presence_labels=moved.presence_labels,
                    presence_mask=moved.presence_mask,
                    auxiliary_output=prediction.auxiliary,
                    step=self.optimizer_step,
                    total_steps=self.total_optimizer_steps,
                    config=self.model.config,
                )
                contribution = _window_loss_contribution(
                    terms,
                    denominators,
                    self.model.config,
                )
                _require_finite_scalar(contribution, "training loss contribution")
                self._scaler.scale(contribution).backward()
                terms_by_batch.append(terms)
                self._accumulated += 1

            self._scaler.unscale_(self.optimizer)
            _require_finite_gradients(self.model)
            norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.training_config.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            gradient_norm = float(norm.detach().cpu())
            learning_rate = learning_rate_at_step(
                self.optimizer_step,
                self.total_optimizer_steps,
                self.training_config,
            )
            apply_learning_rate(self.optimizer, learning_rate)
            before_update = tuple(
                parameter.detach().clone()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            self._scaler.step(self.optimizer)
            self._scaler.update()
            squared_update = torch.zeros((), dtype=torch.float64, device=self.device)
            for parameter, previous in zip(
                (item for item in self.model.parameters() if item.requires_grad),
                before_update,
                strict=True,
            ):
                difference = parameter.detach().to(torch.float64) - previous.to(
                    torch.float64
                )
                squared_update = squared_update + difference.square().sum()
            update_norm = float(torch.sqrt(squared_update).cpu())
            if not math.isfinite(update_norm):
                raise FloatingPointError("optimizer update norm became non-finite")
            self.optimizer.zero_grad(set_to_none=True)
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            self._accumulated = 0
            raise

        previous_micro_step = self.micro_step
        previous_optimizer_step = self.optimizer_step
        self.micro_step += len(moved_batches)
        self.optimizer_step += 1
        self._accumulated = 0

        statistics = aggregate_chmt_q_loss_statistics(
            [terms.statistics for terms in terms_by_batch]
        )
        auxiliary_statistics = aggregate_chmt_q_loss_statistics(
            [
                terms.auxiliary_statistics
                for terms in terms_by_batch
                if terms.auxiliary_statistics is not None
            ]
        )
        auxiliary_scale = terms_by_batch[0].auxiliary_scale
        final_objective = chmt_q_objective_from_statistics(
            statistics, self.model.config
        )
        auxiliary_objective = chmt_q_objective_from_statistics(
            auxiliary_statistics, self.model.config
        )
        total = final_objective + auxiliary_scale * auxiliary_objective

        micro_results: list[TrainingStepResult] = []
        for index, terms in enumerate(terms_by_batch, start=1):
            is_last = index == len(terms_by_batch)
            micro_results.append(
                _result(
                    terms,
                    micro_step=previous_micro_step + index,
                    optimizer_step=(
                        self.optimizer_step if is_last else previous_optimizer_step
                    ),
                    optimizer_updated=is_last,
                    learning_rate=learning_rate,
                    gradient_norm=gradient_norm if is_last else None,
                )
            )

        return TrainingWindowResult(
            micro_batches=tuple(micro_results),
            optimizer_step=self.optimizer_step,
            learning_rate=learning_rate,
            loss=float(total.detach().cpu()),
            pinball=float(statistics.pinball.detach().cpu()),
            median_huber=float(statistics.median_huber.detach().cpu()),
            presence=float(statistics.presence.detach().cpu()),
            auxiliary_scale=auxiliary_scale,
            gradient_norm=gradient_norm,
            update_norm=update_norm,
        )

    def fit(self, batches: Iterable[ChmtQBatch]) -> list[TrainingStepResult]:
        """Cycle a re-iterable loader until the frozen optimizer-step budget is met."""

        results: list[TrainingStepResult] = []
        window: list[ChmtQBatch] = []
        while self.optimizer_step < self.total_optimizer_steps:
            progressed = False
            for batch in batches:
                progressed = True
                window.append(batch)
                if len(window) == self.training_config.accumulation_steps:
                    update = self.train_accumulation_window(tuple(window))
                    results.extend(update.micro_batches)
                    window.clear()
                    if self.optimizer_step >= self.total_optimizer_steps:
                        break
            if not progressed:
                raise ValueError("training iterable produced no batches")
        if window:
            raise RuntimeError("training ended with an incomplete accumulation window")
        return results


def evaluate_loss(
    model: ChmtQTransformer,
    batches: Iterable[ChmtQBatch],
    *,
    device: torch.device | str = "cpu",
) -> float:
    """Return the full-set objective reconstructed from additive component sums."""

    return evaluate_model(model, batches, device=device).loss


def evaluate_model(
    model: ChmtQTransformer,
    batches: Iterable[ChmtQBatch],
    *,
    device: torch.device | str = "cpu",
    metric_scaler: RobustScalerArtifact | None = None,
) -> ChmtQEvaluationResult:
    """Evaluate the normalized objective and optional physical-unit metrics."""

    if metric_scaler is not None:
        from ..scaling import RobustScalerArtifact

        if not isinstance(metric_scaler, RobustScalerArtifact):
            raise TypeError("metric_scaler must be a RobustScalerArtifact")

    resolved = torch.device(device)
    original_device = _model_device(model)
    original_training = model.training
    statistics = ChmtQLossStatisticsAccumulator()
    metric_accumulator = ChmtQMetricsAccumulator(model.config)
    physical_accumulator = (
        ChmtQMetricsAccumulator(model.config) if metric_scaler is not None else None
    )
    try:
        model.to(device=resolved, dtype=torch.float32).eval()
        with torch.no_grad():
            for batch in batches:
                moved = _move_batch(batch, resolved)
                moved.validate(model.config, require_supervision=True)
                prediction = model.forward_distribution(moved)
                terms = chmt_q_loss(
                    prediction,
                    targets=moved.targets,
                    target_mask=moved.target_mask,
                    presence_labels=moved.presence_labels,
                    presence_mask=moved.presence_mask,
                    config=model.config,
                )
                statistics.update(terms.statistics)
                metric_accumulator.update(
                    prediction,
                    targets=moved.targets,
                    target_mask=moved.target_mask,
                    presence_labels=moved.presence_labels,
                    presence_mask=moved.presence_mask,
                )
                if physical_accumulator is not None and metric_scaler is not None:
                    assert moved.targets is not None
                    physical_accumulator.update(
                        metric_scaler.denormalize_output(prediction, model.config),
                        targets=metric_scaler.denormalize_targets(moved.targets),
                        target_mask=moved.target_mask,
                        presence_labels=moved.presence_labels,
                        presence_mask=moved.presence_mask,
                    )
                del moved, prediction, terms
    finally:
        model.to(device=original_device, dtype=torch.float32)
        model.train(original_training)
    if statistics.updates == 0:
        raise ValueError("evaluation iterable produced no batches")
    aggregate = statistics.finalize()
    objective = chmt_q_objective_from_statistics(aggregate, model.config)
    return ChmtQEvaluationResult(
        loss=float(objective.cpu()),
        pinball=float(aggregate.pinball.cpu()),
        median_huber=float(aggregate.median_huber.cpu()),
        presence=float(aggregate.presence.cpu()),
        statistics=aggregate,
        metrics=metric_accumulator.finalize(),
        physical_metrics=(
            physical_accumulator.finalize()
            if physical_accumulator is not None
            else None
        ),
    )


def _window_denominators(
    batches: list[ChmtQBatch], config: ChmtQConfig
) -> _WindowDenominators:
    if not batches:
        raise ValueError("accumulation window must not be empty")
    device = batches[0].device
    horizon_weights = torch.tensor(
        config.horizon_weights,
        dtype=torch.float32,
        device=device,
    ).view(-1, 1)
    target_weights = torch.tensor(
        config.target_weights,
        dtype=torch.float32,
        device=device,
    ).view(1, -1)
    weighted_forecast_count = torch.zeros((), dtype=torch.float32, device=device)
    presence_count = torch.zeros((), dtype=torch.float32, device=device)
    for batch in batches:
        if batch.target_mask is not None:
            counts = batch.target_mask.sum(dim=0, dtype=torch.float32)
            weighted_forecast_count = weighted_forecast_count + (
                counts * horizon_weights * target_weights
            ).sum()
        if batch.presence_mask is not None:
            presence_count = presence_count + batch.presence_mask.sum(
                dtype=torch.float32
            )
    return _WindowDenominators(
        pinball=weighted_forecast_count * config.quantiles,
        median_huber=weighted_forecast_count,
        presence=presence_count,
    )


def _window_loss_contribution(
    terms: ChmtQLossTerms,
    denominators: _WindowDenominators,
    config: ChmtQConfig,
) -> Tensor:
    statistics = terms.statistics
    final = (
        _divide_for_window(statistics.pinball_numerator, denominators.pinball)
        + config.median_huber_weight
        * _divide_for_window(
            statistics.median_huber_numerator, denominators.median_huber
        )
        + config.presence_weight
        * _divide_for_window(statistics.presence_numerator, denominators.presence)
    )
    if terms.auxiliary_statistics is None:
        return final
    auxiliary = terms.auxiliary_statistics
    auxiliary_objective = (
        _divide_for_window(auxiliary.pinball_numerator, denominators.pinball)
        + config.median_huber_weight
        * _divide_for_window(
            auxiliary.median_huber_numerator, denominators.median_huber
        )
        + config.presence_weight
        * _divide_for_window(auxiliary.presence_numerator, denominators.presence)
    )
    return final + terms.auxiliary_scale * auxiliary_objective


def _divide_for_window(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator / denominator.clamp_min(1.0)


def _model_device(model: torch.nn.Module) -> torch.device:
    devices = {
        value.device
        for value in (*tuple(model.parameters()), *tuple(model.buffers()))
    }
    if len(devices) != 1:
        raise ValueError("model parameters and buffers must use one device")
    return next(iter(devices))


def _move_batch(batch: ChmtQBatch, device: torch.device) -> ChmtQBatch:
    def move(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.to(device=device)

    return ChmtQBatch(
        bars=move(batch.bars),  # type: ignore[arg-type]
        bars_mask=move(batch.bars_mask),  # type: ignore[arg-type]
        l1=move(batch.l1),  # type: ignore[arg-type]
        l1_mask=move(batch.l1_mask),  # type: ignore[arg-type]
        l2=move(batch.l2),  # type: ignore[arg-type]
        l2_level_mask=move(batch.l2_level_mask),  # type: ignore[arg-type]
        l2_snapshot_mask=move(batch.l2_snapshot_mask),  # type: ignore[arg-type]
        quality=move(batch.quality),  # type: ignore[arg-type]
        targets=move(batch.targets),
        target_mask=move(batch.target_mask),
        presence_labels=move(batch.presence_labels),
        presence_mask=move(batch.presence_mask),
    )


def _require_finite_scalar(value: Tensor, name: str) -> None:
    if value.ndim != 0 or not bool(torch.isfinite(value).item()):
        raise FloatingPointError(f"{name} must be one finite scalar")


def _state_integer(value: object, name: str, *, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"trainer state {name} must be a {qualifier} integer")
    return value


def _require_finite_gradients(model: torch.nn.Module) -> None:
    found = False
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise FloatingPointError(f"non-finite gradient in {name}")
    if not found:
        raise RuntimeError("training step produced no gradients")


def _result(
    terms: ChmtQLossTerms,
    *,
    micro_step: int,
    optimizer_step: int,
    optimizer_updated: bool,
    learning_rate: float,
    gradient_norm: float | None,
) -> TrainingStepResult:
    return TrainingStepResult(
        micro_step=micro_step,
        optimizer_step=optimizer_step,
        optimizer_updated=optimizer_updated,
        learning_rate=learning_rate,
        loss=float(terms.total.detach().cpu()),
        pinball=float(terms.pinball.detach().cpu()),
        median_huber=float(terms.median_huber.detach().cpu()),
        presence=float(terms.presence.detach().cpu()),
        auxiliary_scale=terms.auxiliary_scale,
        gradient_norm=gradient_norm,
    )
