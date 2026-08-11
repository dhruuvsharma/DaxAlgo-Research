"""Masked CHMT-Q forecast, presence, and auxiliary objectives."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ChmtQConfig
from .output import ChmtQOutput


class NoSupervisedTermsError(RuntimeError):
    """Raised when a batch supplies no forecast or presence supervision."""


@dataclass(frozen=True, slots=True)
class ChmtQLossStatistics:
    """Additive FP32 objective sums plus explicit per-cell supervision counts."""

    pinball_numerator: Tensor
    pinball_denominator: Tensor
    median_huber_numerator: Tensor
    median_huber_denominator: Tensor
    presence_numerator: Tensor
    presence_denominator: Tensor
    forecast_counts: Tensor
    presence_counts: Tensor

    @property
    def pinball(self) -> Tensor:
        return _safe_ratio(self.pinball_numerator, self.pinball_denominator)

    @property
    def median_huber(self) -> Tensor:
        return _safe_ratio(self.median_huber_numerator, self.median_huber_denominator)

    @property
    def presence(self) -> Tensor:
        return _safe_ratio(self.presence_numerator, self.presence_denominator)

    def detached(self) -> ChmtQLossStatistics:
        return ChmtQLossStatistics(
            pinball_numerator=self.pinball_numerator.detach(),
            pinball_denominator=self.pinball_denominator.detach(),
            median_huber_numerator=self.median_huber_numerator.detach(),
            median_huber_denominator=self.median_huber_denominator.detach(),
            presence_numerator=self.presence_numerator.detach(),
            presence_denominator=self.presence_denominator.detach(),
            forecast_counts=self.forecast_counts.detach(),
            presence_counts=self.presence_counts.detach(),
        )


@dataclass(frozen=True, slots=True)
class ChmtQLossTerms:
    """Inspectable scalar terms for logging and deterministic tests."""

    total: Tensor
    final_objective: Tensor
    pinball: Tensor
    median_huber: Tensor
    presence: Tensor
    auxiliary_objective: Tensor
    auxiliary_scale: float
    statistics: ChmtQLossStatistics
    auxiliary_statistics: ChmtQLossStatistics | None


class ChmtQLossStatisticsAccumulator:
    """Fixed-size detached CPU accumulation for full-set evaluation."""

    __slots__ = ("_total", "_updates")

    def __init__(self) -> None:
        self._total: ChmtQLossStatistics | None = None
        self._updates = 0

    @property
    def updates(self) -> int:
        return self._updates

    def update(
        self, statistics: ChmtQLossStatistics
    ) -> ChmtQLossStatisticsAccumulator:
        if not isinstance(statistics, ChmtQLossStatistics):
            raise TypeError("statistics must be ChmtQLossStatistics")
        detached = _detached_cpu_statistics(statistics)
        if self._total is None:
            self._total = detached
        else:
            _validate_compatible_statistics(self._total, detached)
            with torch.no_grad():
                self._total.pinball_numerator.add_(detached.pinball_numerator)
                self._total.pinball_denominator.add_(detached.pinball_denominator)
                self._total.median_huber_numerator.add_(
                    detached.median_huber_numerator
                )
                self._total.median_huber_denominator.add_(
                    detached.median_huber_denominator
                )
                self._total.presence_numerator.add_(detached.presence_numerator)
                self._total.presence_denominator.add_(detached.presence_denominator)
                self._total.forecast_counts.add_(detached.forecast_counts)
                self._total.presence_counts.add_(detached.presence_counts)
        self._updates += 1
        return self

    def finalize(self) -> ChmtQLossStatistics:
        if self._total is None:
            raise ValueError("at least one CHMT-Q loss statistic is required")
        return _clone_statistics(self._total)


def aggregate_chmt_q_loss_statistics(
    values: list[ChmtQLossStatistics] | tuple[ChmtQLossStatistics, ...],
) -> ChmtQLossStatistics:
    """Add independently reduced batches without averaging their ratios."""

    if not values:
        raise ValueError("at least one CHMT-Q loss statistic is required")
    first = values[0]
    for value in values:
        if value.forecast_counts.shape != first.forecast_counts.shape:
            raise ValueError("forecast count shapes do not match")
        if value.presence_counts.shape != first.presence_counts.shape:
            raise ValueError("presence count shapes do not match")
        if value.pinball_numerator.device != first.pinball_numerator.device:
            raise ValueError("loss statistics must use one device")

    def add(name: str) -> Tensor:
        return torch.stack([getattr(value, name) for value in values]).sum(dim=0)

    return ChmtQLossStatistics(
        pinball_numerator=add("pinball_numerator"),
        pinball_denominator=add("pinball_denominator"),
        median_huber_numerator=add("median_huber_numerator"),
        median_huber_denominator=add("median_huber_denominator"),
        presence_numerator=add("presence_numerator"),
        presence_denominator=add("presence_denominator"),
        forecast_counts=add("forecast_counts"),
        presence_counts=add("presence_counts"),
    )


def _detached_cpu_statistics(
    statistics: ChmtQLossStatistics,
) -> ChmtQLossStatistics:
    def copied(value: Tensor) -> Tensor:
        return value.detach().to(device="cpu").clone()

    return ChmtQLossStatistics(
        pinball_numerator=copied(statistics.pinball_numerator),
        pinball_denominator=copied(statistics.pinball_denominator),
        median_huber_numerator=copied(statistics.median_huber_numerator),
        median_huber_denominator=copied(statistics.median_huber_denominator),
        presence_numerator=copied(statistics.presence_numerator),
        presence_denominator=copied(statistics.presence_denominator),
        forecast_counts=copied(statistics.forecast_counts),
        presence_counts=copied(statistics.presence_counts),
    )


def _clone_statistics(statistics: ChmtQLossStatistics) -> ChmtQLossStatistics:
    return ChmtQLossStatistics(
        pinball_numerator=statistics.pinball_numerator.clone(),
        pinball_denominator=statistics.pinball_denominator.clone(),
        median_huber_numerator=statistics.median_huber_numerator.clone(),
        median_huber_denominator=statistics.median_huber_denominator.clone(),
        presence_numerator=statistics.presence_numerator.clone(),
        presence_denominator=statistics.presence_denominator.clone(),
        forecast_counts=statistics.forecast_counts.clone(),
        presence_counts=statistics.presence_counts.clone(),
    )


def _validate_compatible_statistics(
    expected: ChmtQLossStatistics, actual: ChmtQLossStatistics
) -> None:
    for name in (
        "pinball_numerator",
        "pinball_denominator",
        "median_huber_numerator",
        "median_huber_denominator",
        "presence_numerator",
        "presence_denominator",
        "forecast_counts",
        "presence_counts",
    ):
        left = getattr(expected, name)
        right = getattr(actual, name)
        if left.shape != right.shape:
            raise ValueError(f"loss statistic shape differs: {name}")
        if left.dtype != right.dtype:
            raise ValueError(f"loss statistic dtype differs: {name}")


def chmt_q_objective_from_statistics(
    statistics: ChmtQLossStatistics,
    config: ChmtQConfig | None = None,
) -> Tensor:
    """Reconstruct the final-head objective from additive component sums."""

    cfg = config or ChmtQConfig()
    return (
        statistics.pinball
        + cfg.median_huber_weight * statistics.median_huber
        + cfg.presence_weight * statistics.presence
    )


def auxiliary_loss_weight(
    step: int, total_steps: int, config: ChmtQConfig | None = None
) -> float:
    """Return the frozen 60%-to-90% linear auxiliary-loss schedule."""

    cfg = config or ChmtQConfig()
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("total_steps must be a positive integer")
    progress = min(step / total_steps, 1.0)
    if progress <= cfg.auxiliary_decay_start:
        return cfg.auxiliary_initial_weight
    if progress >= cfg.auxiliary_decay_end:
        return cfg.auxiliary_final_weight
    span = cfg.auxiliary_decay_end - cfg.auxiliary_decay_start
    fraction = (progress - cfg.auxiliary_decay_start) / span
    return cfg.auxiliary_initial_weight + fraction * (
        cfg.auxiliary_final_weight - cfg.auxiliary_initial_weight
    )


def chmt_q_loss(
    output: ChmtQOutput,
    *,
    targets: Tensor | None = None,
    target_mask: Tensor | None = None,
    presence_labels: Tensor | None = None,
    presence_mask: Tensor | None = None,
    auxiliary_output: ChmtQOutput | None = None,
    step: int = 0,
    total_steps: int | None = None,
    config: ChmtQConfig | None = None,
) -> ChmtQLossTerms:
    """Compute masked final and optional deep-supervision objectives."""

    cfg = config or ChmtQConfig()
    output.validate(cfg)
    _validate_label_pairs(
        output,
        targets=targets,
        target_mask=target_mask,
        presence_labels=presence_labels,
        presence_mask=presence_mask,
        config=cfg,
    )

    statistics, has_forecast, has_presence = _loss_statistics(
        output,
        targets=targets,
        target_mask=target_mask,
        presence_labels=presence_labels,
        presence_mask=presence_mask,
        config=cfg,
    )
    if not has_forecast and not has_presence:
        raise NoSupervisedTermsError(
            "batch has no supervised forecast or presence term"
        )

    pinball = statistics.pinball
    huber = statistics.median_huber
    presence = statistics.presence
    final_objective = chmt_q_objective_from_statistics(statistics, cfg)
    auxiliary_objective = _zero(output)
    auxiliary_scale = 0.0
    auxiliary_statistics: ChmtQLossStatistics | None = None
    if auxiliary_output is not None:
        auxiliary_output.validate(cfg)
        if (
            auxiliary_output.quantiles.shape != output.quantiles.shape
            or auxiliary_output.presence_logits.shape != output.presence_logits.shape
            or auxiliary_output.quantiles.dtype != output.quantiles.dtype
            or auxiliary_output.quantiles.device != output.quantiles.device
        ):
            raise ValueError("final and auxiliary outputs must share shape, dtype, and device")
        if total_steps is None:
            raise ValueError("total_steps is required when auxiliary_output is supplied")
        auxiliary_statistics, _, _ = _loss_statistics(
            auxiliary_output,
            targets=targets,
            target_mask=target_mask,
            presence_labels=presence_labels,
            presence_mask=presence_mask,
            config=cfg,
        )
        auxiliary_objective = chmt_q_objective_from_statistics(
            auxiliary_statistics, cfg
        )
        auxiliary_scale = auxiliary_loss_weight(step, total_steps, cfg)

    total = final_objective + auxiliary_scale * auxiliary_objective
    if not bool(torch.isfinite(total).item()):
        raise ValueError("CHMT-Q objective is not finite")
    return ChmtQLossTerms(
        total=total,
        final_objective=final_objective,
        pinball=pinball,
        median_huber=huber,
        presence=presence,
        auxiliary_objective=auxiliary_objective,
        auxiliary_scale=auxiliary_scale,
        statistics=statistics,
        auxiliary_statistics=auxiliary_statistics,
    )


def _validate_label_pairs(
    output: ChmtQOutput,
    *,
    targets: Tensor | None,
    target_mask: Tensor | None,
    presence_labels: Tensor | None,
    presence_mask: Tensor | None,
    config: ChmtQConfig,
) -> None:
    forecast_present = targets is not None or target_mask is not None
    if forecast_present and (targets is None or target_mask is None):
        raise ValueError("targets and target_mask must both be set or both be None")
    presence_present = presence_labels is not None or presence_mask is not None
    if presence_present and (presence_labels is None or presence_mask is None):
        raise ValueError(
            "presence_labels and presence_mask must both be set or both be None"
        )

    batch_size = output.quantiles.shape[0]
    if targets is not None and target_mask is not None:
        shape = (batch_size, config.horizons, config.targets)
        _validate_labels(targets, target_mask, shape, output.quantiles, "target")
    if presence_labels is not None and presence_mask is not None:
        shape = (batch_size, config.horizons, config.presence_targets)
        _validate_labels(
            presence_labels,
            presence_mask,
            shape,
            output.presence_logits,
            "presence",
        )
        valid = presence_labels[presence_mask]
        if valid.numel() and not bool(((valid == 0.0) | (valid == 1.0)).all().item()):
            raise ValueError("supervised presence labels must be binary")


def _validate_labels(
    labels: Tensor,
    mask: Tensor,
    expected_shape: tuple[int, ...],
    reference: Tensor,
    name: str,
) -> None:
    if not isinstance(labels, Tensor) or not isinstance(mask, Tensor):
        raise TypeError(f"{name} labels and mask must be torch.Tensor instances")
    if tuple(labels.shape) != expected_shape or tuple(mask.shape) != expected_shape:
        raise ValueError(f"{name} labels and mask must have shape {expected_shape}")
    allowed_dtypes = {reference.dtype}
    if reference.dtype in (torch.float16, torch.bfloat16):
        allowed_dtypes.add(torch.float32)
    if labels.dtype not in allowed_dtypes:
        raise TypeError(
            f"{name} labels must match the output dtype, "
            "or use float32 with reduced-precision output"
        )
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} mask must have dtype torch.bool")
    if labels.device != reference.device or mask.device != reference.device:
        raise ValueError(f"{name} labels, mask, and output must use one device")
    finite = torch.where(mask, torch.isfinite(labels), torch.ones_like(mask))
    if not bool(finite.all().item()):
        raise ValueError(f"{name} labels contain NaN or infinity at a supervised cell")


def _loss_statistics(
    output: ChmtQOutput,
    *,
    targets: Tensor | None,
    target_mask: Tensor | None,
    presence_labels: Tensor | None,
    presence_mask: Tensor | None,
    config: ChmtQConfig,
) -> tuple[ChmtQLossStatistics, bool, bool]:
    (
        pinball_numerator,
        pinball_denominator,
        huber_numerator,
        huber_denominator,
        forecast_counts,
        has_forecast,
    ) = _forecast_statistics(output, targets, target_mask, config)
    (
        presence_numerator,
        presence_denominator,
        presence_counts,
        has_presence,
    ) = _presence_statistics(output, presence_labels, presence_mask)
    return (
        ChmtQLossStatistics(
            pinball_numerator=pinball_numerator,
            pinball_denominator=pinball_denominator,
            median_huber_numerator=huber_numerator,
            median_huber_denominator=huber_denominator,
            presence_numerator=presence_numerator,
            presence_denominator=presence_denominator,
            forecast_counts=forecast_counts,
            presence_counts=presence_counts,
        ),
        has_forecast,
        has_presence,
    )


def _forecast_statistics(
    output: ChmtQOutput,
    targets: Tensor | None,
    target_mask: Tensor | None,
    config: ChmtQConfig,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, bool]:
    if targets is None or target_mask is None or not bool(target_mask.any().item()):
        zero = _zero(output)
        denominator = zero.detach()
        counts = torch.zeros(
            config.horizons,
            config.targets,
            dtype=torch.int64,
            device=output.quantiles.device,
        )
        return zero, denominator, zero, denominator, counts, False

    quantiles = output.quantiles.float()
    median = output.median.float()
    targets_fp32 = targets.float()
    safe_targets = torch.where(target_mask, targets_fp32, median.detach())
    errors = safe_targets.unsqueeze(-1) - quantiles
    levels = torch.tensor(
        config.quantile_levels,
        dtype=torch.float32,
        device=quantiles.device,
    ).view(1, 1, 1, -1)
    pinball_cells = torch.maximum(levels * errors, (levels - 1.0) * errors)

    horizon_weights = torch.tensor(
        config.horizon_weights,
        dtype=torch.float32,
        device=quantiles.device,
    ).view(1, -1, 1)
    target_weights = torch.tensor(
        config.target_weights,
        dtype=torch.float32,
        device=quantiles.device,
    ).view(1, 1, -1)
    cell_weights = horizon_weights * target_weights
    quantile_weights = (
        target_mask.unsqueeze(-1).expand_as(quantiles)
        * cell_weights.unsqueeze(-1)
    )
    pinball_numerator = (pinball_cells * quantile_weights).sum()
    pinball_denominator = quantile_weights.sum()

    residual = median - safe_targets
    absolute = residual.abs()
    delta = config.huber_delta
    huber_cells = torch.where(
        absolute <= delta,
        0.5 * residual.square(),
        delta * (absolute - 0.5 * delta),
    )
    active_weights = target_mask * cell_weights
    huber_numerator = (huber_cells * active_weights).sum()
    huber_denominator = active_weights.sum()
    counts = target_mask.sum(dim=0, dtype=torch.int64)
    return (
        pinball_numerator,
        pinball_denominator,
        huber_numerator,
        huber_denominator,
        counts,
        True,
    )


def _presence_statistics(
    output: ChmtQOutput,
    labels: Tensor | None,
    mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, bool]:
    if labels is None or mask is None or not bool(mask.any().item()):
        zero = _zero(output)
        counts = torch.zeros(
            output.presence_logits.shape[1:],
            dtype=torch.int64,
            device=output.presence_logits.device,
        )
        return zero, zero.detach(), counts, False
    logits = output.presence_logits.float()
    labels_fp32 = labels.float()
    safe_labels = torch.where(mask, labels_fp32, torch.zeros_like(labels_fp32))
    cells = F.binary_cross_entropy_with_logits(
        logits, safe_labels, reduction="none"
    )
    return (
        cells.masked_select(mask).sum(),
        mask.sum(dtype=torch.float32),
        mask.sum(dim=0, dtype=torch.int64),
        True,
    )


def _safe_ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
    return numerator / denominator.clamp_min(1.0)


def _zero(output: ChmtQOutput) -> Tensor:
    return (output.quantiles.float().sum() + output.presence_logits.float().sum()) * 0.0


class ChmtQLoss(nn.Module):
    """Module wrapper returning the scalar CHMT-Q objective."""

    def __init__(self, config: ChmtQConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChmtQConfig()

    def forward(
        self,
        output: ChmtQOutput,
        *,
        targets: Tensor | None = None,
        target_mask: Tensor | None = None,
        presence_labels: Tensor | None = None,
        presence_mask: Tensor | None = None,
        auxiliary_output: ChmtQOutput | None = None,
        step: int = 0,
        total_steps: int | None = None,
    ) -> Tensor:
        return chmt_q_loss(
            output,
            targets=targets,
            target_mask=target_mask,
            presence_labels=presence_labels,
            presence_mask=presence_mask,
            auxiliary_output=auxiliary_output,
            step=step,
            total_steps=total_steps,
            config=self.config,
        ).total
