"""Primitive Q/K-normalized attention for the selected CHMT-Q model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

DEFAULT_LOGIT_RUNAWAY_THRESHOLD = 80.0


@dataclass(frozen=True, slots=True)
class AttentionLogitTelemetry:
    """Running eligible pre-softmax maximum for every attention head."""

    max_abs_by_head: tuple[float, ...]
    observations: int
    runaway_threshold: float


def _zero(values: Tensor) -> Tensor:
    return torch.zeros((), dtype=values.dtype, device=values.device)


def _working_values(values: Tensor) -> Tensor:
    if values.dtype in (torch.float16, torch.bfloat16):
        return values.float()
    return values


def safe_masked_softmax(scores: Tensor, valid_mask: Tensor, dim: int = -1) -> Tensor:
    """Normalize only eligible entries and return exact zero for an empty row."""

    if not isinstance(scores, Tensor) or not scores.is_floating_point():
        raise TypeError("scores must be a floating-point torch.Tensor")
    if not isinstance(valid_mask, Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be a boolean torch.Tensor")
    if valid_mask.device != scores.device:
        raise ValueError("scores and valid_mask must use one device")
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(scores.shape), tuple(valid_mask.shape))
    except RuntimeError as error:
        raise ValueError("valid_mask is not broadcastable to scores") from error
    if broadcast_shape != tuple(scores.shape):
        raise ValueError("valid_mask must not expand the score tensor")

    mask = valid_mask.expand_as(scores)
    work = _working_values(scores)
    negative_large = torch.finfo(work.dtype).min
    eligible_scores = torch.where(mask, work, work.new_tensor(negative_large))
    eligible_max = eligible_scores.amax(dim=dim, keepdim=True)
    shifted = eligible_scores - eligible_max
    exponentials = torch.where(mask, torch.exp(shifted), _zero(work))
    denominator = exponentials.sum(dim=dim, keepdim=True)
    has_eligible_key = mask.any(dim=dim, keepdim=True)
    safe_denominator = torch.where(has_eligible_key, denominator, torch.ones_like(denominator))
    probabilities = exponentials / safe_denominator
    probabilities = torch.where(mask & has_eligible_key, probabilities, _zero(probabilities))
    return probabilities.to(dtype=scores.dtype)


def rms_normalize_heads(values: Tensor, epsilon: float) -> Tensor:
    """RMS-normalize the final per-head dimension without a learned gain."""

    if not values.is_floating_point():
        raise TypeError("head values must use a floating-point dtype")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    work = _working_values(values)
    inverse_rms = torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + epsilon)
    return (work * inverse_rms).to(dtype=values.dtype)


class ChmtQAttention(nn.Module):
    """Explicit multi-head attention with learned head temperatures and hard masks."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        qk_epsilon: float = 1.0e-6,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if qk_epsilon <= 0.0 or not math.isfinite(qk_epsilon):
            raise ValueError("qk_epsilon must be finite and positive")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qk_epsilon = qk_epsilon
        self.score_scale = 1.0 / math.sqrt(self.head_dim)
        self.q_projection = nn.Linear(d_model, d_model, bias=bias)
        self.k_projection = nn.Linear(d_model, d_model, bias=bias)
        self.v_projection = nn.Linear(d_model, d_model, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        self.log_temperature = nn.Parameter(torch.zeros(num_heads))
        self.attention_dropout = nn.Dropout(dropout)
        self._logit_telemetry_enabled = False
        self._logit_runaway_threshold = DEFAULT_LOGIT_RUNAWAY_THRESHOLD
        self._logit_max_abs: tuple[float, ...] | None = None
        self._logit_observations = 0

    def forward(
        self,
        query: Tensor,
        key_value: Tensor | None = None,
        *,
        key_mask: Tensor | None = None,
        query_mask: Tensor | None = None,
        attention_mask: Tensor | None = None,
        additive_bias: Tensor | None = None,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        memory = query if key_value is None else key_value
        self._validate_inputs(query, memory, key_mask, query_mask)
        batch_size, query_count, _ = query.shape
        key_count = memory.shape[1]

        safe_query = query
        if query_mask is not None:
            safe_query = torch.where(query_mask.unsqueeze(-1), query, _zero(query))
        safe_memory = memory
        if key_mask is not None:
            safe_memory = torch.where(key_mask.unsqueeze(-1), memory, _zero(memory))

        q = rms_normalize_heads(self._split_heads(self.q_projection(safe_query)), self.qk_epsilon)
        k = rms_normalize_heads(self._split_heads(self.k_projection(safe_memory)), self.qk_epsilon)
        v = self._split_heads(self.v_projection(safe_memory))

        q_work = _working_values(q)
        k_work = _working_values(k)
        temperature = torch.exp(_working_values(self.log_temperature)).to(dtype=q_work.dtype)
        scores = torch.matmul(q_work, k_work.transpose(-2, -1))
        scores = scores * (temperature[None, :, None, None] * self.score_scale)

        if additive_bias is not None:
            if not additive_bias.is_floating_point() or additive_bias.device != scores.device:
                raise ValueError("additive_bias must be floating point on the attention device")
            self._validate_broadcastable(additive_bias, scores, "additive_bias")
            scores = scores + additive_bias.to(dtype=scores.dtype)

        valid = torch.ones((), dtype=torch.bool, device=query.device)
        if key_mask is not None:
            valid = valid & key_mask[:, None, None, :]
        if query_mask is not None:
            valid = valid & query_mask[:, None, :, None]
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool or attention_mask.device != query.device:
                raise ValueError("attention_mask must be boolean on the attention device")
            mask_view = self._attention_mask_view(
                attention_mask,
                batch_size=batch_size,
                query_count=query_count,
                key_count=key_count,
            )
            self._validate_broadcastable(mask_view, scores, "attention_mask")
            valid = valid & mask_view

        if self._logit_telemetry_enabled:
            self._observe_logits(scores, valid)
        weights = safe_masked_softmax(scores, valid, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        value_work = _working_values(v)
        context = torch.matmul(dropped_weights.to(value_work.dtype), value_work)
        context = context.to(dtype=query.dtype)
        context = context.transpose(1, 2).contiguous().view(batch_size, query_count, self.d_model)
        output = self.output_projection(context)

        row_has_key = valid.expand(batch_size, self.num_heads, query_count, key_count)
        effective_query = row_has_key.any(dim=-1).any(dim=1)
        output = torch.where(effective_query.unsqueeze(-1), output, _zero(output))
        if need_weights:
            weights = torch.where(effective_query[:, None, :, None], weights, _zero(weights))
            return output, weights
        return output

    def enable_logit_telemetry(
        self,
        *,
        runaway_threshold: float = DEFAULT_LOGIT_RUNAWAY_THRESHOLD,
    ) -> None:
        """Enable fail-fast eligible-logit monitoring and start a fresh window.

        The default absolute threshold of 80 is deliberately far above the
        reference model's Q/K-normalized operating range while remaining low
        enough to expose a diverging learned temperature before softmax.
        """

        if (
            isinstance(runaway_threshold, bool)
            or not isinstance(runaway_threshold, (int, float))
            or not math.isfinite(float(runaway_threshold))
            or float(runaway_threshold) <= 0.0
        ):
            raise ValueError("runaway_threshold must be finite and positive")
        self._logit_telemetry_enabled = True
        self._logit_runaway_threshold = float(runaway_threshold)
        self.reset_logit_telemetry()

    def disable_logit_telemetry(self, *, reset: bool = False) -> None:
        """Disable monitoring; optionally discard the last observation window."""

        self._logit_telemetry_enabled = False
        if reset:
            self.reset_logit_telemetry()

    def reset_logit_telemetry(self) -> None:
        """Discard all recorded maxima without changing enabled state."""

        self._logit_max_abs = None
        self._logit_observations = 0

    def collect_logit_telemetry(self) -> AttentionLogitTelemetry | None:
        """Return the current running maximum, or ``None`` before observation."""

        if self._logit_max_abs is None:
            return None
        return AttentionLogitTelemetry(
            max_abs_by_head=self._logit_max_abs,
            observations=self._logit_observations,
            runaway_threshold=self._logit_runaway_threshold,
        )

    def _observe_logits(self, scores: Tensor, valid: Tensor) -> None:
        eligible = valid.expand_as(scores)
        detached = scores.detach()
        finite = torch.where(eligible, torch.isfinite(detached), torch.ones_like(eligible))
        if not bool(finite.all().item()):
            raise FloatingPointError("attention produced a non-finite eligible pre-softmax logit")
        head_max = torch.where(eligible, detached.abs(), _zero(detached)).amax(
            dim=(0, 2, 3)
        )
        runaway = head_max > self._logit_runaway_threshold
        if bool(runaway.any().item()):
            heads = runaway.nonzero(as_tuple=False).flatten().tolist()
            observed = float(head_max.max().item())
            raise FloatingPointError(
                "attention pre-softmax logit exceeded "
                f"{self._logit_runaway_threshold:g}; heads={heads}, max={observed:g}"
            )
        current = tuple(float(value) for value in head_max.cpu().tolist())
        if self._logit_max_abs is None:
            self._logit_max_abs = current
        else:
            self._logit_max_abs = tuple(
                max(previous, observed)
                for previous, observed in zip(self._logit_max_abs, current, strict=True)
            )
        self._logit_observations += 1

    def _split_heads(self, values: Tensor) -> Tensor:
        batch_size, token_count, _ = values.shape
        return values.view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)

    def _validate_inputs(
        self,
        query: Tensor,
        memory: Tensor,
        key_mask: Tensor | None,
        query_mask: Tensor | None,
    ) -> None:
        if query.ndim != 3 or memory.ndim != 3:
            raise ValueError("query and key_value must have shape [B,T,D]")
        if query.shape[0] != memory.shape[0]:
            raise ValueError("query and key_value batch sizes must match")
        if query.shape[-1] != self.d_model or memory.shape[-1] != self.d_model:
            raise ValueError(f"attention inputs must have width {self.d_model}")
        if query.dtype != memory.dtype or query.device != memory.device:
            raise ValueError("query and key_value must share dtype and device")
        if not query.is_floating_point():
            raise TypeError("attention inputs must use a floating-point dtype")
        if key_mask is not None:
            self._validate_sequence_mask(key_mask, memory.shape[:2], "key_mask", query.device)
        if query_mask is not None:
            self._validate_sequence_mask(query_mask, query.shape[:2], "query_mask", query.device)

    @staticmethod
    def _validate_sequence_mask(
        mask: Tensor,
        expected_shape: torch.Size,
        name: str,
        device: torch.device,
    ) -> None:
        if mask.dtype != torch.bool or tuple(mask.shape) != tuple(expected_shape):
            raise ValueError(f"{name} must be boolean with shape {tuple(expected_shape)}")
        if mask.device != device:
            raise ValueError(f"{name} must use the attention device")

    @staticmethod
    def _attention_mask_view(
        mask: Tensor,
        *,
        batch_size: int,
        query_count: int,
        key_count: int,
    ) -> Tensor:
        if mask.ndim == 2 and tuple(mask.shape) == (query_count, key_count):
            return mask[None, None, :, :]
        if mask.ndim == 3 and tuple(mask.shape) == (
            batch_size,
            query_count,
            key_count,
        ):
            return mask[:, None, :, :]
        if mask.ndim == 4:
            return mask
        raise ValueError("attention_mask has an invalid rank or shape")

    @staticmethod
    def _validate_broadcastable(value: Tensor, target: Tensor, name: str) -> None:
        try:
            result = torch.broadcast_shapes(tuple(value.shape), tuple(target.shape))
        except RuntimeError as error:
            raise ValueError(f"{name} is not broadcastable to attention scores") from error
        if result != tuple(target.shape):
            raise ValueError(f"{name} must not expand attention scores")


PrimitiveChmtQAttention = ChmtQAttention
