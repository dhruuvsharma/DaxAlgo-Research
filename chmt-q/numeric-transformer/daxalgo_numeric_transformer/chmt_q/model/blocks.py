"""RMSNorm, elapsed-time, SwiGLU, and LayerScale CHMT-Q blocks."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import ChmtQAttention, _working_values, _zero


def apply_token_mask(values: Tensor, valid_mask: Tensor) -> Tensor:
    if valid_mask.dtype != torch.bool:
        raise TypeError("valid_mask must be boolean")
    if tuple(values.shape[:-1]) != tuple(valid_mask.shape):
        raise ValueError("valid_mask must match all token dimensions")
    if values.device != valid_mask.device:
        raise ValueError("values and valid_mask must use one device")
    return torch.where(valid_mask.unsqueeze(-1), values, _zero(values))


def causal_mask(token_count: int, *, device: torch.device) -> Tensor:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    positions = torch.arange(token_count, device=device)
    return positions.unsqueeze(0) <= positions.unsqueeze(1)


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned gain and FP32 reductions."""

    def __init__(self, width: int, epsilon: float = 1.0e-6) -> None:
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        if epsilon <= 0.0 or not math.isfinite(epsilon):
            raise ValueError("epsilon must be finite and positive")
        self.width = width
        self.epsilon = epsilon
        self.gain = nn.Parameter(torch.ones(width))

    def forward(self, values: Tensor) -> Tensor:
        if not values.is_floating_point() or values.shape[-1] != self.width:
            raise ValueError(f"values must be floating point with final width {self.width}")
        work = _working_values(values)
        normalized = work * torch.rsqrt(work.square().mean(dim=-1, keepdim=True) + self.epsilon)
        normalized = normalized.to(dtype=values.dtype)
        return normalized * self.gain.to(dtype=values.dtype)


class NumericTimeEncoding(nn.Module):
    """Project deterministic physical-age features to a token width."""

    def __init__(
        self,
        output_dim: int,
        scales: Sequence[float],
        *,
        hidden_dim: int,
        context_max: float,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive")
        if context_max <= 0.0 or not math.isfinite(context_max):
            raise ValueError("context_max must be finite and positive")
        scale_values = tuple(float(value) for value in scales)
        if not scale_values or any(value <= 0.0 for value in scale_values):
            raise ValueError("scales must contain finite positive values")
        scale_tensor = torch.tensor(scale_values, dtype=torch.float32)
        if not bool(torch.isfinite(scale_tensor).all().item()):
            raise ValueError("scales must contain finite positive values")
        self.register_buffer("scales", scale_tensor)
        self.register_buffer("context_max", torch.tensor(float(context_max)))
        feature_dim = 2 + 2 * len(scale_values)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, ages: Tensor) -> Tensor:
        if ages.ndim not in (1, 2) or not ages.is_floating_point():
            raise ValueError("ages must be floating point with shape [T] or [B,T]")
        if ages.device != self.scales.device:
            raise ValueError("ages and time encoder must use one device")
        work = _working_values(ages).clamp_min(0.0)
        context_max = self.context_max.to(dtype=work.dtype)
        scales = self.scales.to(dtype=work.dtype)
        phases = work.unsqueeze(-1) / scales
        features = torch.cat(
            (
                (work / context_max).unsqueeze(-1),
                (torch.log1p(work) / torch.log1p(context_max)).unsqueeze(-1),
                torch.sin(phases),
                torch.cos(phases),
            ),
            dim=-1,
        ).to(dtype=ages.dtype)
        return self.output_projection(F.silu(self.input_projection(features)))


class RelativeTimeBias(nn.Module):
    """Learned nonnegative per-head recency penalty over physical elapsed time."""

    def __init__(self, num_heads: int, time_scale: float) -> None:
        super().__init__()
        if num_heads <= 0 or time_scale <= 0.0 or not math.isfinite(time_scale):
            raise ValueError("num_heads and time_scale must be positive")
        self.gamma = nn.Parameter(torch.zeros(num_heads))
        self.register_buffer("time_scale", torch.tensor(float(time_scale)))

    def forward(self, query_ages: Tensor, key_ages: Tensor | None = None) -> Tensor:
        keys = query_ages if key_ages is None else key_ages
        if query_ages.ndim != 1 or keys.ndim != 1:
            raise ValueError("query_ages and key_ages must be one-dimensional")
        if query_ages.device != self.gamma.device or keys.device != self.gamma.device:
            raise ValueError("age tensors and relative-time bias must use one device")
        query_work = _working_values(query_ages)
        key_work = _working_values(keys)
        elapsed = (key_work.unsqueeze(0) - query_work.unsqueeze(1)).clamp_min(0.0)
        distance = torch.log1p(elapsed / self.time_scale.to(dtype=elapsed.dtype))
        decay = F.softplus(self.gamma).to(dtype=distance.dtype)
        return -decay[None, :, None, None] * distance[None, None, :, :]


class SwiGLU(nn.Module):
    """SiLU-gated feed-forward network used by every CHMT-Q block."""

    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0:
            raise ValueError("d_model and hidden_dim must be positive")
        self.gate_projection = nn.Linear(d_model, hidden_dim)
        self.value_projection = nn.Linear(d_model, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, d_model)

    def forward(self, values: Tensor) -> Tensor:
        gated = F.silu(self.gate_projection(values)) * self.value_projection(values)
        return self.output_projection(gated)


class ChmtQEncoderBlock(nn.Module):
    """Pre-RMSNorm attention/SwiGLU block with explicit LayerScale."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        *,
        rms_epsilon: float,
        qk_epsilon: float,
        layer_scale_initial: float,
        causal: bool,
        relative_time_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.attention_norm = RMSNorm(d_model, rms_epsilon)
        self.attention = ChmtQAttention(d_model, num_heads, dropout, qk_epsilon=qk_epsilon)
        self.attention_dropout = nn.Dropout(dropout)
        self.ffn_norm = RMSNorm(d_model, rms_epsilon)
        self.ffn = SwiGLU(d_model, ff_dim)
        self.ffn_dropout = nn.Dropout(dropout)
        self.attention_layer_scale = nn.Parameter(
            torch.full((d_model,), float(layer_scale_initial))
        )
        self.ffn_layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale_initial)))
        self.relative_time_bias = (
            None
            if relative_time_scale is None
            else RelativeTimeBias(num_heads, relative_time_scale)
        )

    def forward(
        self,
        values: Tensor,
        valid_mask: Tensor,
        *,
        ages: Tensor | None = None,
    ) -> Tensor:
        if values.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError("values and valid_mask must have shapes [B,T,D] and [B,T]")
        if tuple(values.shape[:2]) != tuple(valid_mask.shape):
            raise ValueError("valid_mask must match values")
        token_count = values.shape[1]
        triangle = causal_mask(token_count, device=values.device) if self.causal else None
        bias = None
        if self.relative_time_bias is not None:
            if ages is None or tuple(ages.shape) != (token_count,):
                raise ValueError(f"ages must have shape {(token_count,)}")
            bias = self.relative_time_bias(ages)

        state = apply_token_mask(values, valid_mask)
        attention_output = self.attention(
            self.attention_norm(state),
            key_mask=valid_mask,
            query_mask=valid_mask,
            attention_mask=triangle,
            additive_bias=bias,
        )
        state = state + self.attention_layer_scale * self.attention_dropout(attention_output)
        state = apply_token_mask(state, valid_mask)
        ffn_output = self.ffn(self.ffn_norm(state))
        state = state + self.ffn_layer_scale * self.ffn_dropout(ffn_output)
        return apply_token_mask(state, valid_mask)


class ChmtQDecoderBlock(nn.Module):
    """Non-causal horizon self-attention, memory cross-attention, and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        *,
        rms_epsilon: float,
        qk_epsilon: float,
        layer_scale_initial: float,
    ) -> None:
        super().__init__()
        self.self_norm = RMSNorm(d_model, rms_epsilon)
        self.self_attention = ChmtQAttention(d_model, num_heads, dropout, qk_epsilon=qk_epsilon)
        self.cross_norm = RMSNorm(d_model, rms_epsilon)
        self.cross_attention = ChmtQAttention(d_model, num_heads, dropout, qk_epsilon=qk_epsilon)
        self.ffn_norm = RMSNorm(d_model, rms_epsilon)
        self.ffn = SwiGLU(d_model, ff_dim)
        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.self_layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale_initial)))
        self.cross_layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale_initial)))
        self.ffn_layer_scale = nn.Parameter(torch.full((d_model,), float(layer_scale_initial)))

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        *,
        memory_bias: Tensor | None = None,
    ) -> Tensor:
        if queries.ndim != 3 or memory.ndim != 3 or memory_mask.ndim != 2:
            raise ValueError("decoder inputs must have shapes [B,Q,D], [B,K,D], [B,K]")
        if tuple(memory.shape[:2]) != tuple(memory_mask.shape):
            raise ValueError("memory_mask must match memory")
        query_mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)

        self_output = self.self_attention(
            self.self_norm(queries), key_mask=query_mask, query_mask=query_mask
        )
        state = queries + self.self_layer_scale * self.self_dropout(self_output)
        cross_output = self.cross_attention(
            self.cross_norm(state),
            memory,
            key_mask=memory_mask,
            query_mask=query_mask,
            additive_bias=memory_bias,
        )
        state = state + self.cross_layer_scale * self.cross_dropout(cross_output)
        ffn_output = self.ffn(self.ffn_norm(state))
        return state + self.ffn_layer_scale * self.ffn_dropout(ffn_output)
