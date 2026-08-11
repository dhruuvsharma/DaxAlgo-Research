"""Pre-normalized primitive transformer blocks and numeric time encoding."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..masks import apply_token_mask, causal_mask
from .attention import PrimitiveMultiheadAttention


class NumericTimeEncoding(nn.Module):
    """Project deterministic numeric age features to a branch width."""

    def __init__(
        self,
        output_dim: int,
        scales: Sequence[float],
        *,
        hidden_dim: int = 32,
        context_max: float,
    ) -> None:
        super().__init__()
        if output_dim <= 0 or hidden_dim <= 0:
            raise ValueError("output_dim and hidden_dim must be positive")
        if context_max <= 0.0 or not math.isfinite(context_max):
            raise ValueError("context_max must be finite and positive")
        scale_tensor = torch.as_tensor(tuple(scales), dtype=torch.float32)
        if scale_tensor.ndim != 1 or scale_tensor.numel() == 0:
            raise ValueError("scales must be a non-empty one-dimensional sequence")
        if not bool(torch.isfinite(scale_tensor).all()) or bool((scale_tensor <= 0).any()):
            raise ValueError("scales must contain finite positive values")
        self.register_buffer("scales", scale_tensor)
        self.register_buffer("context_max", torch.tensor(float(context_max)))
        feature_dim = 2 + 2 * scale_tensor.numel()
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, ages: Tensor) -> Tensor:
        if ages.ndim not in (1, 2):
            raise ValueError("ages must have shape [T] or [B,T]")
        if not ages.is_floating_point():
            raise TypeError("ages must use a floating-point dtype")
        if ages.device != self.scales.device:
            raise ValueError("ages and time encoder must share a device")
        non_negative_ages = ages.clamp_min(0.0)
        context_max = self.context_max.to(dtype=ages.dtype)
        raw = non_negative_ages / context_max
        logged = torch.log1p(non_negative_ages) / torch.log1p(context_max)
        phase = non_negative_ages.unsqueeze(-1) / self.scales.to(dtype=ages.dtype)
        features = torch.cat(
            (raw.unsqueeze(-1), logged.unsqueeze(-1), torch.sin(phase), torch.cos(phase)),
            dim=-1,
        )
        return self.output_projection(F.silu(self.input_projection(features)))


class RelativeTimeBias(nn.Module):
    """Learned per-head decay over logical age distance."""

    def __init__(self, num_heads: int, time_scale: float) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if time_scale <= 0.0:
            raise ValueError("time_scale must be positive")
        self.gamma = nn.Parameter(torch.zeros(num_heads))
        self.register_buffer("time_scale", torch.tensor(float(time_scale)))

    def forward(self, query_ages: Tensor, key_ages: Tensor | None = None) -> Tensor:
        if query_ages.ndim != 1:
            raise ValueError("query_ages must have shape [Q]")
        keys = query_ages if key_ages is None else key_ages
        if keys.ndim != 1:
            raise ValueError("key_ages must have shape [K]")
        if query_ages.device != self.gamma.device or keys.device != self.gamma.device:
            raise ValueError("age tensors and relative-time module must share a device")
        delta = (keys.unsqueeze(0) - query_ages.unsqueeze(1)).clamp_min(0.0)
        distance = torch.log1p(delta / self.time_scale.to(dtype=delta.dtype))
        decay = F.softplus(self.gamma).to(dtype=distance.dtype)
        return -decay[None, :, None, None] * distance[None, None, :, :]


class GEGLU(nn.Module):
    """Gated GELU feed-forward network."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        if d_model <= 0 or ff_dim <= 0:
            raise ValueError("d_model and ff_dim must be positive")
        self.gate_projection = nn.Linear(d_model, ff_dim)
        self.value_projection = nn.Linear(d_model, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(ff_dim, d_model)

    def forward(self, values: Tensor) -> Tensor:
        gate = F.gelu(self.gate_projection(values))
        value = self.value_projection(values)
        return self.output_projection(self.dropout(gate * value))


class PreNormTransformerBlock(nn.Module):
    """Pre-norm attention + GEGLU block with exact invalid-query zeros."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
        *,
        causal: bool = False,
        relative_time_scale: float | None = None,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = PrimitiveMultiheadAttention(d_model, num_heads, dropout)
        self.attention_residual_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = GEGLU(d_model, ff_dim, dropout)
        self.ffn_residual_dropout = nn.Dropout(dropout)
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
            raise ValueError("valid_mask must match values batch/token dimensions")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        token_count = values.shape[1]
        attention_mask = (
            causal_mask(token_count, device=values.device) if self.causal else None
        )
        additive_bias = None
        if self.relative_time_bias is not None:
            if ages is None:
                raise ValueError("ages are required when relative_time_scale is configured")
            if tuple(ages.shape) != (token_count,):
                raise ValueError(f"ages must have shape {(token_count,)}, got {tuple(ages.shape)}")
            additive_bias = self.relative_time_bias(ages)

        masked_values = apply_token_mask(values, valid_mask)
        attention_output = self.attention(
            self.attention_norm(masked_values),
            key_mask=valid_mask,
            query_mask=valid_mask,
            attention_mask=attention_mask,
            additive_bias=additive_bias,
        )
        residual = apply_token_mask(
            masked_values + self.attention_residual_dropout(attention_output),
            valid_mask,
        )
        output = residual + self.ffn_residual_dropout(self.ffn(self.ffn_norm(residual)))
        return apply_token_mask(output, valid_mask)


class DecoderBlock(nn.Module):
    """Direct horizon-query self/cross-attention decoder block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(d_model)
        self.self_attention = PrimitiveMultiheadAttention(d_model, num_heads, dropout)
        self.cross_norm = nn.LayerNorm(d_model)
        self.cross_attention = PrimitiveMultiheadAttention(d_model, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = GEGLU(d_model, ff_dim, dropout)
        self.self_dropout = nn.Dropout(dropout)
        self.cross_dropout = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)

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
        if memory.shape[:2] != memory_mask.shape:
            raise ValueError("memory_mask must match memory batch/token dimensions")
        query_mask = torch.ones(
            queries.shape[:2], dtype=torch.bool, device=queries.device
        )
        self_output = self.self_attention(
            self.self_norm(queries),
            key_mask=query_mask,
            query_mask=query_mask,
        )
        queries = queries + self.self_dropout(self_output)
        cross_output = self.cross_attention(
            self.cross_norm(queries),
            memory,
            key_mask=memory_mask,
            query_mask=query_mask,
            additive_bias=memory_bias,
        )
        queries = queries + self.cross_dropout(cross_output)
        return queries + self.ffn_dropout(self.ffn(self.ffn_norm(queries)))


TransformerBlock = PreNormTransformerBlock
