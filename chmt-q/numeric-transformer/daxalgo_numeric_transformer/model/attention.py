"""Primitive Q/K/V attention used throughout the numeric transformer."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..masks import _zero_scalar_like, masked_softmax


class PrimitiveMultiheadAttention(nn.Module):
    """Multi-head attention built only from projections and tensor primitives.

    The implementation uses only explicit projections, matrix multiplication,
    masking, and softmax. ``True`` mask entries are valid. Invalid queries and
    rows with no valid key return exact zeros.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)

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
        """Apply self-attention or cross-attention.

        Args:
            query: ``[B,Q,D]`` query tokens.
            key_value: optional ``[B,K,D]`` memory; omitted for self-attention.
            key_mask: optional ``[B,K]`` validity mask.
            query_mask: optional ``[B,Q]`` validity mask.
            attention_mask: optional boolean mask broadcastable to ``[B,H,Q,K]``.
            additive_bias: optional numeric bias broadcastable to ``[B,H,Q,K]``.
            need_weights: return per-head weights as a second result when true.
        """

        memory = query if key_value is None else key_value
        self._validate_inputs(query, memory, key_mask, query_mask)
        batch, query_length, _ = query.shape
        key_length = memory.shape[1]

        safe_query = (
            query
            if query_mask is None
            else torch.where(query_mask.unsqueeze(-1), query, _zero_scalar_like(query))
        )
        safe_memory = (
            memory
            if key_mask is None
            else torch.where(key_mask.unsqueeze(-1), memory, _zero_scalar_like(memory))
        )
        q = self._split_heads(self.q_proj(safe_query))
        k = self._split_heads(self.k_proj(safe_memory))
        v = self._split_heads(self.v_proj(safe_memory))
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if additive_bias is not None:
            if not additive_bias.is_floating_point():
                raise TypeError("additive_bias must use a floating-point dtype")
            if additive_bias.device != scores.device:
                raise ValueError("additive_bias and attention inputs must share a device")
            self._validate_broadcastable(additive_bias, scores, "additive_bias")
            scores = scores + additive_bias.to(dtype=scores.dtype)

        valid = torch.ones((), dtype=torch.bool, device=query.device)
        if key_mask is not None:
            valid = valid & key_mask[:, None, None, :]
        if attention_mask is not None:
            if attention_mask.dtype != torch.bool:
                raise TypeError("attention_mask must have dtype torch.bool")
            if attention_mask.device != query.device:
                raise ValueError("attention_mask and attention inputs must share a device")
            expanded_attention_mask = self._attention_mask_view(
                attention_mask,
                batch=batch,
                query_length=query_length,
                key_length=key_length,
            )
            self._validate_broadcastable(expanded_attention_mask, scores, "attention_mask")
            valid = valid & expanded_attention_mask

        weights = masked_softmax(scores, valid, dim=-1)
        dropped_weights = self.attention_dropout(weights)
        context = torch.matmul(dropped_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, query_length, self.d_model)
        output = self.out_proj(context)

        row_has_key = (
            valid.expand(batch, self.num_heads, query_length, key_length)
            .any(dim=-1)
            .any(dim=1)
        )
        effective_query_mask = row_has_key
        if query_mask is not None:
            effective_query_mask = effective_query_mask & query_mask
        output = torch.where(
            effective_query_mask.unsqueeze(-1),
            output,
            _zero_scalar_like(output),
        )
        if need_weights:
            weights = torch.where(
                effective_query_mask[:, None, :, None],
                weights,
                _zero_scalar_like(weights),
            )
            return output, weights
        return output

    def _split_heads(self, values: Tensor) -> Tensor:
        batch, tokens, _ = values.shape
        return values.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

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
        if query.device != memory.device or query.dtype != memory.dtype:
            raise ValueError("query and key_value must share dtype and device")
        if not query.is_floating_point():
            raise TypeError("attention inputs must use a floating-point dtype")
        if key_mask is not None:
            self._validate_sequence_mask(
                key_mask,
                (memory.shape[0], memory.shape[1]),
                "key_mask",
                memory.device,
            )
        if query_mask is not None:
            self._validate_sequence_mask(
                query_mask,
                (query.shape[0], query.shape[1]),
                "query_mask",
                query.device,
            )

    @staticmethod
    def _validate_sequence_mask(
        mask: Tensor,
        expected_shape: tuple[int, int],
        name: str,
        device: torch.device,
    ) -> None:
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool")
        if tuple(mask.shape) != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(mask.shape)}")
        if mask.device != device:
            raise ValueError(f"{name} and attention inputs must share a device")

    @staticmethod
    def _attention_mask_view(
        mask: Tensor,
        *,
        batch: int,
        query_length: int,
        key_length: int,
    ) -> Tensor:
        if mask.ndim == 2:
            if tuple(mask.shape) != (query_length, key_length):
                raise ValueError(
                    "2D attention_mask must have shape "
                    f"{(query_length, key_length)}, got {tuple(mask.shape)}"
                )
            return mask[None, None, :, :]
        if mask.ndim == 3:
            if tuple(mask.shape) != (batch, query_length, key_length):
                raise ValueError(
                    "3D attention_mask must have shape "
                    f"{(batch, query_length, key_length)}, got {tuple(mask.shape)}"
                )
            return mask[:, None, :, :]
        if mask.ndim == 4:
            return mask
        raise ValueError("attention_mask must have rank 2, 3, or 4")

    @staticmethod
    def _validate_broadcastable(value: Tensor, target: Tensor, name: str) -> None:
        try:
            shape = torch.broadcast_shapes(tuple(value.shape), tuple(target.shape))
        except RuntimeError as exc:
            raise ValueError(
                f"{name} shape {tuple(value.shape)} is not broadcastable to {tuple(target.shape)}"
            ) from exc
        if shape != tuple(target.shape):
            raise ValueError(f"{name} would expand attention scores to {shape}")


# The same primitive implements self- and cross-attention.
PrimitiveSelfAttention = PrimitiveMultiheadAttention
PrimitiveCrossAttention = PrimitiveMultiheadAttention
