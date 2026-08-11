"""Mask-aware learned pooling primitives."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..masks import _zero_scalar_like, masked_softmax


class LearnedQueryPool(nn.Module):
    """Pool a variable-validity token set with one learned query."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        self.d_model = d_model
        self.query = nn.Parameter(torch.empty(d_model))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        values: Tensor,
        valid_mask: Tensor,
        *,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if values.ndim != 3 or valid_mask.ndim != 2:
            raise ValueError("values and valid_mask must have shapes [B,T,D] and [B,T]")
        if tuple(values.shape[:2]) != tuple(valid_mask.shape):
            raise ValueError("valid_mask must match values batch/token dimensions")
        if values.shape[-1] != self.d_model:
            raise ValueError(f"values must have width {self.d_model}")
        if valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must have dtype torch.bool")
        scores = torch.matmul(values, self.query.to(dtype=values.dtype)) / math.sqrt(self.d_model)
        weights = masked_softmax(scores, valid_mask, dim=-1)
        safe_values = torch.where(
            valid_mask.unsqueeze(-1), values, _zero_scalar_like(values)
        )
        pooled = torch.sum(weights.unsqueeze(-1) * safe_values, dim=1)
        pooled = torch.where(
            valid_mask.any(dim=1, keepdim=True), pooled, _zero_scalar_like(pooled)
        )
        if need_weights:
            return pooled, weights
        return pooled


class CausalPatchPool(nn.Module):
    """Learned pooling over fixed, non-overlapping past-only frame patches."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 32,
        patch_width: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0 or patch_width <= 0:
            raise ValueError("d_model, hidden_dim, and patch_width must be positive")
        self.d_model = d_model
        self.patch_width = patch_width
        self.score_in = nn.Linear(d_model, hidden_dim)
        self.score_out = nn.Linear(hidden_dim, 1)

    def forward(self, frames: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor]:
        if frames.ndim != 3 or frame_mask.ndim != 2:
            raise ValueError("frames and frame_mask must have shapes [B,T,D] and [B,T]")
        if tuple(frames.shape[:2]) != tuple(frame_mask.shape):
            raise ValueError("frame_mask must match frames batch/time dimensions")
        if frames.shape[-1] != self.d_model:
            raise ValueError(f"frames must have width {self.d_model}")
        if frame_mask.dtype != torch.bool:
            raise TypeError("frame_mask must have dtype torch.bool")
        if frames.shape[1] % self.patch_width:
            raise ValueError(
                f"frame count {frames.shape[1]} must be divisible by patch_width "
                f"{self.patch_width}"
            )

        batch, frame_count, width = frames.shape
        patch_count = frame_count // self.patch_width
        patched_frames = frames.reshape(batch, patch_count, self.patch_width, width)
        patched_mask = frame_mask.reshape(batch, patch_count, self.patch_width)
        safe_frames = torch.where(
            patched_mask.unsqueeze(-1), patched_frames, _zero_scalar_like(patched_frames)
        )
        scores = self.score_out(torch.tanh(self.score_in(safe_frames))).squeeze(-1)
        weights = masked_softmax(scores, patched_mask, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * safe_frames, dim=2)
        patch_mask = patched_mask.any(dim=-1)
        pooled = torch.where(
            patch_mask.unsqueeze(-1), pooled, _zero_scalar_like(pooled)
        )
        return pooled, patch_mask


def causal_patch_pool(
    pool: CausalPatchPool,
    frames: Tensor,
    frame_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Functional spelling for callers that keep the learned pool explicitly."""

    if not isinstance(pool, CausalPatchPool):
        raise TypeError("pool must be a CausalPatchPool instance")
    return pool(frames, frame_mask)
