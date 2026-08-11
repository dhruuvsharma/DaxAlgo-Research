"""Masked four-frame and hierarchical-book pooling for CHMT-Q."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .attention import _zero, safe_masked_softmax
from .blocks import RMSNorm


class MaskedPatchPool(nn.Module):
    """Pool fixed non-overlapping frames after adding learned within-patch offsets."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_dim: int = 32,
        patch_size: int = 4,
    ) -> None:
        super().__init__()
        if d_model <= 0 or hidden_dim <= 0 or patch_size <= 0:
            raise ValueError("d_model, hidden_dim, and patch_size must be positive")
        self.d_model = d_model
        self.patch_size = patch_size
        self.within_patch_embedding = nn.Parameter(torch.empty(patch_size, d_model))
        self.score_projection = nn.Linear(d_model, hidden_dim)
        self.score_output = nn.Linear(hidden_dim, 1)
        nn.init.trunc_normal_(self.within_patch_embedding, std=0.02)

    def forward(self, frames: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor]:
        if frames.ndim != 3 or frame_mask.ndim != 2:
            raise ValueError("frames and frame_mask must have shapes [B,T,D] and [B,T]")
        if tuple(frames.shape[:2]) != tuple(frame_mask.shape):
            raise ValueError("frame_mask must match frames")
        if frames.shape[-1] != self.d_model or frame_mask.dtype != torch.bool:
            raise ValueError("frames have the wrong width or frame_mask is not boolean")
        if frames.shape[1] % self.patch_size:
            raise ValueError("frame count must be divisible by patch_size")

        batch_size, frame_count, width = frames.shape
        patch_count = frame_count // self.patch_size
        patched = frames.reshape(batch_size, patch_count, self.patch_size, width)
        patched_mask = frame_mask.reshape(batch_size, patch_count, self.patch_size)
        with_offsets = patched + self.within_patch_embedding[None, None, :, :].to(
            dtype=frames.dtype
        )
        safe_frames = torch.where(patched_mask.unsqueeze(-1), with_offsets, _zero(with_offsets))
        scores = self.score_output(torch.tanh(self.score_projection(safe_frames))).squeeze(-1)
        weights = safe_masked_softmax(scores, patched_mask, dim=-1)
        pooled = (weights.unsqueeze(-1) * safe_frames).sum(dim=2)
        patch_mask = patched_mask.any(dim=-1)
        pooled = torch.where(patch_mask.unsqueeze(-1), pooled, _zero(pooled))
        return pooled, patch_mask


class BookQueryPool(nn.Module):
    """RMS-normalized learned-query pooling over one snapshot's eligible levels."""

    def __init__(self, d_level: int, *, rms_epsilon: float) -> None:
        super().__init__()
        if d_level <= 0:
            raise ValueError("d_level must be positive")
        self.d_level = d_level
        self.query = nn.Parameter(torch.empty(d_level))
        self.level_norm = RMSNorm(d_level, rms_epsilon)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        levels: Tensor,
        level_mask: Tensor,
        *,
        need_weights: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if levels.ndim != 3 or level_mask.ndim != 2:
            raise ValueError("levels and level_mask must have shapes [B,L,D] and [B,L]")
        if tuple(levels.shape[:2]) != tuple(level_mask.shape):
            raise ValueError("level_mask must match levels")
        if levels.shape[-1] != self.d_level or level_mask.dtype != torch.bool:
            raise ValueError("levels have the wrong width or level_mask is not boolean")

        safe_levels = torch.where(level_mask.unsqueeze(-1), levels, _zero(levels))
        normalized = self.level_norm(safe_levels)
        scores = torch.matmul(normalized, self.query.to(dtype=levels.dtype)) / math.sqrt(
            self.d_level
        )
        weights = safe_masked_softmax(scores, level_mask, dim=-1)
        pooled = (weights.unsqueeze(-1) * safe_levels).sum(dim=1)
        pooled = torch.where(level_mask.any(dim=1, keepdim=True), pooled, _zero(pooled))
        if need_weights:
            return pooled, weights
        return pooled
