"""Masking primitives with explicit all-masked behavior."""

from __future__ import annotations

import torch
from torch import Tensor


def _require_bool_mask(mask: Tensor, name: str = "mask") -> None:
    if not isinstance(mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name} must have dtype torch.bool, got {mask.dtype}")


def _zero_scalar_like(values: Tensor) -> Tensor:
    """Return a broadcastable scalar zero without materializing a full ONNX constant."""

    return torch.zeros((), dtype=values.dtype, device=values.device)


def masked_softmax(scores: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax over valid entries, returning exact zeros for all-masked rows.

    ``mask`` follows the usual PyTorch broadcasting rules, with ``True`` denoting
    a valid entry. The output always has the same shape as ``scores``.
    """

    if not isinstance(scores, Tensor):
        raise TypeError("scores must be a torch.Tensor")
    if not scores.is_floating_point():
        raise TypeError("scores must use a floating-point dtype")
    _require_bool_mask(mask)
    if mask.device != scores.device:
        raise ValueError("scores and mask must be on the same device")
    try:
        broadcast_shape = torch.broadcast_shapes(tuple(scores.shape), tuple(mask.shape))
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(scores.shape)}"
        ) from exc
    if broadcast_shape != tuple(scores.shape):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} would expand scores to {broadcast_shape}"
        )

    expanded_mask = mask.expand_as(scores)
    negative_large = torch.finfo(scores.dtype).min
    masked_scores = scores.masked_fill(~expanded_mask, negative_large)
    probabilities = torch.softmax(masked_scores, dim=dim)
    probabilities = torch.where(
        expanded_mask,
        probabilities,
        torch.zeros((), dtype=probabilities.dtype, device=probabilities.device),
    )
    denominator = probabilities.sum(dim=dim, keepdim=True)
    normalized = probabilities / denominator.clamp_min(torch.finfo(scores.dtype).tiny)
    return torch.where(
        denominator > 0,
        normalized,
        torch.zeros((), dtype=normalized.dtype, device=normalized.device),
    )


def causal_mask(
    query_length: int,
    key_length: int | None = None,
    *,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return a lower-triangular boolean attention mask."""

    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if key_length is None:
        key_length = query_length
    if key_length <= 0:
        raise ValueError("key_length must be positive")
    query_positions = torch.arange(query_length, device=device).unsqueeze(-1)
    key_positions = torch.arange(key_length, device=device).unsqueeze(0)
    return key_positions <= query_positions


def apply_token_mask(values: Tensor, valid_mask: Tensor) -> Tensor:
    """Zero invalid token rows without allowing masked NaNs to leak."""

    _require_bool_mask(valid_mask, "valid_mask")
    if values.ndim < 2:
        raise ValueError("values must have at least batch and token dimensions")
    if tuple(values.shape[:-1]) != tuple(valid_mask.shape):
        raise ValueError(
            "valid_mask must match every values dimension except the last: "
            f"{tuple(valid_mask.shape)} vs {tuple(values.shape)}"
        )
    if values.device != valid_mask.device:
        raise ValueError("values and valid_mask must be on the same device")
    return torch.where(valid_mask.unsqueeze(-1), values, _zero_scalar_like(values))


def last_valid_token(values: Tensor, valid_mask: Tensor) -> Tensor:
    """Gather the latest valid token, or exact zero when the sequence is empty."""

    _require_bool_mask(valid_mask, "valid_mask")
    if values.ndim != 3 or valid_mask.ndim != 2:
        raise ValueError("values and valid_mask must have shapes [B,T,D] and [B,T]")
    if tuple(values.shape[:2]) != tuple(valid_mask.shape):
        raise ValueError("valid_mask shape must match values batch/token dimensions")
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    indices = torch.where(valid_mask, positions, _zero_scalar_like(positions)).amax(dim=1)
    gathered = values.gather(
        1,
        indices[:, None, None].expand(-1, 1, values.shape[-1]),
    ).squeeze(1)
    return torch.where(
        valid_mask.any(dim=1, keepdim=True), gathered, _zero_scalar_like(gathered)
    )
