"""Deterministic CHMT-Q supervision for wiring and optimizer smoke tests only."""

from __future__ import annotations

import torch

from ..config import ModelConfig as LegacyModelConfig
from ..synthetic import L2Mode, make_synthetic_batch
from .config import ChmtQConfig
from .contracts import ChmtQBatch


def make_chmt_q_synthetic_batch(
    config: ChmtQConfig,
    *,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
    l2_mode: L2Mode = "present",
) -> ChmtQBatch:
    """Reuse stable synthetic inputs with two-sided L2 present by default."""

    if not isinstance(config, ChmtQConfig):
        raise TypeError("config must be a ChmtQConfig")
    legacy = make_synthetic_batch(
        LegacyModelConfig.tiny(),
        batch_size=batch_size,
        seed=seed,
        device=device,
        l2_mode=l2_mode,
    )
    if legacy.targets is None:
        raise RuntimeError("legacy synthetic generator did not return targets")
    targets = legacy.targets.clone()
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    sample = torch.arange(batch_size, device=targets.device)[:, None]
    horizon = torch.arange(config.horizons, device=targets.device)[None, :]
    buy_present = (sample + horizon) % 5 != 0
    sell_present = (2 * sample + horizon) % 7 != 0
    target_mask[..., 1] = buy_present
    target_mask[..., 2] = sell_present
    targets = torch.where(target_mask, targets, torch.zeros_like(targets))
    presence_labels = torch.stack((buy_present, sell_present), dim=-1).to(torch.float32)
    presence_mask = torch.ones_like(presence_labels, dtype=torch.bool)

    return ChmtQBatch(
        bars=legacy.bars,
        bars_mask=legacy.bars_mask,
        l1=legacy.l1,
        l1_mask=legacy.l1_mask,
        l2=legacy.l2,
        l2_level_mask=legacy.l2_level_mask,
        l2_snapshot_mask=legacy.l2_snapshot_mask,
        quality=legacy.quality,
        targets=targets,
        target_mask=target_mask,
        presence_labels=presence_labels,
        presence_mask=presence_mask,
    ).validate(config, require_supervision=True)
