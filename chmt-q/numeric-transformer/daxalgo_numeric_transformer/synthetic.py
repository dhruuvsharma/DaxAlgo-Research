"""Deterministic synthetic tensors for wiring and overfit tests only."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from .config import ModelConfig
from .contracts import MarketBatch

L2Mode = Literal["mixed", "present", "missing", "one_sided"]


def make_synthetic_batch(
    config: ModelConfig,
    batch_size: int,
    seed: int,
    device: torch.device | str = "cpu",
    *,
    l2_mode: L2Mode = "mixed",
) -> MarketBatch:
    """Build deterministic, learnable examples without real market data.

    ``mixed`` deliberately includes a wholly missing L2 sample and, when the
    batch has at least two rows, a one-sided sample. Targets are a low-noise
    numeric function of valid bar/L1/L2 summaries so a tiny model can overfit
    the fixed batch as a wiring test.
    """

    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if l2_mode not in ("mixed", "present", "missing", "one_sided"):
        raise ValueError(f"unsupported l2_mode: {l2_mode!r}")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    cfg = config
    latent = torch.randn(batch_size, 4, generator=generator, dtype=torch.float32)

    bar_time = torch.linspace(-1.0, 0.0, cfg.bar_steps, dtype=torch.float32)
    bars = 0.04 * torch.randn(
        batch_size,
        cfg.bar_steps,
        cfg.bar_features,
        generator=generator,
        dtype=torch.float32,
    )
    trend = latent[:, 0:1, None]
    volatility = latent[:, 1:2, None].abs() + 0.3
    bars[:, :, 0:4] += trend * (bar_time[None, :, None] + 1.2)
    bars[:, :, 4] += 0.15 * latent[:, 0:1] * bar_time[None, :]
    bars[:, :, 5] = volatility.squeeze(-1) + bars[:, :, 5].abs()
    bars[:, :, 9] = torch.sigmoid(bars[:, :, 9] + latent[:, 0:1])
    bars[:, :, 10] += 0.6 * latent[:, 2:3] + 1.0
    bars[:, :, 11] += 0.25 * latent[:, 2:3]
    bars[:, :, 12] = torch.linspace(63.0, 0.0, cfg.bar_steps)[None, :]
    bars[:, :, 13] = torch.sigmoid(bars[:, :, 13])
    session_phase = torch.linspace(0.0, 1.0, cfg.bar_steps)
    bars[:, :, 14] = torch.sin(2.0 * torch.pi * session_phase)[None, :]
    bars[:, :, 15] = torch.cos(2.0 * torch.pi * session_phase)[None, :]
    bars_mask = torch.rand(
        batch_size, cfg.bar_steps, generator=generator
    ) > 0.08
    bars_mask[:, -1] = True
    bars = torch.where(bars_mask.unsqueeze(-1), bars, torch.zeros_like(bars))

    micro_time = torch.linspace(-1.0, 0.0, cfg.micro_steps, dtype=torch.float32)
    l1 = 0.05 * torch.randn(
        batch_size,
        cfg.micro_steps,
        cfg.l1_features,
        generator=generator,
        dtype=torch.float32,
    )
    flow = latent[:, 2:3]
    l1[:, :, 0] += latent[:, 0:1] * (micro_time[None, :] + 1.0)
    l1[:, :, 1] += 0.25 * latent[:, 0:1]
    l1[:, :, 2] = l1[:, :, 2].abs() + 0.5
    l1[:, :, 3] += 0.5 * flow + 1.0
    l1[:, :, 4] -= 0.5 * flow - 1.0
    l1[:, :, 5] = torch.tanh(flow + l1[:, :, 5])
    l1[:, :, 6] += 0.2 * flow
    l1[:, :, 11] = torch.linspace(1.5, 0.0, cfg.micro_steps)[None, :]
    l1[:, :, 13] = torch.sigmoid(l1[:, :, 13])
    l1[:, :, 14] = (
        torch.rand(batch_size, cfg.micro_steps, generator=generator) < 0.12
    ).to(torch.float32)
    l1[:, :, 15] = (
        torch.rand(batch_size, cfg.micro_steps, generator=generator) < 0.04
    ).to(torch.float32)
    l1_mask = torch.rand(
        batch_size, cfg.micro_steps, generator=generator
    ) > 0.06
    l1_mask[:, -1] = True
    l1 = torch.where(l1_mask.unsqueeze(-1), l1, torch.zeros_like(l1))

    l2 = 0.04 * torch.randn(
        batch_size,
        cfg.micro_steps,
        cfg.l2_sides,
        cfg.l2_levels,
        cfg.l2_features,
        generator=generator,
        dtype=torch.float32,
    )
    ranks = torch.arange(cfg.l2_levels, dtype=torch.float32)
    bid_distance = -(ranks + 1.0)
    ask_distance = ranks + 1.0
    l2[:, :, 0, :, 0] += bid_distance[None, None, :]
    l2[:, :, 1, :, 0] += ask_distance[None, None, :]
    l2[:, :, :, :, 1] += 1.0
    side_flow = latent[:, 2:3, None, None]
    l2[:, :, 0, :, 2] += 1.0 + 0.35 * side_flow.squeeze(-1)
    l2[:, :, 1, :, 2] += 1.0 - 0.35 * side_flow.squeeze(-1)
    cumulative = torch.log1p(torch.arange(1, cfg.l2_levels + 1, dtype=torch.float32))
    l2[:, :, :, :, 3] += cumulative[None, None, None, :]
    l2[:, :, :, :, 4:6] = torch.sigmoid(l2[:, :, :, :, 4:6])
    l2[:, :, :, :, 6] = torch.tanh(l2[:, :, :, :, 6] + side_flow)
    l2[:, :, :, :, 7] += torch.linspace(0.9, 0.0, cfg.micro_steps)[
        None, :, None, None
    ]
    l2[:, :, :, :, 9] = torch.sigmoid(l2[:, :, :, :, 9])
    l2[:, :, :, :, 10] = (
        torch.rand(
            batch_size,
            cfg.micro_steps,
            cfg.l2_sides,
            cfg.l2_levels,
            generator=generator,
        )
        < 0.10
    ).to(torch.float32)
    l2[:, :, :, :, 11] = (
        torch.rand(
            batch_size,
            cfg.micro_steps,
            cfg.l2_sides,
            cfg.l2_levels,
            generator=generator,
        )
        < 0.03
    ).to(torch.float32)
    l2_level_mask = torch.rand(
        batch_size,
        cfg.micro_steps,
        cfg.l2_sides,
        cfg.l2_levels,
        generator=generator,
    ) > 0.08
    rejected_snapshots = torch.rand(
        batch_size, cfg.micro_steps, generator=generator
    ) < 0.10
    l2_level_mask &= ~rejected_snapshots[:, :, None, None]

    if l2_mode == "missing":
        l2_level_mask.zero_()
    elif l2_mode == "one_sided":
        l2_level_mask[:, :, 1, :].zero_()
    elif l2_mode == "mixed":
        l2_level_mask[0].zero_()
        if batch_size > 1:
            l2_level_mask[1, :, 1, :].zero_()

    l2_snapshot_mask = l2_level_mask.any(dim=(-1, -2))
    l2 = torch.where(l2_level_mask.unsqueeze(-1), l2, torch.zeros_like(l2))

    quality = torch.zeros(batch_size, cfg.quality_features, dtype=torch.float32)
    quality[:, 0] = bars_mask.to(torch.float32).mean(dim=1)
    quality[:, 1] = l1_mask.to(torch.float32).mean(dim=1)
    quality[:, 2] = l2_snapshot_mask.to(torch.float32).mean(dim=1)
    quality[:, 3] = l2_level_mask.to(torch.float32).mean(dim=(1, 2, 3))
    quality[:, 4] = (l1[:, :, 14] * l1_mask).sum(dim=1) / l1_mask.sum(dim=1).clamp_min(1)
    level_carried = l2[:, :, :, :, 10] * l2_level_mask
    quality[:, 5] = level_carried.sum(dim=(1, 2, 3)) / l2_level_mask.sum(
        dim=(1, 2, 3)
    ).clamp_min(1)
    quality[:, 6] = (l1[:, :, 15] * l1_mask).sum(dim=1) / l1_mask.sum(dim=1).clamp_min(1)
    quality[:, 7] = (
        (l2[:, :, :, :, 11] * l2_level_mask).sum(dim=(1, 2, 3))
        / l2_level_mask.sum(dim=(1, 2, 3)).clamp_min(1)
    )
    quality[:, 8:12] = torch.sigmoid(latent[:, :4]) * 0.1
    bid_present = l2_level_mask[:, :, 0, :].any(dim=-1)
    ask_present = l2_level_mask[:, :, 1, :].any(dim=-1)
    quality[:, 12] = (bid_present ^ ask_present).to(torch.float32).mean(dim=1)
    quality[:, 13] = rejected_snapshots.to(torch.float32).mean(dim=1)
    quality[:, 14] = 1.0
    quality[:, 15] = torch.sigmoid(latent[:, 3])

    bar_signal = _masked_feature_mean(bars, bars_mask, 3)
    volume_signal = _masked_feature_mean(bars, bars_mask, 11)
    l1_signal = _masked_feature_mean(l1, l1_mask, 5)
    l2_signal = _masked_level_feature_mean(l2, l2_level_mask, 6)
    depth_size = _masked_level_feature_mean(l2, l2_level_mask, 2)
    horizons = torch.arange(1, cfg.horizons + 1, dtype=torch.float32)[None, :]
    base_price = (
        0.55 * bar_signal[:, None]
        + 0.35 * l1_signal[:, None]
        + 0.20 * l2_signal[:, None]
    ) * (1.0 + 0.08 * horizons)
    poc = base_price
    buy_poc = base_price + 0.15 + 0.03 * horizons + 0.08 * l1_signal[:, None]
    sell_poc = base_price - 0.15 - 0.02 * horizons - 0.08 * l1_signal[:, None]
    log_volume = (
        0.45 * volume_signal[:, None]
        + 0.20 * depth_size[:, None]
        + 0.04 * horizons
    )
    delta = torch.tanh(
        0.65 * l1_signal[:, None]
        + 0.45 * l2_signal[:, None]
        - 0.025 * horizons
    )
    targets = torch.stack((poc, buy_poc, sell_poc, log_volume, delta), dim=-1)
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    sample_ids = torch.arange(batch_size)[:, None]
    horizon_ids = torch.arange(cfg.horizons)[None, :]
    target_mask[:, :, 1] = (sample_ids + horizon_ids) % 7 != 0
    target_mask[:, :, 2] = (2 * sample_ids + horizon_ids) % 11 != 0
    targets = torch.where(target_mask, targets, torch.zeros_like(targets))

    batch = MarketBatch(
        bars=bars,
        bars_mask=bars_mask,
        l1=l1,
        l1_mask=l1_mask,
        l2=l2,
        l2_level_mask=l2_level_mask,
        l2_snapshot_mask=l2_snapshot_mask,
        quality=quality,
        targets=targets,
        target_mask=target_mask,
    )
    return batch.to(device).validate(cfg, require_targets=True)


def _masked_feature_mean(values: Tensor, mask: Tensor, feature: int) -> Tensor:
    selected = torch.where(mask, values[:, :, feature], torch.zeros_like(values[:, :, feature]))
    return selected.sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def _masked_level_feature_mean(values: Tensor, mask: Tensor, feature: int) -> Tensor:
    selected = torch.where(
        mask, values[:, :, :, :, feature], torch.zeros_like(values[:, :, :, :, feature])
    )
    return selected.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1)


class SyntheticBatchGenerator:
    """Small deterministic seed sequence for repeatable smoke experiments."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config
        self.seed = seed
        self.device = device
        self._index = 0

    def make_batch(self, batch_size: int, *, l2_mode: L2Mode = "mixed") -> MarketBatch:
        batch = make_synthetic_batch(
            self.config,
            batch_size,
            self.seed + self._index,
            self.device,
            l2_mode=l2_mode,
        )
        self._index += 1
        return batch
