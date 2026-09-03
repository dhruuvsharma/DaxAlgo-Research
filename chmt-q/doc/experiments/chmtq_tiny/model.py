"""Compact CHMT-Q model that preserves the published tensor contract."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def safe_masked_softmax(scores: Tensor, eligible: Tensor, dim: int = -1) -> Tensor:
    """Softmax that returns exact zeros for an entirely ineligible row."""
    eligible = eligible.to(torch.bool)
    has_key = eligible.any(dim=dim, keepdim=True)
    masked = scores.masked_fill(~eligible, -torch.inf)
    row_max = masked.amax(dim=dim, keepdim=True)
    row_max = torch.where(has_key, row_max, torch.zeros_like(row_max))
    numerator = torch.exp(masked - row_max)
    denominator = numerator.sum(dim=dim, keepdim=True)
    return torch.where(has_key, numerator / denominator.clamp_min(1e-12), torch.zeros_like(numerator))


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        scale = torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return x * scale * self.weight


class SwiGLU(nn.Module):
    def __init__(self, width: int, hidden: int) -> None:
        super().__init__()
        self.gate = nn.Linear(width, hidden)
        self.up = nn.Linear(width, hidden)
        self.down = nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class SafeMultiheadAttention(nn.Module):
    """Q/K-normalized attention with total behavior for empty mask rows."""

    def __init__(self, width: int, heads: int, dropout: float) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.log_temperature = nn.Parameter(torch.zeros(heads))
        self.dropout = nn.Dropout(dropout)

    def _heads(self, x: Tensor) -> Tensor:
        batch, steps, _ = x.shape
        return x.view(batch, steps, self.heads, self.head_width).transpose(1, 2)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        key_mask: Tensor,
        *,
        query_mask: Tensor | None = None,
        causal: bool = False,
        additive_bias: Tensor | None = None,
    ) -> Tensor:
        batch, query_steps, _ = query.shape
        key_steps = memory.shape[1]
        q = self._heads(self.q_proj(query))
        k = self._heads(self.k_proj(memory))
        v = self._heads(self.v_proj(memory))

        q = q * torch.rsqrt(q.square().mean(dim=-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().mean(dim=-1, keepdim=True) + 1e-6)
        temperature = self.log_temperature.exp().view(1, self.heads, 1, 1)
        scores = temperature * torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_width)
        if additive_bias is not None:
            scores = scores + additive_bias.unsqueeze(1)

        eligible = key_mask[:, None, None, :].expand(batch, self.heads, query_steps, key_steps)
        if causal:
            if query_steps != key_steps:
                raise ValueError("causal self-attention requires matching query and key lengths")
            triangle = torch.arange(key_steps, device=query.device)[None, :] <= torch.arange(
                query_steps, device=query.device
            )[:, None]
            eligible = eligible & triangle[None, None, :, :]

        weights = self.dropout(safe_masked_softmax(scores, eligible))
        output = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, query_steps, self.width)
        output = self.out_proj(output)
        if query_mask is not None:
            output = output * query_mask.unsqueeze(-1)
        return output


class EncoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(width)
        self.attention = SafeMultiheadAttention(width, heads, dropout)
        self.ff_norm = RMSNorm(width)
        self.ff = SwiGLU(width, ff_width)
        self.attention_scale = nn.Parameter(torch.full((width,), 0.1))
        self.ff_scale = nn.Parameter(torch.full((width,), 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor, *, causal: bool) -> Tensor:
        x = x + self.attention_scale * self.dropout(
            self.attention(
                self.attention_norm(x),
                self.attention_norm(x),
                mask,
                query_mask=mask,
                causal=causal,
            )
        )
        x = x * mask.unsqueeze(-1)
        x = x + self.ff_scale * self.dropout(self.ff(self.ff_norm(x)))
        return x * mask.unsqueeze(-1)


class PatchPool(nn.Module):
    def __init__(self, width: int, patch_size: int = 4) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.offset = nn.Parameter(torch.empty(patch_size, width))
        self.score_in = nn.Linear(width, width)
        self.score_out = nn.Linear(width, 1)
        nn.init.normal_(self.offset, std=0.02)

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        batch, steps, width = x.shape
        if steps % self.patch_size:
            raise ValueError("sequence length must be divisible by patch size")
        patches = steps // self.patch_size
        x = x.view(batch, patches, self.patch_size, width) + self.offset.view(
            1, 1, self.patch_size, width
        )
        mask = mask.view(batch, patches, self.patch_size)
        scores = self.score_out(torch.tanh(self.score_in(x))).squeeze(-1)
        weights = safe_masked_softmax(scores, mask)
        pooled = (weights.unsqueeze(-1) * x).sum(dim=2)
        pooled_mask = mask.any(dim=2)
        return pooled * pooled_mask.unsqueeze(-1), pooled_mask


class TimeEmbedding(nn.Module):
    def __init__(self, width: int, scales: tuple[float, ...], hidden: int = 16) -> None:
        super().__init__()
        self.register_buffer("scales", torch.tensor(scales, dtype=torch.float32), persistent=False)
        self.in_proj = nn.Linear(2 + 2 * len(scales), hidden)
        self.out_proj = nn.Linear(hidden, width)

    def forward(self, ages: Tensor) -> Tensor:
        maximum = ages.max().clamp_min(1.0)
        scaled = ages[:, None] / self.scales[None, :]
        features = torch.cat(
            [
                (ages / maximum)[:, None],
                (torch.log1p(ages) / torch.log1p(maximum))[:, None],
                torch.sin(scaled),
                torch.cos(scaled),
            ],
            dim=-1,
        )
        return self.out_proj(F.silu(self.in_proj(features)))


class DecoderBlock(nn.Module):
    def __init__(self, width: int, heads: int, ff_width: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = RMSNorm(width)
        self.cross_norm = RMSNorm(width)
        self.ff_norm = RMSNorm(width)
        self.self_attention = SafeMultiheadAttention(width, heads, dropout)
        self.cross_attention = SafeMultiheadAttention(width, heads, dropout)
        self.ff = SwiGLU(width, ff_width)
        self.self_scale = nn.Parameter(torch.full((width,), 0.1))
        self.cross_scale = nn.Parameter(torch.full((width,), 0.1))
        self.ff_scale = nn.Parameter(torch.full((width,), 0.1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, memory_mask: Tensor, memory_bias: Tensor) -> Tensor:
        query_mask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        normed = self.self_norm(queries)
        queries = queries + self.self_scale * self.dropout(
            self.self_attention(normed, normed, query_mask, query_mask=query_mask)
        )
        queries = queries + self.cross_scale * self.dropout(
            self.cross_attention(
                self.cross_norm(queries),
                memory,
                memory_mask,
                query_mask=query_mask,
                additive_bias=memory_bias,
            )
        )
        return queries + self.ff_scale * self.dropout(self.ff(self.ff_norm(queries)))


@dataclass(frozen=True)
class TinyConfig:
    width: int = 64
    heads: int = 4
    ff_width: int = 128
    level_width: int = 16
    level_heads: int = 4
    level_ff_width: int = 32
    encoder_layers: int = 1
    decoder_layers: int = 2
    gate_hidden: int = 32
    dropout: float = 0.10


class ChmtQTiny(nn.Module):
    """A small CHMT-Q preserving Bars/L1/L2/quality inputs and [B,5,17] output."""

    def __init__(self, config: TinyConfig = TinyConfig()) -> None:
        super().__init__()
        c = config
        self.config = c
        self.bar_projection = nn.Linear(16, c.width)
        self.l1_projection = nn.Linear(16, c.width)
        self.level_projection = nn.Linear(12, c.level_width)
        self.side_embedding = nn.Parameter(torch.empty(2, c.level_width))
        self.rank_embedding = nn.Parameter(torch.empty(10, c.level_width))
        self.modality_embedding = nn.Parameter(torch.empty(3, c.width))

        self.bar_time = TimeEmbedding(c.width, (60, 120, 240, 480, 960, 1920, 3840))
        self.micro_time = TimeEmbedding(c.width, (0.5, 1, 2, 4, 8, 16, 32, 64))
        self.l1_pool = PatchPool(c.width)
        self.l2_pool = PatchPool(c.width)
        self.level_block = EncoderBlock(c.level_width, c.level_heads, c.level_ff_width, c.dropout)
        self.level_pool_norm = RMSNorm(c.level_width)
        self.book_query = nn.Parameter(torch.empty(c.level_width))
        self.book_projection = nn.Linear(c.level_width, c.width)

        self.bar_blocks = nn.ModuleList(
            EncoderBlock(c.width, c.heads, c.ff_width, c.dropout) for _ in range(c.encoder_layers)
        )
        self.l1_blocks = nn.ModuleList(
            EncoderBlock(c.width, c.heads, c.ff_width, c.dropout) for _ in range(c.encoder_layers)
        )
        self.l2_blocks = nn.ModuleList(
            EncoderBlock(c.width, c.heads, c.ff_width, c.dropout) for _ in range(c.encoder_layers)
        )

        self.quality_in = nn.Linear(16, c.gate_hidden)
        self.quality_out = nn.Linear(c.gate_hidden, c.width)
        self.gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * c.width, c.gate_hidden),
                nn.SiLU(),
                nn.Linear(c.gate_hidden, 1),
            )
            for _ in range(3)
        )
        self.context_norm = RMSNorm(c.width)
        self.context_projection = nn.Linear(c.width, c.width)
        self.horizon_base = nn.Parameter(torch.empty(5, c.width))
        self.horizon_in = nn.Linear(3, 16)
        self.horizon_out = nn.Linear(16, c.width)
        self.decoder = nn.ModuleList(
            DecoderBlock(c.width, c.heads, c.ff_width, c.dropout) for _ in range(c.decoder_layers)
        )
        self.output_head = nn.Linear(c.width, 17)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.side_embedding, std=0.02)
        nn.init.normal_(self.rank_embedding, std=0.02)
        nn.init.normal_(self.modality_embedding, std=0.02)
        nn.init.normal_(self.book_query, std=0.02)
        nn.init.normal_(self.horizon_base, std=0.02)
        # The third horizon feature is expressed in seconds (60..300). Keep its
        # initial projection small so raw quantile centers begin near zero.
        nn.init.normal_(self.horizon_in.weight, std=0.001)
        nn.init.zeros_(self.horizon_in.bias)

    @staticmethod
    def _latest_valid(x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        available = mask.any(dim=1)
        indices = (mask.long() * torch.arange(mask.shape[1], device=x.device)[None, :]).argmax(dim=1)
        summary = x[torch.arange(x.shape[0], device=x.device), indices]
        return summary * available.unsqueeze(-1), available

    def _encode_bars(self, bars: Tensor, mask: Tensor) -> Tensor:
        ages = torch.arange(63, -1, -1, device=bars.device, dtype=bars.dtype) * 60.0
        x = self.bar_projection(bars) + self.bar_time(ages)[None] + self.modality_embedding[0]
        x = x * mask.unsqueeze(-1)
        for block in self.bar_blocks:
            x = block(x, mask, causal=True)
        return x

    def _encode_l1(self, l1: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        x = self.l1_projection(l1) * mask.unsqueeze(-1)
        x, mask = self.l1_pool(x, mask)
        ages = torch.arange(59, -1, -1, device=l1.device, dtype=l1.dtype) * 2.0
        x = x + self.micro_time(ages)[None] + self.modality_embedding[1]
        x = x * mask.unsqueeze(-1)
        for block in self.l1_blocks:
            x = block(x, mask, causal=True)
        return x, mask

    def _encode_l2(
        self, l2: Tensor, level_mask: Tensor, snapshot_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        batch, steps, _, _, _ = l2.shape
        level_mask = level_mask & snapshot_mask[:, :, None, None]
        side_rank = (
            self.side_embedding[:, None, :] + self.rank_embedding[None, :, :]
        ).reshape(20, self.config.level_width)
        x = self.level_projection(l2).reshape(batch * steps, 20, self.config.level_width)
        x = x + side_rank[None]
        flat_mask = level_mask.reshape(batch * steps, 20)
        x = self.level_block(x * flat_mask.unsqueeze(-1), flat_mask, causal=False)
        pool_scores = (self.level_pool_norm(x) * self.book_query).sum(dim=-1) / math.sqrt(
            self.config.level_width
        )
        pool_weights = safe_masked_softmax(pool_scores, flat_mask)
        books = (pool_weights.unsqueeze(-1) * x).sum(dim=1)
        books = self.book_projection(books).view(batch, steps, self.config.width)
        book_mask = level_mask.flatten(2).any(dim=-1) & snapshot_mask
        books = books * book_mask.unsqueeze(-1)
        books, mask = self.l2_pool(books, book_mask)
        ages = torch.arange(59, -1, -1, device=l2.device, dtype=l2.dtype) * 2.0
        books = books + self.micro_time(ages)[None] + self.modality_embedding[2]
        books = books * mask.unsqueeze(-1)
        for block in self.l2_blocks:
            books = block(books, mask, causal=True)
        return books, mask

    def forward(
        self,
        bars: Tensor,
        bars_mask: Tensor,
        l1: Tensor,
        l1_mask: Tensor,
        l2: Tensor,
        l2_level_mask: Tensor,
        l2_snapshot_mask: Tensor,
        quality: Tensor,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        bars_memory = self._encode_bars(bars, bars_mask)
        l1_memory, l1_patch_mask = self._encode_l1(l1, l1_mask)
        l2_memory, l2_patch_mask = self._encode_l2(l2, l2_level_mask, l2_snapshot_mask)
        memories = (bars_memory, l1_memory, l2_memory)
        masks = (bars_mask, l1_patch_mask, l2_patch_mask)

        summaries_and_availability = [self._latest_valid(x, mask) for x, mask in zip(memories, masks)]
        summaries = [item[0] for item in summaries_and_availability]
        availability = torch.stack([item[1] for item in summaries_and_availability], dim=-1)
        if not bool(availability.any(dim=-1).all()):
            raise ValueError("at least one modality must be available for every example")

        quality_context = self.quality_out(F.silu(self.quality_in(quality)))
        unnormalized = torch.stack(
            [
                availability[:, index]
                * torch.sigmoid(gate(torch.cat([summary, quality_context], dim=-1))).squeeze(-1)
                for index, (gate, summary) in enumerate(zip(self.gates, summaries))
            ],
            dim=-1,
        )
        gates = unnormalized / unnormalized.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        context = quality_context + sum(gates[:, i, None] * summaries[i] for i in range(3))
        context = self.context_norm(context)

        memory = torch.cat(memories, dim=1)
        memory_mask = torch.cat(masks, dim=1)
        modality_lengths = (64, 60, 60)
        memory_bias = torch.cat(
            [torch.log(gates[:, i : i + 1].clamp_min(1e-6)).expand(-1, length) for i, length in enumerate(modality_lengths)],
            dim=1,
        )[:, None, :]

        horizon = torch.arange(1, 6, device=bars.device, dtype=bars.dtype)
        horizon_features = torch.stack((horizon / 5.0, torch.log1p(horizon), horizon * 60.0), dim=-1)
        horizon_context = self.horizon_out(F.silu(self.horizon_in(horizon_features)))
        queries = self.horizon_base[None] + self.context_projection(context)[:, None] + horizon_context[None]
        auxiliary_raw: Tensor | None = None
        for index, block in enumerate(self.decoder):
            queries = block(queries, memory, memory_mask, memory_bias)
            if index == 0:
                auxiliary_raw = self.output_head(queries)
        raw = self.output_head(queries)
        if auxiliary_raw is None:
            auxiliary_raw = raw
        return raw, auxiliary_raw, {"gates": gates, "memory_mask": memory_mask}


def ordered_quantiles(raw: Tensor) -> Tensor:
    """Convert raw [B,5,17] output into [B,5,5,3] q10/q50/q90."""
    parameters = raw[..., :15].view(*raw.shape[:-1], 5, 3)
    center = parameters[..., 0]
    lower = center - F.softplus(parameters[..., 1])
    upper = center + F.softplus(parameters[..., 2])
    quantiles = torch.stack((lower, center, upper), dim=-1)
    bounded_delta = torch.tanh(quantiles[..., 4:5, :])
    return torch.cat((quantiles[..., :4, :], bounded_delta), dim=-2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
