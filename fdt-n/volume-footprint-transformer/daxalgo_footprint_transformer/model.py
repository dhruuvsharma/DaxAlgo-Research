from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import BAR_FEATURE_NAMES, ROW_FEATURE_NAMES, TARGET_NAMES, EncodingConfig, ModelConfig


class RmsNorm(nn.Module):
    def __init__(self, dimensions: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimensions))
        self.epsilon = epsilon

    def forward(self, values: Tensor) -> Tensor:
        scale = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.epsilon)
        return values * scale * self.weight


class SwiGlu(nn.Module):
    def __init__(self, dimensions: int, hidden_dimensions: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(dimensions, hidden_dimensions * 2)
        self.output = nn.Linear(hidden_dimensions, dimensions)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor) -> Tensor:
        gate, content = self.input(values).chunk(2, dim=-1)
        return self.output(self.dropout(F.silu(gate) * content))


class PreNormCrossAttention(nn.Module):
    def __init__(self, dimensions: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = RmsNorm(dimensions)
        self.memory_norm = RmsNorm(dimensions)
        self.attention = nn.MultiheadAttention(dimensions, heads, dropout=dropout, batch_first=True)
        self.feed_norm = RmsNorm(dimensions)
        self.feed = SwiGlu(dimensions, dimensions * 2, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        normalized_query = self.query_norm(query)
        attended, _ = self.attention(
            normalized_query,
            self.memory_norm(memory),
            self.memory_norm(memory),
            key_padding_mask=~memory_mask,
            need_weights=False,
        )
        query = query + self.dropout(attended)
        return query + self.dropout(self.feed(self.feed_norm(query)))


class CausalTemporalBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dimensions = config.d_model
        self.norm = RmsNorm(dimensions)
        self.attention = nn.MultiheadAttention(
            dimensions,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.feed_norm = RmsNorm(dimensions)
        self.feed = SwiGlu(dimensions, dimensions * config.feedforward_multiplier, config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.layer_scale_attention = nn.Parameter(torch.full((dimensions,), 1e-2))
        self.layer_scale_feed = nn.Parameter(torch.full((dimensions,), 1e-2))

    def forward(self, values: Tensor, valid_mask: Tensor, causal_mask: Tensor) -> Tensor:
        normalized = self.norm(values)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=~valid_mask,
            need_weights=False,
        )
        values = values + self.dropout(attended) * self.layer_scale_attention
        values = values + self.dropout(self.feed(self.feed_norm(values))) * self.layer_scale_feed
        return values


class HorizonDecoderBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.cross = PreNormCrossAttention(config.d_model, config.attention_heads, config.dropout)
        self.self_norm = RmsNorm(config.d_model)
        self.self_attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, queries: Tensor, memory: Tensor, memory_mask: Tensor) -> Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + self.dropout(attended)
        return self.cross(queries, memory, memory_mask)


class FootprintDistributionTransformer(nn.Module):
    """Hierarchical full-footprint encoder with direct next-N quantile queries.

    Price-row tokens are pooled by learned auction queries (total, buy, sell, imbalance). The
    pooled footprint is gated with bar aggregates, passed through a causal temporal stack, and
    decoded by one learned query per requested horizon. No future prediction is fed back as input.
    """

    BOUNDED_TARGETS = (6,)

    def __init__(self, encoding: EncodingConfig, config: ModelConfig | None = None) -> None:
        super().__init__()
        encoding.validate()
        self.encoding = encoding
        self.config = config or ModelConfig()
        self.config.validate()
        dimensions = self.config.d_model

        self.row_projection = nn.Sequential(
            nn.Linear(len(ROW_FEATURE_NAMES), dimensions),
            nn.GELU(),
            nn.Linear(dimensions, dimensions),
            RmsNorm(dimensions),
        )
        self.auction_queries = nn.Parameter(
            torch.randn(self.config.row_queries, dimensions) / math.sqrt(dimensions)
        )
        self.row_pool = PreNormCrossAttention(
            dimensions, self.config.attention_heads, self.config.dropout
        )
        self.row_summary = nn.Sequential(
            nn.Linear(self.config.row_queries * dimensions, dimensions),
            nn.GELU(),
            RmsNorm(dimensions),
        )
        self.bar_projection = nn.Sequential(
            nn.Linear(len(BAR_FEATURE_NAMES), dimensions),
            nn.GELU(),
            nn.Linear(dimensions, dimensions),
            RmsNorm(dimensions),
        )
        self.fusion_gate = nn.Linear(dimensions * 2, dimensions)
        self.fusion_projection = nn.Linear(dimensions * 2, dimensions)
        self.time_embedding = nn.Parameter(
            torch.randn(encoding.lookback_bars, dimensions) / math.sqrt(dimensions)
        )
        self.temporal_blocks = nn.ModuleList(
            CausalTemporalBlock(self.config) for _ in range(self.config.temporal_layers)
        )
        self.temporal_norm = RmsNorm(dimensions)

        self.horizon_queries = nn.Parameter(
            torch.randn(encoding.horizon_bars, dimensions) / math.sqrt(dimensions)
        )
        self.horizon_embedding = nn.Embedding(encoding.horizon_bars, dimensions)
        self.decoder_blocks = nn.ModuleList(
            HorizonDecoderBlock(self.config) for _ in range(self.config.decoder_layers)
        )
        self.output_norm = RmsNorm(dimensions)
        self.output_head = nn.Linear(dimensions, len(TARGET_NAMES) * 3)

        causal = torch.triu(
            torch.ones(encoding.lookback_bars, encoding.lookback_bars, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

    def _ordered_quantiles(self, raw: Tensor) -> Tensor:
        raw = raw.view(raw.shape[0], raw.shape[1], len(TARGET_NAMES), 3)
        signed_median = raw[..., 0]
        lower_gap = F.softplus(raw[..., 1]) + 1e-4
        upper_gap = F.softplus(raw[..., 2]) + 1e-4
        median = signed_median
        lower = signed_median - lower_gap
        upper = signed_median + upper_gap
        quantiles = torch.stack((lower, median, upper), dim=-1)
        bounded_mask = torch.tensor(
            [index in self.BOUNDED_TARGETS for index in range(len(TARGET_NAMES))],
            dtype=torch.bool,
            device=raw.device,
        ).view(1, 1, -1, 1)
        quantiles = torch.where(bounded_mask, torch.tanh(quantiles), quantiles)

        # All five prices are direct marginals in the same anchor-relative coordinate. Min, max,
        # and clamp preserve monotonic q10<=q50<=q90 sequences while guaranteeing a physically
        # coherent candle at each reported quantile level.
        poc = quantiles[..., 0, :]
        low = torch.minimum(quantiles[..., 1, :], poc)
        high = torch.maximum(quantiles[..., 2, :], poc)
        buy_poc = torch.clamp(quantiles[..., 3, :], min=low, max=high)
        sell_poc = torch.clamp(quantiles[..., 4, :], min=low, max=high)
        return torch.stack(
            (poc, low, high, buy_poc, sell_poc, quantiles[..., 5, :], quantiles[..., 6, :]),
            dim=-2,
        )

    def forward(
        self,
        row_features: Tensor,
        row_mask: Tensor,
        bar_features: Tensor,
        bar_mask: Tensor,
    ) -> Tensor:
        batch, lookback, rows, row_dimensions = row_features.shape
        if lookback != self.encoding.lookback_bars:
            raise ValueError("unexpected lookback length")
        flat_rows = row_features.reshape(batch * lookback, rows, row_dimensions)
        flat_mask = row_mask.reshape(batch * lookback, rows).bool()
        row_memory = self.row_projection(flat_rows)
        queries = self.auction_queries.unsqueeze(0).expand(batch * lookback, -1, -1)
        pooled = self.row_pool(queries, row_memory, flat_mask)
        row_summary = self.row_summary(pooled.reshape(batch * lookback, -1)).reshape(
            batch, lookback, -1
        )
        bar_summary = self.bar_projection(bar_features)
        combined = torch.cat((row_summary, bar_summary), dim=-1)
        gate = torch.sigmoid(self.fusion_gate(combined))
        projected = self.fusion_projection(combined)
        temporal = gate * row_summary + (1.0 - gate) * bar_summary + projected
        temporal = temporal + self.time_embedding.unsqueeze(0)
        valid = bar_mask.bool()
        for block in self.temporal_blocks:
            temporal = block(temporal, valid, self.causal_mask)
        memory = self.temporal_norm(temporal)

        horizon_ids = torch.arange(
            self.encoding.horizon_bars, device=row_features.device, dtype=torch.long
        )
        horizon = self.horizon_queries + self.horizon_embedding(horizon_ids)
        horizon = horizon.unsqueeze(0).expand(batch, -1, -1)
        for block in self.decoder_blocks:
            horizon = block(horizon, memory, valid)
        raw = self.output_head(self.output_norm(horizon))
        return self._ordered_quantiles(raw)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
