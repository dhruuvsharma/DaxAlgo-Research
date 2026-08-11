"""Hierarchical numeric market transformer."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from ..contracts import MarketBatch
from ..masks import _zero_scalar_like, apply_token_mask, last_valid_token
from .blocks import DecoderBlock, NumericTimeEncoding, PreNormTransformerBlock
from .pooling import CausalPatchPool, LearnedQueryPool


@dataclass(frozen=True, slots=True)
class EncodedMarket:
    """Inspectable encoder state used by focused mask/gate tests."""

    memory: Tensor
    memory_mask: Tensor
    context: Tensor
    gates: Tensor
    normalized_gates: Tensor
    memory_bias: Tensor


class NumericMarketTransformer(nn.Module):
    """Custom primitive transformer for the fixed eight-input market contract."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig.reference()
        cfg = self.config

        self.bar_projection = nn.Linear(cfg.bar_features, cfg.d_model)
        self.l1_projection = nn.Linear(cfg.l1_features, cfg.d_model)
        self.level_projection = nn.Linear(cfg.l2_features, cfg.d_level)

        self.modality_embedding = nn.Parameter(torch.empty(3, cfg.d_model))
        self.side_embedding = nn.Parameter(torch.empty(cfg.l2_sides, cfg.d_level))
        self.rank_embedding = nn.Parameter(torch.empty(cfg.l2_levels, cfg.d_level))

        bar_ages = torch.arange(cfg.bar_steps - 1, -1, -1, dtype=torch.float32) * 60.0
        micro_ages = (
            torch.arange(cfg.micro_steps - 1, -1, -1, dtype=torch.float32) * 0.5
        )
        patch_ages = micro_ages.reshape(cfg.micro_patch_steps, cfg.patch_size)[:, -1]
        self.register_buffer("bar_ages", bar_ages)
        self.register_buffer("micro_ages", micro_ages)
        self.register_buffer("patch_ages", patch_ages)
        self.register_buffer(
            "horizon_numbers", torch.arange(1, cfg.horizons + 1, dtype=torch.float32)
        )

        self.bar_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.bar_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(bar_ages[0]),
        )
        self.l1_frame_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.micro_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(micro_ages[0]),
        )
        self.level_time_encoding = NumericTimeEncoding(
            cfg.d_level,
            cfg.micro_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(micro_ages[0]),
        )
        self.l1_patch_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.micro_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(micro_ages[0]),
        )
        self.l2_patch_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.micro_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(micro_ages[0]),
        )

        self.level_blocks = nn.ModuleList(
            PreNormTransformerBlock(
                cfg.d_level,
                cfg.level_num_heads,
                cfg.level_ff_dim,
                cfg.dropout,
                causal=False,
            )
            for _ in range(cfg.l2_level_layers)
        )
        self.book_pool = LearnedQueryPool(cfg.d_level)
        self.depth_to_model = nn.Linear(cfg.d_level, cfg.d_model)
        self.l1_patch_pool = CausalPatchPool(
            cfg.d_model, hidden_dim=32, patch_width=cfg.patch_size
        )
        self.l2_patch_pool = CausalPatchPool(
            cfg.d_model, hidden_dim=32, patch_width=cfg.patch_size
        )

        self.bar_blocks = self._temporal_stack(
            cfg.bar_layers, cfg.bar_relative_time_scale
        )
        self.l1_blocks = self._temporal_stack(
            cfg.l1_layers, cfg.micro_relative_time_scale
        )
        self.l2_temporal_blocks = self._temporal_stack(
            cfg.l2_temporal_layers, cfg.micro_relative_time_scale
        )

        self.quality_mlp = nn.Sequential(
            nn.Linear(cfg.quality_features, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        )
        self.gate_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * cfg.d_model, cfg.gate_hidden_dim),
                nn.SiLU(),
                nn.Linear(cfg.gate_hidden_dim, 1),
            )
            for _ in range(3)
        )
        self.context_norm = nn.LayerNorm(cfg.d_model)

        self.horizon_queries = nn.Parameter(torch.empty(cfg.horizons, cfg.d_model))
        self.horizon_mlp = nn.Sequential(
            nn.Linear(3, cfg.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_hidden_dim, cfg.d_model),
        )
        self.context_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.decoder_blocks = nn.ModuleList(
            DecoderBlock(cfg.d_model, cfg.num_heads, cfg.ff_dim, cfg.dropout)
            for _ in range(cfg.decoder_layers)
        )

        self.price_head = nn.Linear(cfg.d_model, 3)
        self.volume_head = nn.Linear(cfg.d_model, 1)
        self.delta_head = nn.Linear(cfg.d_model, 1)
        self.reset_parameters()

    def _temporal_stack(self, layers: int, relative_time_scale: float) -> nn.ModuleList:
        cfg = self.config
        return nn.ModuleList(
            PreNormTransformerBlock(
                cfg.d_model,
                cfg.num_heads,
                cfg.ff_dim,
                cfg.dropout,
                causal=True,
                relative_time_scale=relative_time_scale,
            )
            for _ in range(layers)
        )

    def forward(
        self,
        bars: Tensor | MarketBatch,
        bars_mask: Tensor | None = None,
        l1: Tensor | None = None,
        l1_mask: Tensor | None = None,
        l2: Tensor | None = None,
        l2_level_mask: Tensor | None = None,
        l2_snapshot_mask: Tensor | None = None,
        quality: Tensor | None = None,
    ) -> Tensor:
        """Return forecasts ordered as POC/buy/sell/volume/delta, shape ``[B,5,5]``."""

        batch = self._coerce_batch(
            bars,
            bars_mask,
            l1,
            l1_mask,
            l2,
            l2_level_mask,
            l2_snapshot_mask,
            quality,
        )
        batch.require_inference_available()
        encoded = self._encode_validated_batch(batch)
        queries = self._make_horizon_queries(encoded.context)
        for block in self.decoder_blocks:
            queries = block(
                queries,
                encoded.memory,
                encoded.memory_mask,
                memory_bias=encoded.memory_bias,
            )

        forecast = torch.cat(
            (
                self.price_head(queries),
                self.volume_head(queries),
                torch.tanh(self.delta_head(queries)),
            ),
            dim=-1,
        )
        if not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()):
            if not bool(torch.isfinite(forecast).all().item()):
                raise FloatingPointError("numeric transformer produced NaN or infinity")
        return forecast

    def encode_modalities(self, batch: MarketBatch) -> EncodedMarket:
        """Validate and encode a batch, exposing hard-zero gates for diagnostics/tests."""

        batch.validate(self.config)
        return self._encode_validated_batch(batch)

    def modality_gates(self, batch: MarketBatch) -> Tensor:
        """Return unnormalized hard-availability gates in bars/L1/L2 order."""

        return self.encode_modalities(batch).gates

    def _encode_validated_batch(self, batch: MarketBatch) -> EncodedMarket:
        cfg = self.config
        batch_size = batch.batch_size

        safe_bars = torch.where(
            batch.bars_mask.unsqueeze(-1), batch.bars, _zero_scalar_like(batch.bars)
        )
        bars = (
            self.bar_projection(safe_bars)
            + self.bar_time_encoding(self.bar_ages)[None, :, :]
            + self.modality_embedding[0][None, None, :]
        )
        bars = apply_token_mask(bars, batch.bars_mask)

        safe_l1 = torch.where(
            batch.l1_mask.unsqueeze(-1), batch.l1, _zero_scalar_like(batch.l1)
        )
        l1_frames = (
            self.l1_projection(safe_l1)
            + self.l1_frame_time_encoding(self.micro_ages)[None, :, :]
            + self.modality_embedding[1][None, None, :]
        )
        l1_frames = apply_token_mask(l1_frames, batch.l1_mask)

        effective_level_mask = (
            batch.l2_level_mask
            & batch.l2_snapshot_mask[:, :, None, None]
        )
        safe_l2 = torch.where(
            effective_level_mask.unsqueeze(-1), batch.l2, _zero_scalar_like(batch.l2)
        )
        levels = self.level_projection(safe_l2)
        levels = (
            levels
            + self.side_embedding[None, None, :, None, :]
            + self.rank_embedding[None, None, None, :, :]
            + self.level_time_encoding(self.micro_ages)[None, :, None, None, :]
        )
        levels = torch.where(
            effective_level_mask.unsqueeze(-1), levels, _zero_scalar_like(levels)
        )
        flat_levels = levels.reshape(
            batch_size * cfg.micro_steps,
            cfg.l2_sides * cfg.l2_levels,
            cfg.d_level,
        )
        flat_level_mask = effective_level_mask.reshape(
            batch_size * cfg.micro_steps,
            cfg.l2_sides * cfg.l2_levels,
        )
        for block in self.level_blocks:
            flat_levels = block(flat_levels, flat_level_mask)
        pooled_levels = self.book_pool(flat_levels, flat_level_mask)
        depth_frames = self.depth_to_model(pooled_levels).reshape(
            batch_size, cfg.micro_steps, cfg.d_model
        )
        depth_frames = depth_frames + self.modality_embedding[2][None, None, :]
        depth_frames = apply_token_mask(depth_frames, batch.l2_snapshot_mask)

        l1_tokens, l1_patch_mask = self.l1_patch_pool(l1_frames, batch.l1_mask)
        l2_tokens, l2_patch_mask = self.l2_patch_pool(
            depth_frames, batch.l2_snapshot_mask
        )
        l1_tokens = apply_token_mask(
            l1_tokens + self.l1_patch_time_encoding(self.patch_ages)[None, :, :],
            l1_patch_mask,
        )
        l2_tokens = apply_token_mask(
            l2_tokens + self.l2_patch_time_encoding(self.patch_ages)[None, :, :],
            l2_patch_mask,
        )

        for block in self.bar_blocks:
            bars = block(bars, batch.bars_mask, ages=self.bar_ages)
        for block in self.l1_blocks:
            l1_tokens = block(l1_tokens, l1_patch_mask, ages=self.patch_ages)
        for block in self.l2_temporal_blocks:
            l2_tokens = block(l2_tokens, l2_patch_mask, ages=self.patch_ages)

        quality_context = self.quality_mlp(batch.quality)
        branch_values = (
            last_valid_token(bars, batch.bars_mask),
            last_valid_token(l1_tokens, l1_patch_mask),
            last_valid_token(l2_tokens, l2_patch_mask),
        )
        availability = torch.stack(
            (
                batch.bars_mask.any(dim=1),
                l1_patch_mask.any(dim=1),
                l2_patch_mask.any(dim=1),
            ),
            dim=1,
        )
        gates = torch.cat(
            tuple(
                torch.sigmoid(gate(torch.cat((summary, quality_context), dim=-1)))
                * availability[:, index : index + 1].to(dtype=quality_context.dtype)
                for index, (gate, summary) in enumerate(
                    zip(self.gate_mlps, branch_values, strict=True)
                )
            ),
            dim=1,
        )
        gate_total = gates.sum(dim=1, keepdim=True)
        normalized_gates = torch.where(
            gate_total > 0,
            gates / (gate_total + cfg.gate_epsilon),
            _zero_scalar_like(gates),
        )
        global_summary = sum(
            normalized_gates[:, index : index + 1] * summary
            for index, summary in enumerate(branch_values)
        )
        context = self.context_norm(quality_context + global_summary)

        memory = torch.cat((bars, l1_tokens, l2_tokens), dim=1)
        memory_mask = torch.cat(
            (batch.bars_mask, l1_patch_mask, l2_patch_mask), dim=1
        )
        if memory.shape[1] != cfg.memory_tokens:
            raise RuntimeError(
                f"internal memory contract expected {cfg.memory_tokens} tokens, "
                f"got {memory.shape[1]}"
            )
        token_gates = torch.cat(
            (
                normalized_gates[:, 0:1].expand(-1, cfg.bar_steps),
                normalized_gates[:, 1:2].expand(-1, cfg.micro_patch_steps),
                normalized_gates[:, 2:3].expand(-1, cfg.micro_patch_steps),
            ),
            dim=1,
        )
        memory_bias = torch.log(token_gates + cfg.gate_epsilon)[:, None, None, :]
        return EncodedMarket(
            memory=memory,
            memory_mask=memory_mask,
            context=context,
            gates=gates,
            normalized_gates=normalized_gates,
            memory_bias=memory_bias,
        )

    def _make_horizon_queries(self, context: Tensor) -> Tensor:
        cfg = self.config
        horizons = self.horizon_numbers
        horizon_features = torch.stack(
            (
                horizons / float(cfg.horizons),
                torch.log1p(horizons),
                horizons * 60.0,
            ),
            dim=-1,
        )
        encoded_horizons = self.horizon_mlp(horizon_features)
        return (
            self.horizon_queries[None, :, :]
            + encoded_horizons[None, :, :]
            + self.context_projection(context)[:, None, :]
        )

    def _coerce_batch(
        self,
        bars: Tensor | MarketBatch,
        bars_mask: Tensor | None,
        l1: Tensor | None,
        l1_mask: Tensor | None,
        l2: Tensor | None,
        l2_level_mask: Tensor | None,
        l2_snapshot_mask: Tensor | None,
        quality: Tensor | None,
    ) -> MarketBatch:
        if isinstance(bars, MarketBatch):
            extras = (
                bars_mask,
                l1,
                l1_mask,
                l2,
                l2_level_mask,
                l2_snapshot_mask,
                quality,
            )
            if any(value is not None for value in extras):
                raise ValueError("when bars is MarketBatch, no other model inputs may be supplied")
            return bars.validate(self.config)

        named = {
            "bars_mask": bars_mask,
            "l1": l1,
            "l1_mask": l1_mask,
            "l2": l2,
            "l2_level_mask": l2_level_mask,
            "l2_snapshot_mask": l2_snapshot_mask,
            "quality": quality,
        }
        missing = [name for name, value in named.items() if value is None]
        if missing:
            raise ValueError(f"missing model input(s): {', '.join(missing)}")
        return MarketBatch(
            bars=bars,
            bars_mask=bars_mask,  # type: ignore[arg-type]
            l1=l1,  # type: ignore[arg-type]
            l1_mask=l1_mask,  # type: ignore[arg-type]
            l2=l2,  # type: ignore[arg-type]
            l2_level_mask=l2_level_mask,  # type: ignore[arg-type]
            l2_snapshot_mask=l2_snapshot_mask,  # type: ignore[arg-type]
            quality=quality,  # type: ignore[arg-type]
        ).validate(self.config)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        """Return exact top-level parameter counts for run manifests/CLI output."""

        breakdown: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            component = name.split(".", maxsplit=1)[0]
            breakdown[component] = breakdown.get(component, 0) + parameter.numel()
        return dict(sorted(breakdown.items()))

    def reset_parameters(self) -> None:
        """Apply the architecture's deterministic-friendly initialization policy."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        learned_embeddings = (
            self.modality_embedding,
            self.side_embedding,
            self.rank_embedding,
            self.horizon_queries,
            self.book_pool.query,
        )
        for parameter in learned_embeddings:
            nn.init.trunc_normal_(parameter, std=0.02)

        for head in (self.price_head, self.volume_head, self.delta_head):
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)

        self._scale_block_residuals(self.level_blocks, self.config.l2_level_layers)
        self._scale_block_residuals(self.bar_blocks, self.config.bar_layers)
        self._scale_block_residuals(self.l1_blocks, self.config.l1_layers)
        self._scale_block_residuals(
            self.l2_temporal_blocks, self.config.l2_temporal_layers
        )
        decoder_gain = 1.0 / math.sqrt(2.0 * self.config.decoder_layers)
        with torch.no_grad():
            for block in self.decoder_blocks:
                block.self_attention.out_proj.weight.mul_(decoder_gain)
                block.cross_attention.out_proj.weight.mul_(decoder_gain)
                block.ffn.output_projection.weight.mul_(decoder_gain)

    @staticmethod
    def _scale_block_residuals(blocks: nn.ModuleList, depth: int) -> None:
        gain = 1.0 / math.sqrt(2.0 * depth)
        with torch.no_grad():
            for block in blocks:
                block.attention.out_proj.weight.mul_(gain)
                block.ffn.output_projection.weight.mul_(gain)
