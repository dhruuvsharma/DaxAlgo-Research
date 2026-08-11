"""Causal hierarchical market Transformer with quantile-head raw outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..config import ChmtQConfig
from ..contracts import ChmtQBatch
from ..output import ChmtQOutput, transform_raw_output
from .attention import (
    DEFAULT_LOGIT_RUNAWAY_THRESHOLD,
    AttentionLogitTelemetry,
    ChmtQAttention,
    _zero,
)
from .blocks import (
    ChmtQDecoderBlock,
    ChmtQEncoderBlock,
    NumericTimeEncoding,
    RelativeTimeBias,
    RMSNorm,
    apply_token_mask,
)
from .pooling import BookQueryPool, MaskedPatchPool


@dataclass(frozen=True, slots=True)
class EncodedMarket:
    """Inspectable fused state before horizon decoding."""

    memory: Tensor
    memory_mask: Tensor
    context: Tensor
    gates: Tensor
    normalized_gates: Tensor
    memory_bias: Tensor


@dataclass(frozen=True, slots=True)
class ChmtQRawOutputs:
    """Final and block-one raw parameters from the shared 17-value head."""

    final: Tensor
    auxiliary: Tensor


@dataclass(frozen=True, slots=True)
class ChmtQTrainingOutput:
    """Raw and transformed final/auxiliary outputs for the training objective."""

    raw: ChmtQRawOutputs
    final: ChmtQOutput
    auxiliary: ChmtQOutput


def _is_exporting() -> bool:
    return torch.jit.is_tracing() or torch.onnx.is_in_onnx_export()


def _last_valid_token(values: Tensor, valid_mask: Tensor) -> Tensor:
    positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
    indices = torch.where(valid_mask, positions, torch.zeros_like(positions)).amax(dim=1)
    gathered = values.gather(1, indices[:, None, None].expand(-1, 1, values.shape[-1])).squeeze(1)
    return torch.where(valid_mask.any(dim=1, keepdim=True), gathered, _zero(gathered))


class ChmtQTransformer(nn.Module):
    """Selected CHMT-Q v1 architecture over the fixed eight-input tensor contract."""

    def __init__(self, config: ChmtQConfig | None = None) -> None:
        super().__init__()
        self.config = config or ChmtQConfig.reference()
        cfg = self.config

        self.bar_projection = nn.Linear(cfg.bar_features, cfg.d_model)
        self.l1_projection = nn.Linear(cfg.l1_features, cfg.d_model)
        self.level_projection = nn.Linear(cfg.l2_features, cfg.d_level)

        self.modality_embedding = nn.Parameter(torch.empty(3, cfg.d_model))
        self.side_embedding = nn.Parameter(torch.empty(cfg.l2_sides, cfg.d_level))
        self.rank_embedding = nn.Parameter(torch.empty(cfg.l2_levels, cfg.d_level))

        bar_ages = torch.tensor(cfg.bar_ages_seconds, dtype=torch.float32)
        patch_ages = torch.tensor(cfg.patch_ages_seconds, dtype=torch.float32)
        horizon_numbers = torch.arange(1, cfg.horizons + 1, dtype=torch.float32)
        self.register_buffer("bar_ages", bar_ages)
        self.register_buffer("patch_ages", patch_ages)
        self.register_buffer("horizon_numbers", horizon_numbers)

        self.bar_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.bar_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(bar_ages[0]),
        )
        self.l1_patch_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.micro_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(patch_ages[0]),
        )
        self.l2_patch_time_encoding = NumericTimeEncoding(
            cfg.d_model,
            cfg.micro_time_scales,
            hidden_dim=cfg.time_hidden_dim,
            context_max=float(patch_ages[0]),
        )

        self.level_blocks = nn.ModuleList(
            ChmtQEncoderBlock(
                cfg.d_level,
                cfg.level_num_heads,
                cfg.level_ff_dim,
                cfg.dropout,
                rms_epsilon=cfg.rms_epsilon,
                qk_epsilon=cfg.qk_epsilon,
                layer_scale_initial=cfg.layer_scale_initial,
                causal=False,
            )
            for _ in range(cfg.l2_level_layers)
        )
        self.book_pool = BookQueryPool(cfg.d_level, rms_epsilon=cfg.rms_epsilon)
        self.depth_to_model = nn.Linear(cfg.d_level, cfg.d_model)
        self.l1_patch_pool = MaskedPatchPool(cfg.d_model, hidden_dim=32, patch_size=cfg.patch_size)
        self.l2_patch_pool = MaskedPatchPool(cfg.d_model, hidden_dim=32, patch_size=cfg.patch_size)

        self.bar_blocks = self._temporal_stack(cfg.bar_layers, cfg.bar_relative_time_scale)
        self.l1_blocks = self._temporal_stack(cfg.l1_layers, cfg.micro_relative_time_scale)
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
        self.context_norm = RMSNorm(cfg.d_model, cfg.rms_epsilon)

        self.horizon_queries = nn.Parameter(torch.empty(cfg.horizons, cfg.d_model))
        self.horizon_mlp = nn.Sequential(
            nn.Linear(3, cfg.time_hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.time_hidden_dim, cfg.d_model),
        )
        self.context_projection = nn.Linear(cfg.d_model, cfg.d_model)
        self.decoder_blocks = nn.ModuleList(
            ChmtQDecoderBlock(
                cfg.d_model,
                cfg.num_heads,
                cfg.ff_dim,
                cfg.dropout,
                rms_epsilon=cfg.rms_epsilon,
                qk_epsilon=cfg.qk_epsilon,
                layer_scale_initial=cfg.layer_scale_initial,
            )
            for _ in range(cfg.decoder_layers)
        )
        self.output_head = nn.Linear(cfg.d_model, cfg.raw_output_features)
        self.reset_parameters()

    def _temporal_stack(self, layers: int, time_scale: float) -> nn.ModuleList:
        cfg = self.config
        return nn.ModuleList(
            ChmtQEncoderBlock(
                cfg.d_model,
                cfg.num_heads,
                cfg.ff_dim,
                cfg.dropout,
                rms_epsilon=cfg.rms_epsilon,
                qk_epsilon=cfg.qk_epsilon,
                layer_scale_initial=cfg.layer_scale_initial,
                causal=True,
                relative_time_scale=time_scale,
            )
            for _ in range(layers)
        )

    def forward(
        self,
        bars: Tensor | ChmtQBatch,
        bars_mask: Tensor | None = None,
        l1: Tensor | None = None,
        l1_mask: Tensor | None = None,
        l2: Tensor | None = None,
        l2_level_mask: Tensor | None = None,
        l2_snapshot_mask: Tensor | None = None,
        quality: Tensor | None = None,
    ) -> Tensor:
        """Return only final raw parameters ``[B,5,17]`` for export/inference."""

        return self.forward_raw(
            bars,
            bars_mask,
            l1,
            l1_mask,
            l2,
            l2_level_mask,
            l2_snapshot_mask,
            quality,
        )

    def forward_raw(
        self,
        bars: Tensor | ChmtQBatch,
        bars_mask: Tensor | None = None,
        l1: Tensor | None = None,
        l1_mask: Tensor | None = None,
        l2: Tensor | None = None,
        l2_level_mask: Tensor | None = None,
        l2_snapshot_mask: Tensor | None = None,
        quality: Tensor | None = None,
    ) -> Tensor:
        """Return final raw center/gap parameters and presence logits only."""

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
        encoded = self._encode_validated_batch(batch)
        final, _ = self._decode_raw(encoded, capture_auxiliary=False)
        return final

    def forward_distribution(
        self,
        bars: Tensor | ChmtQBatch,
        bars_mask: Tensor | None = None,
        l1: Tensor | None = None,
        l1_mask: Tensor | None = None,
        l2: Tensor | None = None,
        l2_level_mask: Tensor | None = None,
        l2_snapshot_mask: Tensor | None = None,
        quality: Tensor | None = None,
    ) -> ChmtQOutput:
        """Return the ordered final distribution for training or host logic."""

        raw = self.forward_raw(
            bars,
            bars_mask,
            l1,
            l1_mask,
            l2,
            l2_level_mask,
            l2_snapshot_mask,
            quality,
        )
        return transform_raw_output(raw, self.config)

    def forward_training(self, batch: ChmtQBatch) -> ChmtQTrainingOutput:
        """Return final and decoder-block-one outputs through the shared head."""

        validated = self._coerce_batch(batch, None, None, None, None, None, None, None)
        raw = self._forward_raw_outputs(validated)
        return ChmtQTrainingOutput(
            raw=raw,
            final=transform_raw_output(raw.final, self.config),
            auxiliary=transform_raw_output(raw.auxiliary, self.config),
        )

    def encode_modalities(self, batch: ChmtQBatch) -> EncodedMarket:
        """Expose fused memory/context for mask and gate diagnostics."""

        validated = self._coerce_batch(batch, None, None, None, None, None, None, None)
        return self._encode_validated_batch(validated)

    def modality_gates(self, batch: ChmtQBatch) -> Tensor:
        return self.encode_modalities(batch).gates

    def enable_attention_logit_telemetry(
        self,
        *,
        runaway_threshold: float = DEFAULT_LOGIT_RUNAWAY_THRESHOLD,
    ) -> None:
        """Enable a fresh per-layer/per-head pre-softmax observation window."""

        for _name, attention in self._named_attention_modules():
            attention.enable_logit_telemetry(runaway_threshold=runaway_threshold)

    def disable_attention_logit_telemetry(self, *, reset: bool = False) -> None:
        """Disable attention monitoring without changing model parameters."""

        for _name, attention in self._named_attention_modules():
            attention.disable_logit_telemetry(reset=reset)

    def reset_attention_logit_telemetry(self) -> None:
        """Clear observations while preserving each layer's enabled state."""

        for _name, attention in self._named_attention_modules():
            attention.reset_logit_telemetry()

    def collect_attention_logit_telemetry(
        self,
        *,
        reset: bool = False,
    ) -> dict[str, AttentionLogitTelemetry]:
        """Collect observations keyed by stable ``named_modules`` paths."""

        collected = {
            name: telemetry
            for name, attention in self._named_attention_modules()
            if (telemetry := attention.collect_logit_telemetry()) is not None
        }
        if reset:
            self.reset_attention_logit_telemetry()
        return collected

    def _named_attention_modules(self) -> tuple[tuple[str, ChmtQAttention], ...]:
        return tuple(
            (name, module)
            for name, module in self.named_modules()
            if isinstance(module, ChmtQAttention)
        )

    def _forward_raw_outputs(self, batch: ChmtQBatch) -> ChmtQRawOutputs:
        encoded = self._encode_validated_batch(batch)
        final, auxiliary = self._decode_raw(encoded, capture_auxiliary=True)
        if auxiliary is None:
            raise AssertionError("training decode did not produce its block-one output")
        return ChmtQRawOutputs(final=final, auxiliary=auxiliary)

    def _decode_raw(
        self, encoded: EncodedMarket, *, capture_auxiliary: bool
    ) -> tuple[Tensor, Tensor | None]:
        queries = self._make_horizon_queries(encoded.context)
        auxiliary: Tensor | None = None
        for index, block in enumerate(self.decoder_blocks):
            queries = block(
                queries,
                encoded.memory,
                encoded.memory_mask,
                memory_bias=encoded.memory_bias,
            )
            if capture_auxiliary and index == 0:
                auxiliary = self.output_head(queries)
        if not self.decoder_blocks:
            raise RuntimeError("CHMT-Q requires at least one decoder block")
        final = self.output_head(queries)
        checked = (final,) if auxiliary is None else (final, auxiliary)
        if not _is_exporting() and any(
            not bool(torch.isfinite(value).all().item()) for value in checked
        ):
            raise FloatingPointError("CHMT-Q produced NaN or infinity")
        return final, auxiliary

    def _encode_validated_batch(self, batch: ChmtQBatch) -> EncodedMarket:
        cfg = self.config
        batch_size = batch.batch_size

        safe_bars = torch.where(batch.bars_mask.unsqueeze(-1), batch.bars, _zero(batch.bars))
        bars = (
            self.bar_projection(safe_bars)
            + self.bar_time_encoding(self.bar_ages)[None, :, :]
            + self.modality_embedding[0][None, None, :]
        )
        bars = apply_token_mask(bars, batch.bars_mask)

        safe_l1 = torch.where(batch.l1_mask.unsqueeze(-1), batch.l1, _zero(batch.l1))
        l1_frames = apply_token_mask(self.l1_projection(safe_l1), batch.l1_mask)
        l1_tokens, l1_patch_mask = self.l1_patch_pool(l1_frames, batch.l1_mask)
        l1_tokens = apply_token_mask(
            l1_tokens
            + self.l1_patch_time_encoding(self.patch_ages)[None, :, :]
            + self.modality_embedding[1][None, None, :],
            l1_patch_mask,
        )

        effective_level_mask = batch.l2_level_mask & batch.l2_snapshot_mask[:, :, None, None]
        safe_l2 = torch.where(effective_level_mask.unsqueeze(-1), batch.l2, _zero(batch.l2))
        levels = (
            self.level_projection(safe_l2)
            + self.side_embedding[None, None, :, None, :]
            + self.rank_embedding[None, None, None, :, :]
        )
        levels = torch.where(effective_level_mask.unsqueeze(-1), levels, _zero(levels))
        flat_levels = levels.reshape(
            batch_size * cfg.micro_steps,
            cfg.l2_sides * cfg.l2_levels,
            cfg.d_level,
        )
        flat_level_mask = effective_level_mask.reshape(
            batch_size * cfg.micro_steps, cfg.l2_sides * cfg.l2_levels
        )
        for block in self.level_blocks:
            flat_levels = block(flat_levels, flat_level_mask)
        pooled_levels = self.book_pool(flat_levels, flat_level_mask)
        if isinstance(pooled_levels, tuple):
            raise AssertionError("book pool unexpectedly returned weights")
        depth_frames = self.depth_to_model(pooled_levels).reshape(
            batch_size, cfg.micro_steps, cfg.d_model
        )
        depth_frames = apply_token_mask(depth_frames, batch.l2_snapshot_mask)
        l2_tokens, l2_patch_mask = self.l2_patch_pool(depth_frames, batch.l2_snapshot_mask)
        l2_tokens = apply_token_mask(
            l2_tokens
            + self.l2_patch_time_encoding(self.patch_ages)[None, :, :]
            + self.modality_embedding[2][None, None, :],
            l2_patch_mask,
        )

        for block in self.bar_blocks:
            bars = block(bars, batch.bars_mask, ages=self.bar_ages)
        for block in self.l1_blocks:
            l1_tokens = block(l1_tokens, l1_patch_mask, ages=self.patch_ages)
        for block in self.l2_temporal_blocks:
            l2_tokens = block(l2_tokens, l2_patch_mask, ages=self.patch_ages)

        quality_context = self.quality_mlp(batch.quality)
        summaries = (
            _last_valid_token(bars, batch.bars_mask),
            _last_valid_token(l1_tokens, l1_patch_mask),
            _last_valid_token(l2_tokens, l2_patch_mask),
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
                for index, (gate, summary) in enumerate(zip(self.gate_mlps, summaries, strict=True))
            ),
            dim=1,
        )
        gate_total = gates.sum(dim=1, keepdim=True)
        normalized_gates = torch.where(
            gate_total > 0.0,
            gates / (gate_total + cfg.gate_epsilon),
            _zero(gates),
        )
        weighted_summary = sum(
            normalized_gates[:, index : index + 1] * summary
            for index, summary in enumerate(summaries)
        )
        context = self.context_norm(quality_context + weighted_summary)

        memory = torch.cat((bars, l1_tokens, l2_tokens), dim=1)
        memory_mask = torch.cat((batch.bars_mask, l1_patch_mask, l2_patch_mask), dim=1)
        if memory.shape[1] != cfg.memory_tokens:
            raise RuntimeError("fused memory token count violated the fixed contract")
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
        horizons = self.horizon_numbers
        horizon_features = torch.stack(
            (
                horizons / float(self.config.horizons),
                torch.log1p(horizons),
                horizons * self.config.forecast_interval_seconds,
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
        bars: Tensor | ChmtQBatch,
        bars_mask: Tensor | None,
        l1: Tensor | None,
        l1_mask: Tensor | None,
        l2: Tensor | None,
        l2_level_mask: Tensor | None,
        l2_snapshot_mask: Tensor | None,
        quality: Tensor | None,
    ) -> ChmtQBatch:
        if isinstance(bars, ChmtQBatch):
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
                raise ValueError("no separate inputs may accompany a ChmtQBatch")
            return bars if _is_exporting() else bars.validate(self.config)

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
        batch = ChmtQBatch(
            bars=bars,
            bars_mask=bars_mask,  # type: ignore[arg-type]
            l1=l1,  # type: ignore[arg-type]
            l1_mask=l1_mask,  # type: ignore[arg-type]
            l2=l2,  # type: ignore[arg-type]
            l2_level_mask=l2_level_mask,  # type: ignore[arg-type]
            l2_snapshot_mask=l2_snapshot_mask,  # type: ignore[arg-type]
            quality=quality,  # type: ignore[arg-type]
        )
        return batch if _is_exporting() else batch.validate(self.config)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        breakdown: dict[str, int] = {}
        for name, parameter in self.named_parameters():
            component = name.split(".", maxsplit=1)[0]
            breakdown[component] = breakdown.get(component, 0) + parameter.numel()
        return dict(sorted(breakdown.items()))

    def assert_reference_parameter_count(self) -> None:
        if self.config != ChmtQConfig.reference():
            raise ValueError("exact reference count applies only to ChmtQConfig.reference()")
        if self.parameter_count != self.config.reference_parameter_count:
            raise AssertionError(
                f"expected {self.config.reference_parameter_count} parameters, "
                f"got {self.parameter_count}"
            )

    def reset_parameters(self) -> None:
        """Apply Xavier-uniform linear initialization and fixed numeric priors."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.gain)
            elif isinstance(module, ChmtQAttention):
                nn.init.zeros_(module.log_temperature)
            elif isinstance(module, RelativeTimeBias):
                nn.init.zeros_(module.gamma)

        embeddings = (
            self.modality_embedding,
            self.side_embedding,
            self.rank_embedding,
            self.horizon_queries,
            self.book_pool.query,
            self.l1_patch_pool.within_patch_embedding,
            self.l2_patch_pool.within_patch_embedding,
        )
        for parameter in embeddings:
            nn.init.trunc_normal_(parameter, std=0.02)
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                if name.endswith("layer_scale"):
                    parameter.fill_(self.config.layer_scale_initial)
        nn.init.zeros_(self.output_head.bias)


NumericMarketTransformerQ = ChmtQTransformer
