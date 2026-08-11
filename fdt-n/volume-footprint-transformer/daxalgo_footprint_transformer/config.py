from __future__ import annotations

from dataclasses import asdict, dataclass

ROW_FEATURE_NAMES = (
    "price_offset_from_poc",
    "buy_share",
    "sell_share",
    "total_share",
    "delta_share",
    "log_buy_ratio",
    "log_sell_ratio",
    "bid_imbalance",
    "ask_imbalance",
    "zero_bid",
    "zero_ask",
)

BAR_FEATURE_NAMES = (
    "poc_offset_from_anchor",
    "buy_poc_offset",
    "sell_poc_offset",
    "volume_centroid_offset",
    "buy_centroid_offset",
    "sell_centroid_offset",
    "low_gap",
    "high_gap",
    "value_area_low_gap",
    "value_area_high_gap",
    "log_volume_relative",
    "buy_fraction",
    "sell_fraction",
    "delta_fraction",
    "cumulative_delta_change",
    "stacked_buy_fraction",
    "stacked_sell_fraction",
    "row_count_fraction",
    "feed_quality",
    "minute_sin",
    "minute_cos",
    "weekday_sin",
    "weekday_cos",
)

TARGET_NAMES = (
    "poc_offset",
    "low_offset",
    "high_offset",
    "buy_poc_offset",
    "sell_poc_offset",
    "log_volume_relative",
    "delta_fraction",
)


@dataclass(frozen=True)
class EncodingConfig:
    lookback_bars: int = 64
    horizon_bars: int = 8
    max_rows: int = 1024
    row_size: float = 2.5
    interval_seconds: int = 60
    imbalance_ratio: float = 3.0
    price_scale_ticks: float = 64.0
    gap_scale_ticks: float = 32.0
    log_volume_scale: float = 4.0
    cumulative_delta_scale: float = 4.0
    quantity_scale: float = 1_000.0

    def validate(self) -> None:
        if self.lookback_bars < 2:
            raise ValueError("lookback_bars must be at least two")
        if self.horizon_bars < 1:
            raise ValueError("horizon_bars must be positive")
        if self.max_rows < 4:
            raise ValueError("max_rows must be at least four")
        if self.row_size <= 0 or self.interval_seconds <= 0:
            raise ValueError("row_size and interval_seconds must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 64
    row_queries: int = 4
    attention_heads: int = 4
    temporal_layers: int = 3
    decoder_layers: int = 2
    feedforward_multiplier: int = 3
    dropout: float = 0.10

    @classmethod
    def from_preset(cls, name: str) -> ModelConfig:
        if name == "fdt-n":
            return cls()
        if name == "fdt-n-tiny":
            return cls(
                d_model=32,
                row_queries=2,
                attention_heads=4,
                temporal_layers=2,
                decoder_layers=1,
                feedforward_multiplier=2,
                dropout=0.10,
            )
        raise ValueError(f"unknown model preset: {name}")

    def validate(self) -> None:
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if min(self.row_queries, self.temporal_layers, self.decoder_layers) < 1:
            raise ValueError("query/layer counts must be positive")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
