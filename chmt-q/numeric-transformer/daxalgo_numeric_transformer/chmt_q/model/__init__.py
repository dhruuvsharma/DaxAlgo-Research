"""Selected CHMT-Q model primitives and complete Transformer."""

from .attention import (
    DEFAULT_LOGIT_RUNAWAY_THRESHOLD,
    AttentionLogitTelemetry,
    ChmtQAttention,
    PrimitiveChmtQAttention,
    safe_masked_softmax,
)
from .blocks import (
    ChmtQDecoderBlock,
    ChmtQEncoderBlock,
    NumericTimeEncoding,
    RelativeTimeBias,
    RMSNorm,
    SwiGLU,
)
from .pooling import BookQueryPool, MaskedPatchPool
from .transformer import (
    ChmtQRawOutputs,
    ChmtQTrainingOutput,
    ChmtQTransformer,
    EncodedMarket,
    NumericMarketTransformerQ,
)

__all__ = [
    "BookQueryPool",
    "DEFAULT_LOGIT_RUNAWAY_THRESHOLD",
    "AttentionLogitTelemetry",
    "ChmtQAttention",
    "ChmtQDecoderBlock",
    "ChmtQEncoderBlock",
    "ChmtQRawOutputs",
    "ChmtQTrainingOutput",
    "ChmtQTransformer",
    "EncodedMarket",
    "MaskedPatchPool",
    "NumericMarketTransformerQ",
    "NumericTimeEncoding",
    "PrimitiveChmtQAttention",
    "RMSNorm",
    "RelativeTimeBias",
    "SwiGLU",
    "safe_masked_softmax",
]
