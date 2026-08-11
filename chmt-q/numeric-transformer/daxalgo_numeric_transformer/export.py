"""Fixed-shape ONNX smoke export for the non-promotable prototype."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn

from .config import ModelConfig
from .masks import _zero_scalar_like
from .synthetic import make_synthetic_batch

if TYPE_CHECKING:
    from .contracts import MarketBatch
    from .model.numeric_transformer import NumericMarketTransformer


INPUT_NAMES = (
    "bars",
    "bars_mask",
    "l1",
    "l1_mask",
    "l2",
    "l2_level_mask",
    "l2_snapshot_mask",
    "quality",
)
OUTPUT_NAME = "forecast"
OPSET_VERSION = 17
MAX_ONNX_BYTES = 12 * 1024 * 1024

INPUT_SHAPES: dict[str, tuple[int, ...]] = {
    "bars": (1, 64, 16),
    "bars_mask": (1, 64),
    "l1": (1, 240, 16),
    "l1_mask": (1, 240),
    "l2": (1, 240, 2, 10, 12),
    "l2_level_mask": (1, 240, 2, 10),
    "l2_snapshot_mask": (1, 240),
    "quality": (1, 16),
}
OUTPUT_SHAPE = (1, 5, 5)


class FrozenIdentityScaler(nn.Module):
    """Identity statistics with the same immutable boundary as a future fitted scaler."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("bars_center", torch.zeros(16))
        self.register_buffer("bars_scale", torch.ones(16))
        self.register_buffer("l1_center", torch.zeros(16))
        self.register_buffer("l1_scale", torch.ones(16))
        self.register_buffer("l2_center", torch.zeros(12))
        self.register_buffer("l2_scale", torch.ones(12))
        self.register_buffer("quality_center", torch.zeros(16))
        self.register_buffer("quality_scale", torch.ones(16))
        self.register_buffer("target_center", torch.zeros(5, 5))
        self.register_buffer("target_scale", torch.ones(5, 5))


class ExportWrapper(nn.Module):
    """Mask-aware fixed-contract wrapper around a frozen CPU FP32 model copy."""

    def __init__(
        self,
        model: NumericMarketTransformer,
        scaler: FrozenIdentityScaler | None = None,
    ) -> None:
        super().__init__()
        self.model = copy.deepcopy(model).to(device="cpu", dtype=torch.float32).eval()
        self.scaler = scaler or FrozenIdentityScaler()
        self.scaler.to(device="cpu", dtype=torch.float32).eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _masked_normalize(
        values: Tensor,
        valid_mask: Tensor,
        center: Tensor,
        scale: Tensor,
    ) -> Tensor:
        normalized = ((values - center) / scale).clamp(min=-8.0, max=8.0)
        return torch.where(
            valid_mask.unsqueeze(-1), normalized, _zero_scalar_like(normalized)
        )

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
    ) -> Tensor:
        bars = self._masked_normalize(
            bars, bars_mask, self.scaler.bars_center, self.scaler.bars_scale
        )
        l1 = self._masked_normalize(l1, l1_mask, self.scaler.l1_center, self.scaler.l1_scale)

        effective_level_mask = l2_level_mask & l2_snapshot_mask.unsqueeze(-1).unsqueeze(-1)
        l2 = ((l2 - self.scaler.l2_center) / self.scaler.l2_scale).clamp(
            min=-8.0, max=8.0
        )
        l2 = torch.where(
            effective_level_mask.unsqueeze(-1), l2, _zero_scalar_like(l2)
        )
        quality = ((quality - self.scaler.quality_center) / self.scaler.quality_scale).clamp(
            min=-8.0, max=8.0
        )

        forecast = self.model(
            bars=bars,
            bars_mask=bars_mask,
            l1=l1,
            l1_mask=l1_mask,
            l2=l2,
            l2_level_mask=effective_level_mask,
            l2_snapshot_mask=l2_snapshot_mask,
            quality=quality,
        )
        return forecast * self.scaler.target_scale + self.scaler.target_center


def batch_to_inputs(batch: MarketBatch) -> tuple[Tensor, ...]:
    """Return the inference tensors in the immutable ONNX input order."""

    return tuple(getattr(batch, name).detach().to("cpu") for name in INPUT_NAMES)


def export_onnx(
    model: NumericMarketTransformer,
    output_path: str | Path,
    *,
    config: ModelConfig | None = None,
    batch: MarketBatch | None = None,
    seed: int = 17,
) -> Path:
    """Export a fixed batch-one graph; the resulting smoke artifact is never promotable."""

    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise ImportError("ONNX export requires the optional 'onnx' extra") from error

    resolved_config = config or getattr(model, "config", None) or ModelConfig.tiny()
    sample = batch or make_synthetic_batch(
        resolved_config, batch_size=1, seed=seed, device=torch.device("cpu")
    )
    sample.validate(resolved_config).require_inference_available()
    inputs = batch_to_inputs(sample)
    actual_shapes = {name: tuple(value.shape) for name, value in zip(INPUT_NAMES, inputs)}
    if actual_shapes != INPUT_SHAPES:
        raise ValueError(f"ONNX export requires fixed batch-one shapes: {actual_shapes!r}")

    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ExportWrapper(model)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            path,
            export_params=True,
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
            input_names=list(INPUT_NAMES),
            output_names=[OUTPUT_NAME],
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.EVAL,
        )
    validate_onnx(path)
    artifact_bytes = path.stat().st_size
    if artifact_bytes >= MAX_ONNX_BYTES:
        raise ValueError(
            "ONNX artifact exceeds the reference capacity target: "
            f"{artifact_bytes} >= {MAX_ONNX_BYTES} bytes"
        )
    return path


def _onnx_dimensions(value_info: Any) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_param or not dimension.HasField("dim_value"):
            raise ValueError(f"dynamic ONNX dimension is forbidden for {value_info.name!r}")
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)


def validate_onnx(path: str | Path):
    """Run structural checks and enforce the exact fixed tensor signature."""

    try:
        import onnx
    except ImportError as error:
        raise ImportError("ONNX validation requires the optional 'onnx' extra") from error

    graph = onnx.load(str(path), load_external_data=False)
    onnx.checker.check_model(graph, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)

    actual_input_names = tuple(value.name for value in inferred.graph.input)
    actual_output_names = tuple(value.name for value in inferred.graph.output)
    if actual_input_names != INPUT_NAMES:
        raise ValueError(f"unexpected ONNX inputs: {actual_input_names!r}")
    if actual_output_names != (OUTPUT_NAME,):
        raise ValueError(f"unexpected ONNX outputs: {actual_output_names!r}")

    expected_float = onnx.TensorProto.FLOAT
    expected_bool = onnx.TensorProto.BOOL
    for value_info in inferred.graph.input:
        expected_type = expected_bool if value_info.name.endswith("mask") else expected_float
        actual_type = value_info.type.tensor_type.elem_type
        if actual_type != expected_type:
            raise ValueError(f"unexpected dtype for {value_info.name!r}: {actual_type}")
        if _onnx_dimensions(value_info) != INPUT_SHAPES[value_info.name]:
            raise ValueError(f"unexpected shape for {value_info.name!r}")

    output = inferred.graph.output[0]
    if output.type.tensor_type.elem_type != expected_float:
        raise ValueError("forecast must be float32")
    if _onnx_dimensions(output) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected forecast shape: {_onnx_dimensions(output)!r}")
    return inferred


def run_onnx(path: str | Path, batch: MarketBatch) -> np.ndarray:
    """Run the fixed graph through CPU ONNX Runtime for smoke parity."""

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("ONNX execution requires the optional 'onnxruntime' extra") from error

    batch.validate().require_inference_available()
    inputs = batch_to_inputs(batch)
    feed = {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs)}
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (forecast,) = session.run([OUTPUT_NAME], feed)
    return forecast

