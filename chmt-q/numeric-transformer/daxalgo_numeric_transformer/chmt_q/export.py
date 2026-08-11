"""Fixed-shape raw CHMT-Q ONNX export and parity helpers."""

from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .config import ChmtQConfig
from .contracts import ChmtQBatch
from .model import ChmtQTransformer

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
OUTPUT_NAME = "raw_forecast"
OPSET_VERSION = 17
MAX_ONNX_BYTES = 12 * 1024 * 1024
MAX_ONNX_NODES = 5_000
MAX_ONNX_INITIALIZERS = 1_024
ALLOWED_ONNX_OPERATORS = frozenset(
    {
        "Add",
        "And",
        "Cast",
        "Clip",
        "Concat",
        "Constant",
        "ConstantOfShape",
        "Cos",
        "Div",
        "Equal",
        "Exp",
        "Expand",
        "GatherElements",
        "Gemm",
        "Greater",
        "Identity",
        "LessOrEqual",
        "Log",
        "MatMul",
        "Mul",
        "Neg",
        "ReduceMax",
        "ReduceMean",
        "ReduceSum",
        "Reshape",
        "Shape",
        "Sigmoid",
        "Sin",
        "Slice",
        "Softplus",
        "Sqrt",
        "Squeeze",
        "Sub",
        "Tanh",
        "Transpose",
        "Unsqueeze",
        "Where",
    }
)
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
OUTPUT_SHAPE = (1, 5, 17)


class ChmtQExportWrapper(nn.Module):
    """Frozen CPU FP32 core whose inputs are already fold-normalized tensors."""

    def __init__(self, model: ChmtQTransformer) -> None:
        super().__init__()
        if not isinstance(model, ChmtQTransformer):
            raise TypeError("model must be a ChmtQTransformer")
        self.model = copy.deepcopy(model).to(device="cpu", dtype=torch.float32).eval()
        self.model.disable_attention_logit_telemetry(reset=True)
        for parameter in self.parameters():
            parameter.requires_grad_(False)

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
        return self.model(
            bars=bars,
            bars_mask=bars_mask,
            l1=l1,
            l1_mask=l1_mask,
            l2=l2,
            l2_level_mask=l2_level_mask,
            l2_snapshot_mask=l2_snapshot_mask,
            quality=quality,
        )


def batch_to_inputs(batch: ChmtQBatch) -> tuple[Tensor, ...]:
    """Return the eight inference tensors in the immutable external order."""

    return tuple(getattr(batch, name).detach().to(device="cpu") for name in INPUT_NAMES)


def export_chmt_q_onnx(
    model: ChmtQTransformer,
    output_path: str | Path,
    *,
    batch: ChmtQBatch,
) -> Path:
    """Export final raw parameters only; auxiliary and host reconstruction stay outside."""

    if not isinstance(model, ChmtQTransformer):
        raise TypeError("model must be a ChmtQTransformer")
    path = Path(output_path).resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"CHMT-Q ONNX artifact already exists: {path}")
    batch.validate(model.config).require_available()
    inputs = batch_to_inputs(batch)
    actual_shapes = {name: tuple(value.shape) for name, value in zip(INPUT_NAMES, inputs)}
    if actual_shapes != INPUT_SHAPES:
        raise ValueError(f"CHMT-Q ONNX export requires fixed batch-one shapes: {actual_shapes!r}")
    _require_finite_model(model)
    wrapper = ChmtQExportWrapper(model)
    with torch.no_grad():
        eager_output = wrapper(*inputs)
    _validate_eager_output(eager_output)

    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise ImportError("CHMT-Q ONNX export requires the optional 'onnx' extra") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{uuid.uuid4().hex}.onnx")
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                inputs,
                staging,
                export_params=True,
                opset_version=OPSET_VERSION,
                do_constant_folding=True,
                input_names=list(INPUT_NAMES),
                output_names=[OUTPUT_NAME],
                dynamic_axes=None,
                training=torch.onnx.TrainingMode.EVAL,
            )
        validate_chmt_q_onnx(staging)
        if staging.stat().st_size >= MAX_ONNX_BYTES:
            raise ValueError("CHMT-Q ONNX artifact exceeds the 12 MiB reference ceiling")
        try:
            os.link(staging, path)
        except FileExistsError as error:
            raise FileExistsError(f"CHMT-Q ONNX artifact already exists: {path}") from error
    finally:
        if staging.exists() or staging.is_symlink():
            staging.unlink()
    return path


def _require_finite_model(model: ChmtQTransformer) -> None:
    for name, value in (*tuple(model.named_parameters()), *tuple(model.named_buffers())):
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"CHMT-Q model state contains NaN or infinity: {name}")


def _validate_eager_output(output: Tensor) -> None:
    if tuple(output.shape) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected eager CHMT-Q output shape: {tuple(output.shape)!r}")
    if output.dtype != torch.float32 or output.device.type != "cpu":
        raise TypeError("eager CHMT-Q export output must be CPU float32")
    if not bool(torch.isfinite(output).all().item()):
        raise ValueError("eager CHMT-Q export output contains NaN or infinity")


def validate_chmt_q_onnx(path: str | Path) -> Any:
    """Check fixed signature plus the reviewed inference-only graph policy."""

    try:
        import onnx
    except ImportError as error:
        raise ImportError("CHMT-Q ONNX validation requires the optional 'onnx' extra") from error
    resolved = Path(path).resolve()
    if resolved.stat().st_size >= MAX_ONNX_BYTES:
        raise ValueError("CHMT-Q ONNX artifact exceeds the 12 MiB reference ceiling")
    graph = onnx.load(str(resolved), load_external_data=False)
    _validate_graph_policy(graph, onnx)
    onnx.checker.check_model(graph, full_check=True)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    inputs = tuple(value.name for value in inferred.graph.input)
    outputs = tuple(value.name for value in inferred.graph.output)
    if inputs != INPUT_NAMES or outputs != (OUTPUT_NAME,):
        raise ValueError(f"unexpected CHMT-Q ONNX signature: {inputs!r} -> {outputs!r}")
    for value in inferred.graph.input:
        expected_type = (
            onnx.TensorProto.BOOL if value.name.endswith("mask") else onnx.TensorProto.FLOAT
        )
        if value.type.tensor_type.elem_type != expected_type:
            raise ValueError(f"unexpected ONNX dtype for {value.name!r}")
        if _shape(value) != INPUT_SHAPES[value.name]:
            raise ValueError(f"unexpected ONNX shape for {value.name!r}")
    output = inferred.graph.output[0]
    if output.type.tensor_type.elem_type != onnx.TensorProto.FLOAT:
        raise ValueError("raw_forecast must be float32")
    if _shape(output) != OUTPUT_SHAPE:
        raise ValueError(f"unexpected raw_forecast shape: {_shape(output)!r}")
    return inferred


def _validate_graph_policy(graph: Any, onnx: Any) -> None:
    imports = tuple(
        (item.domain or "ai.onnx", int(item.version)) for item in graph.opset_import
    )
    if imports != (("ai.onnx", OPSET_VERSION),):
        raise ValueError(f"unexpected CHMT-Q ONNX opset imports: {imports!r}")

    nodes = tuple(graph.graph.node)
    if len(nodes) > MAX_ONNX_NODES:
        raise ValueError(f"CHMT-Q ONNX graph exceeds the {MAX_ONNX_NODES} node ceiling")
    for node in nodes:
        domain = node.domain or "ai.onnx"
        if domain != "ai.onnx":
            raise ValueError(f"non-standard ONNX node domain is forbidden: {domain!r}")
        if node.op_type not in ALLOWED_ONNX_OPERATORS:
            raise ValueError(f"ONNX operator is not in the inference allowlist: {node.op_type!r}")
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.TENSOR:
                _validate_onnx_tensor(attribute.t, onnx, f"node {node.name!r} attribute")
            elif attribute.type == onnx.AttributeProto.TENSORS:
                for tensor in attribute.tensors:
                    _validate_onnx_tensor(tensor, onnx, f"node {node.name!r} attribute")

    initializers = tuple(graph.graph.initializer)
    sparse_initializers = tuple(graph.graph.sparse_initializer)
    initializer_count = len(initializers) + len(sparse_initializers)
    if initializer_count > MAX_ONNX_INITIALIZERS:
        raise ValueError(
            "CHMT-Q ONNX graph exceeds the "
            f"{MAX_ONNX_INITIALIZERS} initializer ceiling"
        )
    for initializer in initializers:
        _validate_onnx_tensor(initializer, onnx, f"initializer {initializer.name!r}")
    for initializer in sparse_initializers:
        _validate_onnx_tensor(
            initializer.values,
            onnx,
            f"sparse initializer {initializer.values.name!r}",
        )
        _validate_onnx_tensor(
            initializer.indices,
            onnx,
            f"sparse initializer {initializer.indices.name!r}",
        )


def _validate_onnx_tensor(tensor: Any, onnx: Any, description: str) -> None:
    if tensor.data_location == onnx.TensorProto.EXTERNAL or tensor.external_data:
        raise ValueError(f"external ONNX tensor data is forbidden: {description}")
    values = onnx.numpy_helper.to_array(tensor)
    if np.issubdtype(values.dtype, np.inexact) and not bool(np.isfinite(values).all()):
        raise ValueError(f"non-finite ONNX tensor is forbidden: {description}")


def run_chmt_q_onnx(path: str | Path, batch: ChmtQBatch) -> np.ndarray:
    """Execute the fixed graph with CPU ONNX Runtime for parity testing."""

    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError(
            "CHMT-Q ONNX execution requires the optional 'onnxruntime' extra"
        ) from error
    batch.validate(ChmtQConfig.reference())
    inputs = batch_to_inputs(batch)
    actual_shapes = {name: tuple(value.shape) for name, value in zip(INPUT_NAMES, inputs)}
    if actual_shapes != INPUT_SHAPES:
        raise ValueError(
            f"CHMT-Q ONNX execution requires fixed batch-one shapes: {actual_shapes!r}"
        )
    feed = {name: value.numpy() for name, value in zip(INPUT_NAMES, inputs)}
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    (raw,) = session.run([OUTPUT_NAME], feed)
    return raw


def _shape(value_info: Any) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if dimension.dim_param or not dimension.HasField("dim_value"):
            raise ValueError(f"dynamic dimension is forbidden for {value_info.name!r}")
        dimensions.append(dimension.dim_value)
    return tuple(dimensions)
