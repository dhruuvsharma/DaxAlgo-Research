from __future__ import annotations

import numpy as np
import pytest
import torch

from daxalgo_numeric_transformer.config import ModelConfig
from daxalgo_numeric_transformer.export import (
    INPUT_NAMES,
    INPUT_SHAPES,
    MAX_ONNX_BYTES,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    ExportWrapper,
    batch_to_inputs,
    export_onnx,
    run_onnx,
    validate_onnx,
)
from daxalgo_numeric_transformer.model.numeric_transformer import NumericMarketTransformer
from daxalgo_numeric_transformer.synthetic import make_synthetic_batch

pytestmark = pytest.mark.onnx


@pytest.fixture(scope="module", autouse=True)
def require_onnx_dependencies() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")


def test_fixed_onnx_signature_checker_and_runtime_parity(tmp_path) -> None:
    torch.manual_seed(53)
    config = ModelConfig.tiny()
    model = NumericMarketTransformer(config).eval()
    batch = make_synthetic_batch(config, batch_size=1, seed=59, device="cpu")
    path = export_onnx(model, tmp_path / "smoke.onnx", config=config, batch=batch)

    checked = validate_onnx(path)
    assert tuple(value.name for value in checked.graph.input) == INPUT_NAMES
    assert tuple(value.name for value in checked.graph.output) == (OUTPUT_NAME,)
    assert tuple(_shape(value) for value in checked.graph.input) == tuple(INPUT_SHAPES.values())
    assert _shape(checked.graph.output[0]) == OUTPUT_SHAPE

    wrapper = ExportWrapper(model).eval()
    with torch.no_grad():
        expected = wrapper(*batch_to_inputs(batch)).numpy()
    actual = run_onnx(path, batch)

    assert actual.shape == OUTPUT_SHAPE
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=2.0e-4, atol=2.0e-5)


def test_export_wrapper_scalers_are_frozen_identity() -> None:
    torch.manual_seed(61)
    wrapper = ExportWrapper(NumericMarketTransformer(ModelConfig.tiny()))
    buffers = dict(wrapper.scaler.named_buffers())

    assert set(buffers) == {
        "bars_center",
        "bars_scale",
        "l1_center",
        "l1_scale",
        "l2_center",
        "l2_scale",
        "quality_center",
        "quality_scale",
        "target_center",
        "target_scale",
    }
    assert all(not value.requires_grad for value in buffers.values())
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in buffers.items()
        if "center" in name
    )
    assert all(torch.all(value == 1) for name, value in buffers.items() if "scale" in name)


def test_reference_export_stays_below_capacity_target(tmp_path) -> None:
    torch.manual_seed(67)
    config = ModelConfig.reference()
    model = NumericMarketTransformer(config).eval()
    batch = make_synthetic_batch(config, batch_size=1, seed=71, device="cpu")

    path = export_onnx(model, tmp_path / "reference.onnx", config=config, batch=batch)

    assert sum(parameter.numel() for parameter in model.parameters()) <= 2_250_000
    assert path.stat().st_size < MAX_ONNX_BYTES


def _shape(value_info) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim)

