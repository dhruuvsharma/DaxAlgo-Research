from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.contracts import NoAvailableModalitiesError
from daxalgo_numeric_transformer.chmt_q.export import (
    ALLOWED_ONNX_OPERATORS,
    INPUT_NAMES,
    INPUT_SHAPES,
    MAX_ONNX_BYTES,
    MAX_ONNX_INITIALIZERS,
    MAX_ONNX_NODES,
    OPSET_VERSION,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    ChmtQExportWrapper,
    batch_to_inputs,
    export_chmt_q_onnx,
    run_chmt_q_onnx,
    validate_chmt_q_onnx,
)
from daxalgo_numeric_transformer.chmt_q.model import ChmtQTransformer
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch


@pytest.mark.onnx
def test_fixed_raw_signature_without_runtime_dependency(tmp_path) -> None:
    pytest.importorskip("onnx")
    torch.manual_seed(101)
    config = ChmtQConfig.tiny(decoder_layers=2)
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=103)
    path = export_chmt_q_onnx(model, tmp_path / "chmt-q.onnx", batch=batch)

    checked = validate_chmt_q_onnx(path)
    assert tuple(value.name for value in checked.graph.input) == INPUT_NAMES
    assert tuple(value.name for value in checked.graph.output) == (OUTPUT_NAME,)
    assert tuple(_shape(value) for value in checked.graph.input) == tuple(INPUT_SHAPES.values())
    assert _shape(checked.graph.output[0]) == OUTPUT_SHAPE


@pytest.mark.onnx
def test_runtime_parity(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(103)
    config = ChmtQConfig.tiny(decoder_layers=2)
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=105)
    path = export_chmt_q_onnx(model, tmp_path / "chmt-q-runtime.onnx", batch=batch)

    wrapper = ChmtQExportWrapper(model)
    for name, fixture in _parity_fixtures(config, batch).items():
        with torch.no_grad():
            expected = wrapper(*batch_to_inputs(fixture)).numpy()
        actual = run_chmt_q_onnx(path, fixture)
        assert actual.shape == OUTPUT_SHAPE, name
        assert np.isfinite(actual).all(), name
        np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5)

    rejected = copy.deepcopy(batch)
    rejected.bars_mask.zero_()
    rejected.l1_mask.zero_()
    rejected.l2_level_mask.zero_()
    rejected.l2_snapshot_mask.zero_()
    with pytest.raises(NoAvailableModalitiesError):
        run_chmt_q_onnx(path, rejected)


@pytest.mark.onnx
def test_reference_graph_stays_below_capacity_target(tmp_path) -> None:
    pytest.importorskip("onnx")
    torch.manual_seed(107)
    config = ChmtQConfig.reference()
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=109)

    path = export_chmt_q_onnx(model, tmp_path / "reference.onnx", batch=batch)

    assert model.parameter_count == 2_089_554
    assert path.stat().st_size < MAX_ONNX_BYTES


def test_export_never_overwrites_an_existing_artifact(tmp_path) -> None:
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=113)
    path = tmp_path / "existing.onnx"
    path.write_bytes(b"keep-me")

    with pytest.raises(FileExistsError, match="already exists"):
        export_chmt_q_onnx(model, path, batch=batch)

    assert path.read_bytes() == b"keep-me"


def test_export_rejects_nonfinite_model_state_before_writing(tmp_path) -> None:
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=127)
    with torch.no_grad():
        model.output_head.weight[0, 0] = torch.nan
    path = tmp_path / "nonfinite.onnx"

    with pytest.raises(ValueError, match="model state contains NaN or infinity"):
        export_chmt_q_onnx(model, path, batch=batch)

    assert not path.exists()


def test_export_wrapper_forcibly_disables_attention_telemetry() -> None:
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    model.enable_attention_logit_telemetry(runaway_threshold=80.0)
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=131)

    wrapper = ChmtQExportWrapper(model)
    with torch.no_grad():
        wrapper(*batch_to_inputs(batch))

    assert wrapper.model.collect_attention_logit_telemetry() == {}
    assert model.collect_attention_logit_telemetry() == {}


@pytest.mark.onnx
def test_reference_runtime_parity_uses_promotion_tolerance(tmp_path) -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(137)
    config = ChmtQConfig.reference()
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=139)
    path = export_chmt_q_onnx(model, tmp_path / "reference-runtime.onnx", batch=batch)

    with torch.no_grad():
        expected = ChmtQExportWrapper(model)(*batch_to_inputs(batch)).numpy()
    actual = run_chmt_q_onnx(path, batch)

    np.testing.assert_allclose(actual, expected, rtol=1.0e-4, atol=1.0e-5)


@pytest.mark.onnx
def test_validator_enforces_reviewed_graph_policy(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    torch.manual_seed(149)
    config = ChmtQConfig.tiny()
    model = ChmtQTransformer(config).eval()
    batch = make_chmt_q_synthetic_batch(config, batch_size=1, seed=151)
    path = export_chmt_q_onnx(model, tmp_path / "policy.onnx", batch=batch)
    graph = onnx.load(str(path), load_external_data=False)

    assert tuple((item.domain or "ai.onnx", item.version) for item in graph.opset_import) == (
        ("ai.onnx", OPSET_VERSION),
    )
    assert len(graph.graph.node) <= MAX_ONNX_NODES
    assert len(graph.graph.initializer) <= MAX_ONNX_INITIALIZERS
    assert {node.op_type for node in graph.graph.node} <= ALLOWED_ONNX_OPERATORS
    assert all((node.domain or "ai.onnx") == "ai.onnx" for node in graph.graph.node)

    wrong_opset = copy.deepcopy(graph)
    wrong_opset.opset_import[0].version = OPSET_VERSION - 1
    wrong_opset_path = tmp_path / "wrong-opset.onnx"
    onnx.save(wrong_opset, str(wrong_opset_path))
    with pytest.raises(ValueError, match="opset imports"):
        validate_chmt_q_onnx(wrong_opset_path)

    custom_domain = copy.deepcopy(graph)
    custom_domain.graph.node[0].domain = "example.training"
    custom_domain_path = tmp_path / "custom-domain.onnx"
    onnx.save(custom_domain, str(custom_domain_path))
    with pytest.raises(ValueError, match="non-standard"):
        validate_chmt_q_onnx(custom_domain_path)

    control_flow = copy.deepcopy(graph)
    control_flow.graph.node[0].op_type = "Loop"
    control_flow_path = tmp_path / "control-flow.onnx"
    onnx.save(control_flow, str(control_flow_path))
    with pytest.raises(ValueError, match="allowlist"):
        validate_chmt_q_onnx(control_flow_path)

    nonfinite = copy.deepcopy(graph)
    initializer_index = next(
        index
        for index, initializer in enumerate(nonfinite.graph.initializer)
        if onnx.numpy_helper.to_array(initializer).dtype.kind == "f"
    )
    initializer = nonfinite.graph.initializer[initializer_index]
    values = onnx.numpy_helper.to_array(initializer).copy()
    values.flat[0] = np.nan
    replacement = onnx.numpy_helper.from_array(values, name=initializer.name)
    nonfinite.graph.initializer[initializer_index].CopyFrom(replacement)
    nonfinite_path = tmp_path / "nonfinite-initializer.onnx"
    onnx.save(nonfinite, str(nonfinite_path))
    with pytest.raises(ValueError, match="non-finite ONNX tensor"):
        validate_chmt_q_onnx(nonfinite_path)

    external = copy.deepcopy(graph)
    external_initializer = external.graph.initializer[0]
    external_initializer.data_location = onnx.TensorProto.EXTERNAL
    location = external_initializer.external_data.add()
    location.key = "location"
    location.value = "weights.bin"
    external_path = tmp_path / "external-data.onnx"
    onnx.save(external, str(external_path))
    with pytest.raises(ValueError, match="external ONNX tensor data"):
        validate_chmt_q_onnx(external_path)


def _shape(value_info) -> tuple[int, ...]:
    return tuple(dimension.dim_value for dimension in value_info.type.tensor_type.shape.dim)


def _parity_fixtures(config: ChmtQConfig, normal):
    sparse = copy.deepcopy(normal)
    sparse.bars_mask[:, ::3] = False
    sparse.l1_mask[:, ::4] = False
    sparse.l2_level_mask[:, ::5, :, 2:] = False
    sparse.l2_snapshot_mask = sparse.l2_level_mask.any(dim=(-1, -2))

    bars_only = copy.deepcopy(normal)
    bars_only.l1_mask.zero_()
    bars_only.l2_level_mask.zero_()
    bars_only.l2_snapshot_mask.zero_()

    one_sided = copy.deepcopy(normal)
    one_sided.l2_level_mask[:, :, 1].zero_()
    one_sided.l2_snapshot_mask = one_sided.l2_level_mask.any(dim=(-1, -2))

    clipping = copy.deepcopy(normal)
    clipping.bars.copy_(torch.where(clipping.bars >= 0.0, 8.0, -8.0))
    clipping.l1.copy_(torch.where(clipping.l1 >= 0.0, 8.0, -8.0))
    clipping.l2.copy_(torch.where(clipping.l2 >= 0.0, 8.0, -8.0))

    extreme = copy.deepcopy(normal)
    extreme.bars.fill_(1.0e4)
    extreme.l1.fill_(-1.0e4)
    extreme.l2.fill_(1.0e4)

    fixtures = {
        "normal": normal,
        "sparse": sparse,
        "bars_only": bars_only,
        "one_sided": one_sided,
        "clipping": clipping,
        "extreme_finite": extreme,
    }
    for fixture in fixtures.values():
        fixture.validate(config)
    return fixtures
