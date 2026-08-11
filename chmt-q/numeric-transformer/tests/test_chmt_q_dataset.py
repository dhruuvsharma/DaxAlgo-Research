from __future__ import annotations

import gc
import hashlib
import json
import weakref
from contextlib import contextmanager
from dataclasses import replace

import pytest
import torch

import daxalgo_numeric_transformer.chmt_q.data.dataset as dataset_module
from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.data.dataset import (
    FEATURE_SCHEMA_SHA256,
    TARGET_SCHEMA_SHA256,
    ArrowDataset,
    DatasetCapabilities,
    DatasetManifest,
    DatasetShard,
    EngineeredExample,
    ExampleMetadata,
    arrow_schema_sha256,
    chmt_q_arrow_schema,
    write_engineered_dataset,
    write_engineered_fixture_dataset,
)
from daxalgo_numeric_transformer.chmt_q.data.research_authorization import (
    COMPILER_REPORT_NAME,
    LocalResearchAuthorization,
)
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch


def _capabilities(
    *,
    training: bool = True,
    rights: bool = True,
    provenance: bool = True,
) -> DatasetCapabilities:
    return DatasetCapabilities(
        training_ready=training,
        rights_confirmed=rights,
        provenance_complete=provenance,
        readiness_evidence_sha256="a" * 64,
        rights_evidence_sha256="b" * 64,
        provenance_evidence_sha256="c" * 64,
    )


def _examples(count: int = 5) -> list[EngineeredExample]:
    return [
        EngineeredExample(
            metadata=ExampleMetadata(
                example_id=f"example-{index}",
                session_id=f"session-{index // 2}",
                feature_cutoff_ns=index * 1_000 + 90,
                decision_ns=index * 1_000 + 100,
                dependency_start_ns=index * 1_000,
                dependency_end_ns=index * 1_000 + 200,
                instrument_id="BTC-USD",
                anchor_source="l1_mid",
                anchor_price=50_000.0 + index,
                tick_size=0.5,
                log_volume_reference=1.0,
                provenance_sha256=f"{index + 1:064x}",
            ),
            batch=make_chmt_q_synthetic_batch(ChmtQConfig.tiny(), batch_size=1, seed=100 + index),
        )
        for index in range(count)
    ]


def _write(
    tmp_path,
    *,
    training: bool = True,
    rights: bool = True,
    provenance: bool = True,
):
    return write_engineered_fixture_dataset(
        tmp_path,
        _examples(),
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=_capabilities(
            training=training,
            rights=rights,
            provenance=provenance,
        ),
        rows_per_shard=2,
    )


def _write_compiler_report(root, manifest) -> None:
    report = {
        "schema_version": 1,
        "status": "compiled_non_promotable",
        "dataset_id": manifest.dataset_id,
        "dataset_rows": manifest.row_count,
        "instrument_id": manifest.instrument_id,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "capabilities": manifest.capabilities.to_payload(),
    }
    (root / COMPILER_REPORT_NAME).write_text(json.dumps(report, sort_keys=True), encoding="utf-8")


def test_production_writer_refuses_self_authorized_capabilities(tmp_path) -> None:
    with pytest.raises(ValueError, match="cannot self-authorize"):
        write_engineered_dataset(
            tmp_path / "unsafe",
            _examples(1),
            source_manifest_sha256="d" * 64,
            builder_sha256="e" * 64,
            capabilities=_capabilities(),
        )
    assert not (tmp_path / "unsafe").exists()


def _canonical_line(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _rewrite_provenance_artifact(tmp_path, manifest, content: bytes):
    readiness_artifact, rights_artifact, provenance_artifact = (
        manifest.capabilities.evidence_artifacts
    )
    provenance_path = tmp_path / provenance_artifact.path
    provenance_path.write_bytes(content)
    changed_provenance = replace(
        provenance_artifact,
        byte_size=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    readiness_path = tmp_path / readiness_artifact.path
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    readiness["provenance_inventory"] = changed_provenance.to_payload()
    readiness_content = _canonical_line(readiness)
    readiness_path.write_bytes(readiness_content)
    changed_readiness = replace(
        readiness_artifact,
        byte_size=len(readiness_content),
        sha256=hashlib.sha256(readiness_content).hexdigest(),
    )
    capabilities = replace(
        manifest.capabilities,
        readiness_evidence_sha256=changed_readiness.sha256,
        provenance_evidence_sha256=changed_provenance.sha256,
        evidence_artifacts=(
            changed_readiness,
            rights_artifact,
            changed_provenance,
        ),
    )
    changed_manifest = replace(manifest, capabilities=capabilities)
    (tmp_path / "dataset.json").write_text(changed_manifest.to_json(), encoding="utf-8")
    return changed_manifest


def test_fixture_writer_round_trips_exact_schema_and_content_address(tmp_path) -> None:
    manifest = _write(tmp_path)

    restored = DatasetManifest.from_json(manifest.to_json())
    dataset = ArrowDataset.open(
        tmp_path,
        expected_dataset_id=manifest.dataset_id,
        expected_instrument_id="BTC-USD",
    )
    schema = chmt_q_arrow_schema()

    assert restored == manifest
    assert restored.to_json() == manifest.to_json()
    assert manifest.feature_schema_sha256 == FEATURE_SCHEMA_SHA256
    assert manifest.target_schema_sha256 == TARGET_SCHEMA_SHA256
    assert manifest.arrow_schema_sha256 == arrow_schema_sha256()
    assert manifest.instrument_id == "BTC-USD"
    assert tuple(item.purpose for item in manifest.capabilities.evidence_artifacts) == (
        "readiness",
        "rights",
        "provenance",
    )
    for artifact in manifest.capabilities.evidence_artifacts:
        path = tmp_path.joinpath(*artifact.path.split("/"))
        assert path.is_file()
        assert path.stat().st_size == artifact.byte_size
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact.sha256
    readiness_path = tmp_path / manifest.capabilities.evidence_artifacts[0].path
    readiness = json.loads(readiness_path.read_text(encoding="ascii"))
    binding = readiness["binding"]
    assert binding == {
        "arrow_schema_sha256": manifest.arrow_schema_sha256,
        "builder_sha256": manifest.builder_sha256,
        "dataset_format": manifest.format,
        "dataset_schema_version": manifest.schema_version,
        "feature_schema_sha256": manifest.feature_schema_sha256,
        "instrument_id": manifest.instrument_id,
        "row_count": manifest.row_count,
        "source_manifest_sha256": manifest.source_manifest_sha256,
        "target_schema_sha256": manifest.target_schema_sha256,
        "tensor_schema_sha256": manifest.tensor_schema_sha256,
    }
    provenance_path = tmp_path / manifest.capabilities.evidence_artifacts[2].path
    provenance_lines = provenance_path.read_text(encoding="ascii").splitlines()
    assert json.loads(provenance_lines[0])["binding"] == binding
    assert [json.loads(item) for item in provenance_lines[1:]] == [
        {
            "example_id": item.metadata.example_id,
            "provenance_sha256": item.metadata.provenance_sha256,
        }
        for item in _examples()
    ]
    assert schema.names == [
        "example_id",
        "session_id",
        "feature_cutoff_ns",
        "decision_ns",
        "dependency_start_ns",
        "dependency_end_ns",
        "instrument_id",
        "anchor_source",
        "anchor_price",
        "tick_size",
        "log_volume_reference",
        "provenance_sha256",
        "bars",
        "bars_mask",
        "l1",
        "l1_mask",
        "l2",
        "l2_level_mask",
        "l2_snapshot_mask",
        "quality",
        "targets",
        "target_mask",
        "presence_labels",
        "presence_mask",
        "feature_schema_sha256",
        "target_schema_sha256",
    ]
    assert dataset.example_ids == tuple(f"example-{index}" for index in range(5))


def test_vectorized_record_batch_materialization_matches_row_path() -> None:
    examples = _examples(2)
    record_batch = dataset_module._table_from_engineered(
        examples, chmt_q_arrow_schema()
    ).to_batches()[0]

    for candidate in (record_batch, record_batch.slice(1, 1)):
        vectorized = dataset_module._batch_from_record_batch(candidate)
        rowwise = dataset_module._batch_from_rows(
            [dataset_module._tensor_row(candidate, index) for index in range(candidate.num_rows)]
        )

        for name in (
            "bars",
            "bars_mask",
            "l1",
            "l1_mask",
            "l2",
            "l2_level_mask",
            "l2_snapshot_mask",
            "quality",
            "targets",
            "target_mask",
            "presence_labels",
            "presence_mask",
        ):
            assert torch.equal(getattr(vectorized, name), getattr(rowwise, name))


def test_open_verification_does_not_expand_arrow_rows(tmp_path, monkeypatch) -> None:
    manifest = _write(tmp_path)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("dataset verification must not expand Arrow rows")

    monkeypatch.setattr(dataset_module, "_tensor_row", fail_if_called)
    dataset = ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)

    assert dataset.example_ids


def test_vectorized_record_batch_materialization_rejects_nulls() -> None:
    pa = dataset_module._require_pyarrow()
    record_batch = dataset_module._table_from_engineered(
        _examples(2), chmt_q_arrow_schema()
    ).to_batches()[0]
    bars_index = record_batch.schema.get_field_index("bars")
    bars = record_batch.column(bars_index)

    outer_null = pa.array([None, bars[1].as_py()], type=bars.type)
    outer_columns = list(record_batch.columns)
    outer_columns[bars_index] = outer_null
    outer_batch = pa.RecordBatch.from_arrays(outer_columns, schema=record_batch.schema)
    with pytest.raises(ValueError, match="bars contains null rows"):
        dataset_module._batch_from_record_batch(outer_batch)

    child_values = bars.to_pylist()
    child_values[0][0] = None
    child_null = pa.array(child_values, type=bars.type)
    child_columns = list(record_batch.columns)
    child_columns[bars_index] = child_null
    child_batch = pa.RecordBatch.from_arrays(child_columns, schema=record_batch.schema)
    with pytest.raises(ValueError, match="bars contains null values"):
        dataset_module._batch_from_record_batch(child_batch)


def test_vectorized_record_batch_materialization_preserves_domain_checks() -> None:
    examples = _examples(1)
    examples[0].batch.l2[0, 0, 0, 0, 10] = 0.5
    record_batch = dataset_module._table_from_engineered(
        examples, chmt_q_arrow_schema()
    ).to_batches()[0]

    with pytest.raises(ValueError, match="l2 feature 10 must be binary"):
        dataset_module._batch_from_record_batch(record_batch)


def test_vectorized_selected_rows_match_source_across_record_batches_and_shards(
    tmp_path, monkeypatch
) -> None:
    examples = _examples(9)
    manifest = write_engineered_fixture_dataset(
        tmp_path,
        examples,
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=_capabilities(),
        rows_per_shard=5,
        rows_per_record_batch=2,
    )
    dataset = ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("selected-row materialization must not expand Arrow scalars")

    monkeypatch.setattr(dataset_module, "_tensor_row", fail_if_called)
    batches = list(
        dataset.iter_batches(
            reversed(dataset.example_ids),
            batch_size=3,
            partition="evaluation",
        )
    )

    assert [batch.batch.batch_size for batch in batches] == [3, 3, 3]
    assert (
        tuple(example_id for batch in batches for example_id in batch.example_ids)
        == dataset.example_ids
    )
    by_id = {example.metadata.example_id: example for example in examples}
    for materialized in batches:
        expected = tuple(by_id[example_id] for example_id in materialized.example_ids)
        assert materialized.decision_ns == tuple(item.metadata.decision_ns for item in expected)
        for name in dataset_module._DATASET_BATCH_TENSOR_NAMES:
            tensors = tuple(getattr(item.batch, name) for item in expected)
            assert all(isinstance(tensor, torch.Tensor) for tensor in tensors)
            assert torch.equal(
                getattr(materialized.batch, name),
                torch.cat(tensors, dim=0),
            )


def test_materialization_keeps_ids_and_separates_train_from_evaluation_order(
    tmp_path,
) -> None:
    manifest = _write(tmp_path)
    dataset = ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)
    reversed_ids = tuple(reversed(dataset.example_ids))

    evaluation = list(
        dataset.iter_batches(
            reversed_ids,
            batch_size=2,
            partition="evaluation",
        )
    )
    first_train = list(
        dataset.iter_batches(
            dataset.example_ids,
            batch_size=5,
            partition="train",
            seed=41,
        )
    )
    second_train = list(
        dataset.iter_batches(
            reversed_ids,
            batch_size=5,
            partition="train",
            seed=41,
        )
    )

    assert tuple(item for batch in evaluation for item in batch.example_ids) == dataset.example_ids
    assert first_train[0].example_ids == second_train[0].example_ids
    assert first_train[0].batch.bars.shape == (5, 64, 16)
    assert torch.equal(
        first_train[0].batch.l2_snapshot_mask,
        second_train[0].batch.l2_snapshot_mask,
    )
    with pytest.raises(ValueError, match="does not accept a seed"):
        list(
            dataset.iter_batches(
                dataset.example_ids,
                batch_size=2,
                partition="evaluation",
                seed=1,
            )
        )
    with pytest.raises(ValueError, match="requires a non-negative"):
        list(
            dataset.iter_batches(
                dataset.example_ids,
                batch_size=2,
                partition="train",
            )
        )


def test_training_order_is_seeded_at_shard_record_batch_and_row_levels(
    tmp_path,
) -> None:
    manifest = write_engineered_fixture_dataset(
        tmp_path,
        _examples(12),
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=_capabilities(),
        rows_per_shard=6,
        rows_per_record_batch=2,
    )
    dataset = ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)

    first = tuple(
        example_id
        for batch in dataset.iter_batches(
            dataset.example_ids,
            batch_size=3,
            partition="train",
            seed=41,
        )
        for example_id in batch.example_ids
    )
    repeated = tuple(
        example_id
        for batch in dataset.iter_batches(
            reversed(dataset.example_ids),
            batch_size=4,
            partition="train",
            seed=41,
        )
        for example_id in batch.example_ids
    )
    different_seed = tuple(
        example_id
        for batch in dataset.iter_batches(
            dataset.example_ids,
            batch_size=5,
            partition="train",
            seed=42,
        )
        for example_id in batch.example_ids
    )
    dropped = tuple(
        example_id
        for batch in dataset.iter_batches(
            reversed(dataset.example_ids),
            batch_size=5,
            partition="train",
            seed=41,
            drop_last=True,
        )
        for example_id in batch.example_ids
    )

    assert first == repeated
    assert first != different_seed
    assert dropped == first[:10]
    assert len(first) == len(set(first)) == len(dataset.example_ids)
    assert set(first) == set(dataset.example_ids)
    structural_groups = [
        (
            dataset._locations[example_id].shard_index,
            dataset._locations[example_id].record_batch_index,
        )
        for example_id in first
    ]
    completed: set[tuple[int, int]] = set()
    previous = structural_groups[0]
    for current in structural_groups[1:]:
        if current == previous:
            continue
        completed.add(previous)
        assert current not in completed
        previous = current


def test_open_fails_closed_on_capability_and_shard_integrity(tmp_path) -> None:
    manifest = _write(tmp_path, rights=False)

    with pytest.raises(PermissionError, match="rights_confirmed"):
        ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)
    inspected = ArrowDataset.open(
        tmp_path,
        expected_dataset_id=manifest.dataset_id,
        require_training_capabilities=False,
    )
    assert inspected.example_ids
    with pytest.raises(PermissionError, match="rights_confirmed"):
        list(
            inspected.iter_batches(
                inspected.example_ids,
                batch_size=1,
                partition="evaluation",
            )
        )

    shard_path = tmp_path / manifest.shards[0].path
    with shard_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="byte size"):
        ArrowDataset.open(
            tmp_path,
            expected_dataset_id=manifest.dataset_id,
            require_training_capabilities=False,
        )


def test_local_research_authorization_allows_development_batches_only(tmp_path) -> None:
    manifest = _write(
        tmp_path,
        training=False,
        rights=False,
        provenance=False,
    )
    _write_compiler_report(tmp_path, manifest)
    authorization = LocalResearchAuthorization.create(
        manifest,
        tmp_path,
        authorizer="workspace-user",
        authorized_at_utc="2026-07-25T09:45:00Z",
        reason="Explicit local non-promotable development training.",
        rights_confirmed=True,
        incomplete_readiness_acknowledged=True,
        incomplete_provenance_acknowledged=True,
    )

    dataset = ArrowDataset.open(
        tmp_path,
        expected_dataset_id=manifest.dataset_id,
        local_research_authorization=authorization,
    )
    batches = list(
        dataset.iter_batches(
            dataset.example_ids[:2],
            batch_size=2,
            partition="evaluation",
        )
    )

    assert dataset.local_research_only
    assert batches[0].example_ids == dataset.example_ids[:2]
    with pytest.raises(PermissionError, match="training_ready"):
        dataset.manifest.capabilities.require_training()


def test_manifest_rejects_noncanonical_identity_and_unsafe_shard_paths(tmp_path) -> None:
    manifest = _write(tmp_path)

    with pytest.raises(ValueError, match="dataset_id"):
        DatasetManifest.from_json(manifest.to_json().replace(manifest.dataset_id, "f" * 64))
    with pytest.raises(ValueError, match="relative"):
        replace(manifest.shards[0], path="../outside.arrow")
    with pytest.raises(ValueError, match="traversal-free"):
        replace(
            manifest.capabilities.evidence_artifacts[0],
            path="../outside.json",
        )
    with pytest.raises(ValueError, match="manifest row_count"):
        replace(manifest, row_count=manifest.row_count + 1)


def test_open_fails_closed_on_incomplete_missing_or_mismatched_evidence(tmp_path) -> None:
    incomplete_root = tmp_path / "incomplete"
    incomplete = _write(incomplete_root)
    incomplete_capabilities = replace(
        incomplete.capabilities,
        evidence_artifacts=(),
    )
    incomplete_manifest = replace(
        incomplete,
        capabilities=incomplete_capabilities,
    )
    (incomplete_root / "dataset.json").write_text(incomplete_manifest.to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory is incomplete"):
        ArrowDataset.open(
            incomplete_root,
            expected_dataset_id=incomplete_manifest.dataset_id,
            require_training_capabilities=False,
        )

    missing_root = tmp_path / "missing"
    missing = _write(missing_root)
    missing_artifact = missing.capabilities.evidence_artifacts[0]
    missing_root.joinpath(*missing_artifact.path.split("/")).unlink()
    with pytest.raises(ValueError, match="is missing"):
        ArrowDataset.open(
            missing_root,
            expected_dataset_id=missing.dataset_id,
            require_training_capabilities=False,
        )

    nonregular_root = tmp_path / "nonregular"
    nonregular = _write(nonregular_root)
    nonregular_artifact = nonregular.capabilities.evidence_artifacts[0]
    nonregular_path = nonregular_root.joinpath(*nonregular_artifact.path.split("/"))
    nonregular_path.unlink()
    nonregular_path.mkdir()
    with pytest.raises(ValueError, match="regular file"):
        ArrowDataset.open(
            nonregular_root,
            expected_dataset_id=nonregular.dataset_id,
            require_training_capabilities=False,
        )

    mismatch_root = tmp_path / "mismatch"
    mismatch = _write(mismatch_root)
    mismatch_artifact = mismatch.capabilities.evidence_artifacts[0]
    mismatch_path = mismatch_root.joinpath(*mismatch_artifact.path.split("/"))
    content = bytearray(mismatch_path.read_bytes())
    content[0] ^= 1
    mismatch_path.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256"):
        ArrowDataset.open(
            mismatch_root,
            expected_dataset_id=mismatch.dataset_id,
            require_training_capabilities=False,
        )


def test_open_rejects_symlinked_evidence(tmp_path, monkeypatch) -> None:
    root = tmp_path / "symlink"
    manifest = _write(root)
    artifact = manifest.capabilities.evidence_artifacts[0]
    artifact_path = root.joinpath(*artifact.path.split("/"))
    target_path = artifact_path.with_name("readiness-target.json")
    artifact_path.rename(target_path)
    try:
        artifact_path.symlink_to(target_path.name)
    except OSError:
        path_type = type(artifact_path)
        original_is_symlink = path_type.is_symlink

        def simulated_is_symlink(path):
            return path == artifact_path or original_is_symlink(path)

        monkeypatch.setattr(path_type, "is_symlink", simulated_is_symlink)

    with pytest.raises(ValueError, match="symbolic links"):
        ArrowDataset.open(
            root,
            expected_dataset_id=manifest.dataset_id,
            require_training_capabilities=False,
        )


def test_legacy_manifest_without_evidence_inventory_is_rejected(tmp_path) -> None:
    manifest = _write(tmp_path)
    document = json.loads(manifest.to_json())
    del document["payload"]["capabilities"]["evidence_artifacts"]

    with pytest.raises(ValueError, match="missing or unexpected"):
        DatasetManifest.from_json(json.dumps(document))


@pytest.mark.parametrize("field", ["source_manifest_sha256", "builder_sha256"])
def test_readiness_evidence_rejects_manifest_identity_drift(tmp_path, field) -> None:
    manifest = _write(tmp_path)
    changed = replace(manifest, **{field: "f" * 64})
    (tmp_path / "dataset.json").write_text(changed.to_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="readiness evidence"):
        ArrowDataset.open(
            tmp_path,
            expected_dataset_id=changed.dataset_id,
            require_training_capabilities=False,
        )


def test_ordered_provenance_inventory_rejects_rehashed_row_drift(tmp_path) -> None:
    manifest = _write(tmp_path)
    provenance_artifact = manifest.capabilities.evidence_artifacts[2]
    provenance_path = tmp_path / provenance_artifact.path
    lines = provenance_path.read_text(encoding="ascii").splitlines()
    changed_row = json.loads(lines[2])
    changed_row["provenance_sha256"] = "f" * 64
    lines[2] = _canonical_line(changed_row).decode("ascii").rstrip("\n")
    provenance_content = ("\n".join(lines) + "\n").encode("ascii")
    changed_manifest = _rewrite_provenance_artifact(tmp_path, manifest, provenance_content)

    with pytest.raises(ValueError, match="ordered provenance inventory differs"):
        ArrowDataset.open(
            tmp_path,
            expected_dataset_id=changed_manifest.dataset_id,
            require_training_capabilities=False,
        )


@pytest.mark.parametrize("binding_field", ["builder_sha256", "arrow_schema_sha256"])
def test_provenance_header_rejects_rehashed_binding_drift(tmp_path, binding_field) -> None:
    manifest = _write(tmp_path)
    provenance_artifact = manifest.capabilities.evidence_artifacts[2]
    provenance_path = tmp_path / provenance_artifact.path
    lines = provenance_path.read_text(encoding="ascii").splitlines()
    header = json.loads(lines[0])
    header["binding"][binding_field] = "f" * 64
    lines[0] = _canonical_line(header).decode("ascii").rstrip("\n")
    changed_manifest = _rewrite_provenance_artifact(
        tmp_path,
        manifest,
        ("\n".join(lines) + "\n").encode("ascii"),
    )

    with pytest.raises(ValueError, match="provenance evidence"):
        ArrowDataset.open(
            tmp_path,
            expected_dataset_id=changed_manifest.dataset_id,
            require_training_capabilities=False,
        )


def test_writer_and_open_enforce_single_expected_instrument(tmp_path) -> None:
    examples = _examples(3)
    mixed = replace(
        examples[2],
        metadata=replace(examples[2].metadata, instrument_id="ETH-USD"),
    )
    with pytest.raises(ValueError, match="manifest instrument_id"):
        write_engineered_fixture_dataset(
            tmp_path / "mixed",
            [examples[0], examples[1], mixed],
            source_manifest_sha256="d" * 64,
            builder_sha256="e" * 64,
            capabilities=_capabilities(),
        )

    root = tmp_path / "single"
    manifest = _write(root)
    with pytest.raises(ValueError, match="expected_instrument_id"):
        ArrowDataset.open(
            root,
            expected_dataset_id=manifest.dataset_id,
            expected_instrument_id="ETH-USD",
        )


def test_fixture_writer_emits_bounded_record_batches(tmp_path) -> None:
    manifest = write_engineered_fixture_dataset(
        tmp_path,
        _examples(9),
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=_capabilities(),
        rows_per_shard=7,
        rows_per_record_batch=2,
    )
    pa = dataset_module._require_pyarrow()
    observed: list[tuple[int, ...]] = []
    for shard in manifest.shards:
        path = tmp_path.joinpath(*shard.path.split("/"))
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            observed.append(
                tuple(
                    reader.get_batch(index).num_rows for index in range(reader.num_record_batches)
                )
            )

    assert observed == [(2, 2, 2, 1), (2,)]
    with pytest.raises(ValueError, match="rows_per_record_batch"):
        write_engineered_fixture_dataset(
            tmp_path / "unbounded",
            _examples(1),
            source_manifest_sha256="d" * 64,
            builder_sha256="e" * 64,
            capabilities=_capabilities(),
            rows_per_record_batch=dataset_module.MAX_RECORD_BATCH_ROWS + 1,
        )


def test_streaming_verifies_each_touched_shard_once_and_skips_untouched_shards(
    tmp_path, monkeypatch
) -> None:
    manifest = _write(tmp_path)
    dataset = ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)
    original = dataset_module._file_sha256
    hashed_shards: list[str] = []

    def tracked(path):
        if path.suffix in {".arrow", ".ipc"}:
            hashed_shards.append(path.name)
        return original(path)

    monkeypatch.setattr(dataset_module, "_file_sha256", tracked)
    selected = (dataset.example_ids[0], dataset.example_ids[1], dataset.example_ids[4])
    batches = list(
        dataset.iter_batches(
            selected,
            batch_size=1,
            partition="evaluation",
        )
    )

    assert tuple(batch.example_ids[0] for batch in batches) == selected
    assert hashed_shards == ["part-00000.arrow", "part-00002.arrow"]


def test_streaming_releases_record_batch_before_loading_another(tmp_path, monkeypatch) -> None:
    manifest = write_engineered_fixture_dataset(
        tmp_path,
        _examples(6),
        source_manifest_sha256="d" * 64,
        builder_sha256="e" * 64,
        capabilities=_capabilities(),
        rows_per_shard=6,
        rows_per_record_batch=2,
    )
    dataset = ArrowDataset.open(tmp_path, expected_dataset_id=manifest.dataset_id)
    original = dataset_module._open_verified_reader
    previous: weakref.ReferenceType | None = None
    opened_paths: list[str] = []
    loaded_record_batches: list[int] = []

    class TrackedReader:
        def __init__(self, reader) -> None:
            self._reader = reader

        @property
        def num_record_batches(self):
            return self._reader.num_record_batches

        def get_batch(self, index):
            nonlocal previous
            gc.collect()
            if previous is not None:
                assert previous() is None
            record_batch = self._reader.get_batch(index)
            previous = weakref.ref(record_batch)
            loaded_record_batches.append(index)
            return record_batch

    @contextmanager
    def tracked(root, shard, verified_manifest, **kwargs):
        opened_paths.append(shard.path)
        with original(root, shard, verified_manifest, **kwargs) as reader:
            yield TrackedReader(reader)

    monkeypatch.setattr(dataset_module, "_open_verified_reader", tracked)
    selected = (dataset.example_ids[0], dataset.example_ids[4])
    batches = list(
        dataset.iter_batches(
            selected,
            batch_size=1,
            partition="evaluation",
        )
    )

    assert tuple(batch.example_ids[0] for batch in batches) == selected
    assert opened_paths == [manifest.shards[0].path]
    assert loaded_record_batches == [0, 2]
    del batches
    gc.collect()
    assert previous is not None
    assert previous() is None


def test_writer_rejects_duplicate_ids_and_nonchronological_rows(tmp_path) -> None:
    examples = _examples(3)
    duplicate = replace(
        examples[2],
        metadata=replace(examples[2].metadata, example_id=examples[0].metadata.example_id),
    )
    with pytest.raises(ValueError, match="duplicate"):
        write_engineered_fixture_dataset(
            tmp_path / "duplicate",
            [examples[0], examples[1], duplicate],
            source_manifest_sha256="d" * 64,
            builder_sha256="e" * 64,
            capabilities=_capabilities(),
        )
    with pytest.raises(ValueError, match="chronological"):
        write_engineered_fixture_dataset(
            tmp_path / "reordered",
            list(reversed(examples)),
            source_manifest_sha256="d" * 64,
            builder_sha256="e" * 64,
            capabilities=_capabilities(),
        )


def test_writer_rejects_out_of_domain_unscaled_features(tmp_path) -> None:
    examples = _examples(1)
    examples[0].batch.l2[0, 0, 0, 0, 10] = 0.5

    with pytest.raises(ValueError, match="l2 feature 10 must be binary"):
        write_engineered_fixture_dataset(
            tmp_path / "invalid-feature",
            examples,
            source_manifest_sha256="d" * 64,
            builder_sha256="e" * 64,
            capabilities=_capabilities(),
        )


def test_shard_manifest_requires_exact_schema_hash() -> None:
    with pytest.raises(ValueError, match="Arrow schema"):
        DatasetManifest(
            instrument_id="BTC-USD",
            source_manifest_sha256="a" * 64,
            builder_sha256="b" * 64,
            capabilities=_capabilities(),
            row_count=1,
            arrow_schema_sha256="c" * 64,
            shards=(
                DatasetShard(
                    path="shards/one.arrow",
                    byte_size=1,
                    row_count=1,
                    sha256="d" * 64,
                    schema_sha256="e" * 64,
                    first_example_id="one",
                    last_example_id="one",
                    first_decision_ns=1,
                    last_decision_ns=1,
                ),
            ),
        )
