from __future__ import annotations

from pathlib import Path

import pytest
import torch

import daxalgo_numeric_transformer.chmt_q.scaling as scaling_module
from daxalgo_numeric_transformer.chmt_q import ChmtQBatch
from daxalgo_numeric_transformer.chmt_q.output import ChmtQOutput
from daxalgo_numeric_transformer.chmt_q.scaling import (
    FIT_CAVEAT,
    PARTITION_BINDING,
    RobustScalerArtifact,
    fit_robust_scaler,
)

SOURCE_HASH = "a" * 64


def _batch(offset: float = 0.0, batch_size: int = 2) -> ChmtQBatch:
    l2_level_mask = torch.ones(batch_size, 240, 2, 10, dtype=torch.bool)
    bars = torch.zeros(batch_size, 64, 16)
    l1 = torch.zeros(batch_size, 240, 16)
    l2 = torch.zeros(batch_size, 240, 2, 10, 12)
    quality = torch.zeros(batch_size, 16)
    targets = torch.zeros(batch_size, 5, 5)
    for feature in range(16):
        bars[..., feature] = offset + feature + torch.arange(batch_size).view(-1, 1)
        l1[..., feature] = offset + 2 * feature + torch.arange(batch_size).view(-1, 1)
    for feature in range(12):
        l2[..., feature] = offset + 3 * feature + torch.arange(batch_size).view(-1, 1, 1, 1)
    for horizon in range(5):
        for target in range(5):
            if target == 4:
                targets[:, horizon, target] = torch.linspace(-0.5, 0.5, batch_size)
            else:
                targets[:, horizon, target] = (
                    offset + 10 * horizon + target + torch.arange(batch_size)
                )
    return ChmtQBatch(
        bars=bars,
        bars_mask=torch.ones(batch_size, 64, dtype=torch.bool),
        l1=l1,
        l1_mask=torch.ones(batch_size, 240, dtype=torch.bool),
        l2=l2,
        l2_level_mask=l2_level_mask,
        l2_snapshot_mask=l2_level_mask.any(dim=(-1, -2)),
        quality=quality,
        targets=targets,
        target_mask=torch.ones(batch_size, 5, 5, dtype=torch.bool),
        presence_labels=torch.zeros(batch_size, 5, 2),
        presence_mask=torch.ones(batch_size, 5, 2, dtype=torch.bool),
    )


def test_scaler_is_fit_only_from_explicitly_supplied_batches() -> None:
    supplied = _batch(offset=0.0)
    held_out = _batch(offset=10_000.0)

    first = fit_robust_scaler([supplied], source_batch_set_hash=SOURCE_HASH)
    held_out.bars.add_(1_000_000.0)
    held_out.targets.add_(1_000_000.0)  # type: ignore[union-attr]
    second = fit_robust_scaler([supplied], source_batch_set_hash=SOURCE_HASH)

    assert first == second
    assert first.artifact_sha256 == second.artifact_sha256


def test_disk_spooled_statistics_match_known_exact_median_and_mad() -> None:
    first = _batch(offset=-3.0)
    second = _batch(offset=10.0, batch_size=3)
    first.bars_mask.zero_()
    second.bars_mask.zero_()
    first.bars_mask[:, 0] = True
    second.bars_mask[:, 0] = True
    identities = {
        "bars": tuple(range(1, 16)),
        "l1": tuple(range(16)),
        "l2": tuple(range(12)),
        "quality": tuple(range(16)),
        "targets": tuple(range(5)),
    }
    artifact = fit_robust_scaler(
        [first, second],
        source_batch_set_hash=SOURCE_HASH,
        identity_indices=identities,
        max_chunk_values=3,
    )

    assert artifact.bars.centers[0] == pytest.approx(10.0)
    assert artifact.bars.scales[0] == pytest.approx(1.4826 * 2.0)


def test_chunk_size_does_not_change_statistics_or_artifact_identity() -> None:
    first = _batch(offset=-2.0)
    second = _batch(offset=7.0, batch_size=3)
    first.bars_mask.zero_()
    second.bars_mask.zero_()
    first.bars_mask[:, 0] = True
    second.bars_mask[:, 0] = True
    batches = [first, second]
    identities = {
        "bars": tuple(range(1, 16)),
        "l1": tuple(range(16)),
        "l2": tuple(range(12)),
        "quality": tuple(range(16)),
        "targets": tuple(range(5)),
    }

    tiny_chunks = fit_robust_scaler(
        batches,
        source_batch_set_hash=SOURCE_HASH,
        identity_indices=identities,
        max_chunk_values=2,
    )
    larger_chunks = fit_robust_scaler(
        batches,
        source_batch_set_hash=SOURCE_HASH,
        identity_indices=identities,
        max_chunk_values=4096,
    )

    assert tiny_chunks == larger_chunks
    assert tiny_chunks.artifact_sha256 == larger_chunks.artifact_sha256


def test_fitter_streams_batches_and_caps_every_work_chunk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_chunk_sizes: list[int] = []
    original_write = scaling_module._write_float64_chunk

    def recording_write(handle: object, values: object) -> None:
        observed_chunk_sizes.append(len(values))  # type: ignore[arg-type]
        original_write(handle, values)  # type: ignore[arg-type]

    class StreamingBatches:
        def __init__(self) -> None:
            first = _batch(offset=0.0)
            second = _batch(offset=2.0)
            first.bars_mask.zero_()
            second.bars_mask.zero_()
            first.bars_mask[:, 0] = True
            second.bars_mask[:, 0] = True
            self._batches = iter((first, second))

        def __iter__(self) -> StreamingBatches:
            return self

        def __next__(self) -> ChmtQBatch:
            return next(self._batches)

        def __length_hint__(self) -> int:
            raise AssertionError("batch iterable must not be materialized")

    monkeypatch.setattr(scaling_module, "_write_float64_chunk", recording_write)
    artifact = fit_robust_scaler(
        StreamingBatches(),
        source_batch_set_hash=SOURCE_HASH,
        identity_indices={
            "bars": tuple(range(1, 16)),
            "l1": tuple(range(16)),
            "l2": tuple(range(12)),
            "quality": tuple(range(16)),
            "targets": tuple(range(5)),
        },
        spool_directory=tmp_path,
        max_chunk_values=5,
    )

    assert artifact.bars.valid_counts[0] == 4
    assert observed_chunk_sizes
    assert max(observed_chunk_sizes) <= 5
    assert list(tmp_path.iterdir()) == []


def test_masked_garbage_does_not_affect_statistics() -> None:
    clean = _batch()
    garbage = _batch()
    garbage.bars_mask[:, 0] = False
    garbage.bars[:, 0] = 1.0e20
    garbage.l1_mask[:, 0] = False
    garbage.l1[:, 0] = -1.0e20
    garbage.l2_level_mask[:, 0] = False
    garbage.l2_snapshot_mask[:, 0] = False
    garbage.l2[:, 0] = 1.0e20
    assert garbage.target_mask is not None and garbage.targets is not None
    garbage.target_mask[:, 0, 0] = False
    garbage.targets[:, 0, 0] = 1.0e20

    clean.bars_mask[:, 0] = False
    clean.l1_mask[:, 0] = False
    clean.l2_level_mask[:, 0] = False
    clean.l2_snapshot_mask[:, 0] = False
    assert clean.target_mask is not None
    clean.target_mask[:, 0, 0] = False

    clean_artifact = fit_robust_scaler([clean], source_batch_set_hash=SOURCE_HASH)
    garbage_artifact = fit_robust_scaler([garbage], source_batch_set_hash=SOURCE_HASH)

    assert clean_artifact == garbage_artifact


def test_degenerate_mad_uses_minimum_scale_and_identity_fields_bypass() -> None:
    batch = _batch(batch_size=1)
    artifact = fit_robust_scaler(
        [batch], source_batch_set_hash=SOURCE_HASH, minimum_scale=0.125
    )

    assert artifact.bars.scales[0] == pytest.approx(0.125)
    assert artifact.targets.scales[0] == pytest.approx(0.125)
    assert artifact.bars.centers[9] == 0.0
    assert artifact.bars.scales[9] == 1.0
    for horizon in range(5):
        delta_index = horizon * 5 + 4
        assert artifact.targets.centers[delta_index] == 0.0
        assert artifact.targets.scales[delta_index] == 1.0


def test_normalization_clips_and_zeroes_invalid_values_afterward() -> None:
    training = _batch()
    artifact = fit_robust_scaler([training], source_batch_set_hash=SOURCE_HASH)
    evaluation = _batch(offset=100_000.0)
    evaluation.bars_mask[:, 0] = False
    evaluation.bars[:, 0] = -1.0e20
    evaluation.l1_mask[:, 0] = False
    evaluation.l1[:, 0] = 1.0e20
    evaluation.l2_level_mask[:, 0] = False
    evaluation.l2_snapshot_mask[:, 0] = False
    evaluation.l2[:, 0] = 1.0e20
    assert evaluation.target_mask is not None and evaluation.targets is not None
    evaluation.target_mask[:, 0, 0] = False
    evaluation.targets[:, 0, 0] = torch.nan

    normalized = artifact.normalize_batch(evaluation)

    assert normalized.bars[:, 0].count_nonzero() == 0
    assert normalized.l1[:, 0].count_nonzero() == 0
    assert normalized.l2[:, 0].count_nonzero() == 0
    assert normalized.targets is not None
    assert normalized.targets[:, 0, 0].count_nonzero() == 0
    assert normalized.bars.abs().max() <= 8.0
    assert normalized.l1.abs().max() <= 8.0
    assert normalized.l2.abs().max() <= 8.0
    assert normalized.targets.abs().max() <= 8.0


def test_scaler_json_round_trip_preserves_canonical_identity_and_caveat() -> None:
    artifact = fit_robust_scaler([_batch()], source_batch_set_hash=SOURCE_HASH)

    serialized = artifact.to_json()
    restored = RobustScalerArtifact.from_json(serialized)

    assert restored == artifact
    assert restored.to_json() == serialized
    assert restored.artifact_sha256 == artifact.artifact_sha256
    assert restored.fit_caveat == FIT_CAVEAT
    assert restored.partition_binding == PARTITION_BINDING
    assert "fit_partition_role" not in restored.payload()
    assert "train" not in restored.partition_binding
    with pytest.raises(ValueError, match="does not match"):
        RobustScalerArtifact.from_json(serialized.replace(SOURCE_HASH, "b" * 64))


def test_valid_values_are_rejected_when_training_statistics_are_unavailable() -> None:
    training = _batch()
    training.l1_mask.zero_()
    artifact = fit_robust_scaler([training], source_batch_set_hash=SOURCE_HASH)
    evaluation = _batch()

    non_identity = set(range(16)) - set(artifact.l1.identity_indices)
    assert not any(artifact.l1.fitted[index] for index in non_identity)
    with pytest.raises(ValueError, match="without training statistics"):
        artifact.normalize_batch(evaluation)


def test_target_inverse_restores_robustly_scaled_values() -> None:
    artifact = fit_robust_scaler([_batch()], source_batch_set_hash=SOURCE_HASH)
    batch = _batch(offset=0.25)
    normalized = artifact.normalize_batch(batch)

    assert batch.targets is not None and normalized.targets is not None
    restored = artifact.denormalize_targets(normalized.targets)
    torch.testing.assert_close(restored, batch.targets)

    quantiles = normalized.targets.unsqueeze(-1).expand(-1, -1, -1, 3).clone()
    output = ChmtQOutput(quantiles, torch.zeros(batch.batch_size, 5, 2))
    physical = artifact.denormalize_output(output)
    torch.testing.assert_close(physical.median, batch.targets)


def test_target_statistics_are_independent_per_horizon_and_target() -> None:
    artifact = fit_robust_scaler([_batch()], source_batch_set_hash=SOURCE_HASH)
    centers = torch.tensor(artifact.targets.centers).reshape(5, 5)

    assert centers[0, 0] != centers[1, 0]
    assert centers[2, 1] != centers[2, 2]
    assert torch.all(centers[:, 4] == 0.0)  # bounded delta fraction is explicit identity
