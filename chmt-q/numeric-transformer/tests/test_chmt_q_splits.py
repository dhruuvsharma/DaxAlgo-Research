from __future__ import annotations

from dataclasses import replace

import pytest

from daxalgo_numeric_transformer.chmt_q.splits import (
    ROLE_ORDER,
    ExampleDependency,
    SplitManifest,
    build_chronological_split,
)

SOURCE_HASH = "c" * 64
ROLE_COUNTS = {
    "development": 1,
    "train": 1,
    "validation": 1,
    "calibration": 1,
    "outer_test": 1,
}


def _example(
    example_id: str,
    session_id: str,
    start: int,
    decision: int,
    end: int,
) -> ExampleDependency:
    return ExampleDependency(example_id, session_id, decision, start, end)


def _non_overlapping_examples() -> list[ExampleDependency]:
    return [
        _example("dev", "s0", 0, 100, 200),
        _example("train", "s1", 1_000, 1_100, 1_200),
        _example("validation", "s2", 2_000, 2_100, 2_200),
        _example("calibration", "s3", 3_000, 3_100, 3_200),
        _example("test", "s4", 4_000, 4_100, 4_200),
    ]


def test_split_is_deterministic_session_grouped_and_role_complete() -> None:
    examples = _non_overlapping_examples()

    first = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
    )
    second = build_chronological_split(
        reversed(examples),
        role_session_counts=dict(reversed(tuple(ROLE_COUNTS.items()))),
        source_dataset_hash=SOURCE_HASH,
    )

    assert tuple(role.role for role in first.roles) == ROLE_ORDER
    assert first == second
    assert first.to_json() == second.to_json()
    assert first.artifact_sha256 == second.artifact_sha256
    assert [first.role(role).session_ids for role in ROLE_ORDER] == [
        ("s0",),
        ("s1",),
        ("s2",),
        ("s3",),
        ("s4",),
    ]


def test_exact_dependency_overlap_purges_only_the_conflicting_earlier_example() -> None:
    examples = [
        _example("dev", "s0", -200, -150, -100),
        _example("train-safe", "s1", 100, 150, 240),
        _example("train-overlap", "s1", 200, 250, 390),
        _example("validation", "s2", 350, 400, 520),
        _example("calibration", "s3", 600, 650, 720),
        _example("test", "s4", 800, 850, 920),
    ]

    manifest = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
    )

    assert manifest.role("train").example_ids == ("train-safe",)
    assert manifest.role("train").purged_example_ids == ("train-overlap",)
    record = next(item for item in manifest.purges if item.example_id == "train-overlap")
    assert record.against_role == "validation"
    assert record.reason == "dependency_overlap"
    manifest.validate(examples)


def test_configurable_embargo_removes_near_boundary_without_calling_it_overlap() -> None:
    examples = [
        _example("dev", "s0", -200, -150, -100),
        _example("train-safe", "s1", 100, 150, 250),
        _example("train-near", "s1", 200, 260, 340),
        _example("validation", "s2", 400, 450, 520),
        _example("calibration", "s3", 700, 750, 820),
        _example("test", "s4", 1_000, 1_050, 1_120),
    ]

    without_embargo = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
    )
    with_embargo = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
        embargo_ns=100,
    )

    assert without_embargo.role("train").example_ids == ("train-safe", "train-near")
    assert with_embargo.role("train").example_ids == ("train-safe",)
    record = next(item for item in with_embargo.purges if item.example_id == "train-near")
    assert record.reason == "embargo"


def test_manifest_round_trip_preserves_canonical_identity() -> None:
    manifest = build_chronological_split(
        _non_overlapping_examples(),
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
        embargo_ns=50,
    )

    serialized = manifest.to_json()
    restored = SplitManifest.from_json(serialized)

    assert restored == manifest
    assert restored.to_json() == serialized
    assert restored.artifact_sha256 == manifest.artifact_sha256
    with pytest.raises(ValueError, match="does not match"):
        SplitManifest.from_json(serialized.replace(SOURCE_HASH, "d" * 64))


def test_validation_detects_dependency_mutation_that_reintroduces_leakage() -> None:
    examples = _non_overlapping_examples()
    manifest = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
    )
    mutated = [
        replace(item, dependency_end_ns=2_050) if item.example_id == "train" else item
        for item in examples
    ]

    with pytest.raises(ValueError, match="retained leakage"):
        manifest.validate(mutated)


def test_train_decisions_precede_validation_calibration_and_outer_test() -> None:
    examples = _non_overlapping_examples()
    manifest = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
    )
    by_id = {item.example_id: item for item in examples}
    train_end = max(by_id[item].decision_ns for item in manifest.role("train").example_ids)

    for role in ("validation", "calibration", "outer_test"):
        role_start = min(by_id[item].decision_ns for item in manifest.role(role).example_ids)
        assert train_end < role_start


def test_an_example_purged_from_a_later_role_does_not_over_purge_train() -> None:
    examples = [
        _example("dev", "s0", -500, -450, -400),
        _example("train", "s1", 100, 150, 300),
        _example("validation-purged", "s2", 250, 350, 550),
        _example("validation-safe", "s2", 350, 400, 450),
        _example("calibration", "s3", 500, 600, 700),
        _example("test", "s4", 900, 950, 1_000),
    ]

    manifest = build_chronological_split(
        examples,
        role_session_counts=ROLE_COUNTS,
        source_dataset_hash=SOURCE_HASH,
    )

    assert manifest.role("train").example_ids == ("train",)
    assert manifest.role("validation").example_ids == ("validation-safe",)
    assert manifest.role("validation").purged_example_ids == ("validation-purged",)


def test_interleaved_session_decisions_are_rejected() -> None:
    examples = [
        _example("dev-a", "s0", 0, 100, 150),
        _example("dev-b", "s0", 400, 500, 550),
        _example("train", "s1", 200, 300, 350),
        _example("validation", "s2", 600, 650, 700),
        _example("calibration", "s3", 800, 850, 900),
        _example("test", "s4", 1_000, 1_050, 1_100),
    ]

    with pytest.raises(ValueError, match="interleave"):
        build_chronological_split(
            examples,
            role_session_counts=ROLE_COUNTS,
            source_dataset_hash=SOURCE_HASH,
        )
