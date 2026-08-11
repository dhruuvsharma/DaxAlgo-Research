from __future__ import annotations

from dataclasses import replace

import pytest

from daxalgo_numeric_transformer.chmt_q.dependencies import (
    TRAINING_DEPENDENCY_ROOTS,
    RuntimeDependencyLock,
    capture_runtime_dependency_lock,
    verify_runtime_dependency_lock,
    write_runtime_dependency_lock,
)


def test_runtime_dependency_lock_round_trips_and_matches_active_environment(tmp_path) -> None:
    path = tmp_path / "training.lock.json"
    captured = write_runtime_dependency_lock(path)

    assert captured.roots == TRAINING_DEPENDENCY_ROOTS
    assert RuntimeDependencyLock.from_json(captured.to_json()) == captured
    assert verify_runtime_dependency_lock(path)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_runtime_dependency_lock(path)


def test_runtime_dependency_lock_rejects_arbitrary_or_mismatched_content(tmp_path) -> None:
    arbitrary = tmp_path / "requirements.lock"
    arbitrary.write_text("torch==2.5.1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="valid JSON"):
        verify_runtime_dependency_lock(arbitrary)

    captured = capture_runtime_dependency_lock()
    mismatched = replace(captured, python_version="0.0.0")
    path = tmp_path / "mismatched.lock.json"
    path.write_text(mismatched.to_json() + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="active environment differs"):
        verify_runtime_dependency_lock(path)
