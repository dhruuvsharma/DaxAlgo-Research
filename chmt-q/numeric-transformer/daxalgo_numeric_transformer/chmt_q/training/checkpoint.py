"""Verified, immutable local checkpoints for CHMT-Q development runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

_SCHEMA_VERSION = 1
_REQUIRED_FILES = frozenset(
    {"weights.npz", "optimizer.pt", "rng.npz", "state.json", "hashes.json"}
)


class CheckpointVerificationError(RuntimeError):
    """Raised when an immutable checkpoint fails identity or content checks."""


def save_checkpoint(
    root: str | Path,
    *,
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    run_identity: dict[str, str],
    trainer_state: dict[str, Any] | None = None,
) -> Path:
    """Atomically publish a new checkpoint directory without overwriting an old one."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")
    identity = _validated_identity(run_identity)
    state = _json_value(trainer_state or {}, "trainer_state")
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / f"step-{step:08d}"
    if target.exists():
        raise FileExistsError(f"checkpoint already exists: {target}")
    staging = root_path / f".{target.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        _save_weights(staging / "weights.npz", model)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        _save_rng(staging / "rng.npz")
        _write_json(
            staging / "state.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "step": step,
                "run_identity": identity,
                "trainer_state": state,
                "python_random_state": _lists(random.getstate()),
            },
        )
        hashes = {
            path.name: _sha256(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _write_json(staging / "hashes.json", {"sha256": hashes})
        os.replace(staging, target)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target


def load_checkpoint(
    checkpoint: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    expected_identity: dict[str, str],
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Verify every file, then restore model, optimizer, and optional RNG state."""

    path = Path(checkpoint).resolve()
    if not path.is_dir():
        raise CheckpointVerificationError(f"checkpoint directory is missing: {path}")
    actual_files = {item.name for item in path.iterdir() if item.is_file()}
    if actual_files != _REQUIRED_FILES:
        raise CheckpointVerificationError(
            f"checkpoint file set mismatch: expected {sorted(_REQUIRED_FILES)}, "
            f"got {sorted(actual_files)}"
        )
    hashes = _read_json(path / "hashes.json")
    expected_hashes = hashes.get("sha256")
    if not isinstance(expected_hashes, dict):
        raise CheckpointVerificationError("hashes.json has no sha256 mapping")
    hashed_names = actual_files - {"hashes.json"}
    if set(expected_hashes) != hashed_names:
        raise CheckpointVerificationError("checkpoint hash inventory does not match files")
    for name in sorted(hashed_names):
        if expected_hashes[name] != _sha256(path / name):
            raise CheckpointVerificationError(f"checkpoint hash mismatch: {name}")

    state = _read_json(path / "state.json")
    if state.get("schema_version") != _SCHEMA_VERSION:
        raise CheckpointVerificationError("unsupported checkpoint schema version")
    expected = _validated_identity(expected_identity)
    if state.get("run_identity") != expected:
        raise CheckpointVerificationError("checkpoint run identity does not match")

    _load_weights(path / "weights.npz", model)
    try:
        optimizer_state = torch.load(
            path / "optimizer.pt", map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise CheckpointVerificationError("optimizer state could not be loaded safely") from error
    optimizer.load_state_dict(optimizer_state)
    if restore_rng:
        _load_rng(path / "rng.npz")
        python_state = state.get("python_random_state")
        if not isinstance(python_state, list):
            raise CheckpointVerificationError("Python RNG state is missing")
        random.setstate(_tuples(python_state))
    trainer_state = state.get("trainer_state")
    if not isinstance(trainer_state, dict):
        raise CheckpointVerificationError("trainer_state must be an object")
    return {"step": state["step"], "trainer_state": trainer_state}


def _save_weights(path: Path, model: nn.Module) -> None:
    arrays: dict[str, np.ndarray] = {}
    for name, value in model.state_dict().items():
        tensor = value.detach().to(device="cpu")
        if tensor.dtype == torch.bfloat16:
            raise TypeError(f"checkpoint does not support bfloat16 state tensor {name!r}")
        arrays[name] = tensor.numpy()
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def _load_weights(path: Path, model: nn.Module) -> None:
    reference = model.state_dict()
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(reference):
            raise CheckpointVerificationError("model state keys do not match checkpoint")
        restored: dict[str, torch.Tensor] = {}
        for name, current in reference.items():
            array = archive[name]
            if tuple(array.shape) != tuple(current.shape):
                raise CheckpointVerificationError(f"shape mismatch for model state {name!r}")
            restored[name] = torch.from_numpy(array.copy()).to(
                device=current.device, dtype=current.dtype
            )
    model.load_state_dict(restored, strict=True)


def _save_rng(path: Path) -> None:
    algorithm, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    arrays: dict[str, np.ndarray] = {
        "torch_cpu": torch.get_rng_state().cpu().numpy(),
        "numpy_algorithm": np.asarray(algorithm),
        "numpy_keys": keys,
        "numpy_position": np.asarray(position, dtype=np.int64),
        "numpy_has_gauss": np.asarray(has_gauss, dtype=np.int64),
        "numpy_cached_gaussian": np.asarray(cached_gaussian, dtype=np.float64),
    }
    if torch.cuda.is_available():
        for index, value in enumerate(torch.cuda.get_rng_state_all()):
            arrays[f"torch_cuda_{index}"] = value.cpu().numpy()
    with path.open("wb") as stream:
        np.savez(stream, **arrays)


def _load_rng(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        torch.set_rng_state(torch.from_numpy(archive["torch_cpu"].copy()))
        np.random.set_state(
            (
                str(archive["numpy_algorithm"].item()),
                archive["numpy_keys"].copy(),
                int(archive["numpy_position"].item()),
                int(archive["numpy_has_gauss"].item()),
                float(archive["numpy_cached_gaussian"].item()),
            )
        )
        cuda_names = sorted(name for name in archive.files if name.startswith("torch_cuda_"))
        if cuda_names:
            if not torch.cuda.is_available():
                raise CheckpointVerificationError(
                    "checkpoint contains CUDA RNG state but CUDA is unavailable"
                )
            torch.cuda.set_rng_state_all(
                [torch.from_numpy(archive[name].copy()) for name in cuda_names]
            )


def _validated_identity(value: dict[str, str]) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("run_identity must be a non-empty string mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise ValueError("run_identity keys must be non-empty strings")
    if any(not isinstance(item, str) or not item for item in value.values()):
        raise ValueError("run_identity values must be non-empty strings")
    return dict(sorted(value.items()))


def _json_value(value: Any, name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain only finite JSON values") from error
    result = json.loads(encoded)
    if not isinstance(result, dict):
        raise ValueError(f"{name} must be an object")
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))
    path.write_text(encoded + "\n", encoding="utf-8", newline="\n")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CheckpointVerificationError(f"invalid JSON checkpoint file: {path.name}") from error
    if not isinstance(value, dict):
        raise CheckpointVerificationError(f"checkpoint JSON must be an object: {path.name}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_lists(item) for item in value]
    if isinstance(value, list):
        return [_lists(item) for item in value]
    return value


def _tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuples(item) for item in value)
    return value
