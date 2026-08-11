"""Canonical CHMT-Q experiment configuration and run identity."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from .config import ChmtQConfig
from .training.config import TrainingConfig

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FOLD_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_DEVICE = re.compile(r"(?:cpu|cuda(?::[0-9]+)?)")
_PRIMARY_METRICS = frozenset({"weighted_pinball", "one_step_poc_mae"})


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Strict, content-addressed configuration for one seed and one development fold."""

    dataset_id: str
    instrument_id: str
    split_manifest_sha256: str
    scaler_artifact_sha256: str
    source_tree_sha256: str
    dependency_lock_sha256: str
    fold_id: str
    seed: int
    model: ChmtQConfig
    training: TrainingConfig
    local_research_authorization_sha256: str | None = None
    device: str = "cpu"
    attention_logit_threshold: float = 80.0
    max_epochs: int = 50
    early_stop_patience: int = 8
    checkpoint_every_epochs: int = 1
    primary_metric: str = "weighted_pinball"
    minimum_improvement: float = 0.0
    schema_version: int = 2

    def __post_init__(self) -> None:
        for name in (
            "dataset_id",
            "split_manifest_sha256",
            "scaler_artifact_sha256",
            "source_tree_sha256",
            "dependency_lock_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            not isinstance(self.instrument_id, str)
            or not self.instrument_id
            or self.instrument_id != self.instrument_id.strip()
            or "\x00" in self.instrument_id
        ):
            raise ValueError("instrument_id must be a canonical non-empty string")
        if self.schema_version != 2:
            raise ValueError("unsupported experiment schema_version")
        if not isinstance(self.model, ChmtQConfig):
            raise TypeError("model must be a ChmtQConfig")
        if not isinstance(self.training, TrainingConfig):
            raise TypeError("training must be a TrainingConfig")
        if self.local_research_authorization_sha256 is not None:
            _require_sha256(
                self.local_research_authorization_sha256,
                "local_research_authorization_sha256",
            )
        if not _FOLD_ID.fullmatch(self.fold_id):
            raise ValueError("fold_id must be a short lowercase stable identifier")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.training.seed != self.seed:
            raise ValueError("experiment seed must equal TrainingConfig.seed")
        if not isinstance(self.device, str) or _DEVICE.fullmatch(self.device) is None:
            raise ValueError("device must be cpu, cuda, or cuda:<index>")
        if (
            not math.isfinite(self.attention_logit_threshold)
            or self.attention_logit_threshold <= 0.0
        ):
            raise ValueError("attention_logit_threshold must be finite and positive")
        for name in ("max_epochs", "early_stop_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.checkpoint_every_epochs, bool)
            or not isinstance(self.checkpoint_every_epochs, int)
            or self.checkpoint_every_epochs != 1
        ):
            raise ValueError(
                "checkpoint_every_epochs is fixed at 1 for exact epoch-boundary resume"
            )
        if self.primary_metric not in _PRIMARY_METRICS:
            raise ValueError("primary_metric must be weighted_pinball or one_step_poc_mae")
        if not math.isfinite(self.minimum_improvement) or self.minimum_improvement < 0.0:
            raise ValueError("minimum_improvement must be finite and non-negative")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "instrument_id": self.instrument_id,
            "split_manifest_sha256": self.split_manifest_sha256,
            "scaler_artifact_sha256": self.scaler_artifact_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "fold_id": self.fold_id,
            "seed": self.seed,
            "model": asdict(self.model),
            "training": asdict(self.training),
            "local_research_authorization_sha256": (self.local_research_authorization_sha256),
            "device": self.device,
            "attention_logit_threshold": self.attention_logit_threshold,
            "max_epochs": self.max_epochs,
            "early_stop_patience": self.early_stop_patience,
            "checkpoint_every_epochs": self.checkpoint_every_epochs,
            "primary_metric": self.primary_metric,
            "minimum_improvement": self.minimum_improvement,
        }

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        return self.config_sha256

    @property
    def local_research_only(self) -> bool:
        return self.local_research_authorization_sha256 is not None

    def checkpoint_identity(self) -> dict[str, str]:
        return {
            "config_sha256": self.config_sha256,
            "dataset_id": self.dataset_id,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "fold_id": self.fold_id,
            "instrument_id": self.instrument_id,
            "local_research_authorization_sha256": (
                self.local_research_authorization_sha256 or "strict-dataset-capabilities"
            ),
            "run_id": self.run_id,
            "scaler_artifact_sha256": self.scaler_artifact_sha256,
            "seed": str(self.seed),
            "source_tree_sha256": self.source_tree_sha256,
            "split_manifest_sha256": self.split_manifest_sha256,
        }

    def to_json(self) -> str:
        return _canonical_json({"config_sha256": self.config_sha256, "payload": self.payload()})

    @classmethod
    def from_json(cls, value: str) -> ExperimentConfig:
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("experiment configuration is not valid JSON") from error
        if not isinstance(document, dict) or set(document) != {
            "config_sha256",
            "payload",
        }:
            raise ValueError("experiment JSON must contain config_sha256 and payload")
        payload = document["payload"]
        if not isinstance(payload, dict):
            raise ValueError("experiment payload must be an object")
        expected = {
            "schema_version",
            "dataset_id",
            "instrument_id",
            "split_manifest_sha256",
            "scaler_artifact_sha256",
            "source_tree_sha256",
            "dependency_lock_sha256",
            "fold_id",
            "seed",
            "model",
            "training",
            "local_research_authorization_sha256",
            "device",
            "attention_logit_threshold",
            "max_epochs",
            "early_stop_patience",
            "checkpoint_every_epochs",
            "primary_metric",
            "minimum_improvement",
        }
        if set(payload) != expected:
            raise ValueError("experiment payload has missing or unexpected fields")
        model_values = _strict_dataclass_mapping(payload["model"], ChmtQConfig, "model")
        training_values = _strict_dataclass_mapping(payload["training"], TrainingConfig, "training")
        config = cls(
            schema_version=_require_int(payload["schema_version"], "schema_version"),
            dataset_id=_require_str(payload["dataset_id"], "dataset_id"),
            instrument_id=_require_str(payload["instrument_id"], "instrument_id"),
            split_manifest_sha256=_require_str(
                payload["split_manifest_sha256"], "split_manifest_sha256"
            ),
            scaler_artifact_sha256=_require_str(
                payload["scaler_artifact_sha256"], "scaler_artifact_sha256"
            ),
            source_tree_sha256=_require_str(payload["source_tree_sha256"], "source_tree_sha256"),
            dependency_lock_sha256=_require_str(
                payload["dependency_lock_sha256"], "dependency_lock_sha256"
            ),
            fold_id=_require_str(payload["fold_id"], "fold_id"),
            seed=_require_int(payload["seed"], "seed"),
            model=ChmtQConfig(**model_values),
            training=TrainingConfig.from_mapping(training_values),
            local_research_authorization_sha256=_require_optional_sha256(
                payload["local_research_authorization_sha256"],
                "local_research_authorization_sha256",
            ),
            device=_require_str(payload["device"], "device"),
            attention_logit_threshold=_require_number(
                payload["attention_logit_threshold"],
                "attention_logit_threshold",
            ),
            max_epochs=_require_int(payload["max_epochs"], "max_epochs"),
            early_stop_patience=_require_int(payload["early_stop_patience"], "early_stop_patience"),
            checkpoint_every_epochs=_require_int(
                payload["checkpoint_every_epochs"], "checkpoint_every_epochs"
            ),
            primary_metric=_require_str(payload["primary_metric"], "primary_metric"),
            minimum_improvement=_require_number(
                payload["minimum_improvement"], "minimum_improvement"
            ),
        )
        if document["config_sha256"] != config.config_sha256:
            raise ValueError("experiment config SHA-256 does not match its payload")
        if value not in {config.to_json(), config.to_json() + "\n"}:
            raise ValueError("experiment JSON is not in canonical form")
        return config

    def write_new(self, path: str | Path) -> Path:
        """Atomically write a new immutable config without replacing an existing file."""

        target = Path(path).resolve()
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"experiment config already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(f".{target.name}.staging-{uuid.uuid4().hex}")
        try:
            staging.write_text(self.to_json() + "\n", encoding="utf-8", newline="\n")
            try:
                os.link(staging, target)
            except FileExistsError as error:
                raise FileExistsError(f"experiment config already exists: {target}") from error
        finally:
            if staging.exists() or staging.is_symlink():
                staging.unlink()
        return target


def source_tree_sha256(root: str | Path, *, include_tests: bool = True) -> str:
    """Hash reviewed first-party source inputs without absolute-path identity.

    The frozen identity deliberately excludes virtual environments, build output,
    caches, and other files outside the package, project metadata, and optional
    first-party tests.
    """

    if not isinstance(include_tests, bool):
        raise TypeError("include_tests must be a boolean")
    base = Path(root).resolve()
    pyproject = base / "pyproject.toml"
    package = base / "daxalgo_numeric_transformer"
    if not base.is_dir() or base.is_symlink() or not pyproject.is_file() or pyproject.is_symlink():
        raise ValueError("source root must contain pyproject.toml")
    if not package.is_dir() or package.is_symlink():
        raise ValueError("source root must contain daxalgo_numeric_transformer")
    package_files = tuple(package.rglob("*.py"))
    if not package_files:
        raise ValueError("daxalgo_numeric_transformer must contain Python source")
    files = [pyproject, *package_files]
    tests = base / "tests"
    if include_tests and tests.is_dir():
        files.extend(tests.rglob("*.py"))
    files.sort(key=lambda item: item.relative_to(base).as_posix())
    if any(not path.is_file() or path.is_symlink() for path in files):
        raise ValueError("reviewed source inputs must be regular non-symlink files")
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_dataclass_mapping(value: Any, owner: type[Any], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be an object")
    known = {field.name for field in fields(owner)}
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"unknown {name} option(s): {', '.join(unknown)}")
    missing = sorted(known - set(value))
    if missing:
        raise ValueError(f"missing {name} option(s): {', '.join(missing)}")

    defaults = asdict(owner())
    result: dict[str, Any] = {}
    for key, item in value.items():
        default = defaults[key]
        qualified = f"{name}.{key}"
        if isinstance(default, bool):
            if not isinstance(item, bool):
                raise TypeError(f"{qualified} must be a boolean")
            result[key] = item
        elif isinstance(default, int):
            result[key] = _require_int(item, qualified)
        elif isinstance(default, float):
            result[key] = _require_number(item, qualified)
        elif isinstance(default, tuple):
            if not isinstance(item, list) or len(item) != len(default):
                raise TypeError(f"{qualified} must be a fixed-length array")
            if all(isinstance(part, float) for part in default):
                result[key] = tuple(
                    _require_number(part, f"{qualified}[{index}]")
                    for index, part in enumerate(item)
                )
            else:  # pragma: no cover - no current non-float tuple fields
                raise TypeError(f"{qualified} uses an unsupported tuple type")
        else:  # pragma: no cover - guards future configuration field types
            raise TypeError(f"{qualified} uses an unsupported configuration type")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")


def _require_optional_sha256(value: Any, name: str) -> str | None:
    if value is None:
        return None
    _require_sha256(value, name)
    return value


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value
