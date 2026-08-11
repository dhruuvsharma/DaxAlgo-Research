"""Canonical runtime dependency locks for reproducible CHMT-Q experiments."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

LOCK_SCHEMA_VERSION = 1
TRAINING_DEPENDENCY_ROOTS = ("numpy", "packaging", "pyarrow", "torch")

_PACKAGE_SEPARATOR = re.compile(r"[-_.]+")


@dataclass(frozen=True, slots=True)
class RuntimeDependencyLock:
    """Exact active dependency closure and interpreter/platform identity."""

    roots: tuple[str, ...]
    packages: tuple[tuple[str, str], ...]
    python_version: str
    implementation: str
    platform_system: str
    platform_machine: str
    schema_version: int = LOCK_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != LOCK_SCHEMA_VERSION:
            raise ValueError("unsupported dependency-lock schema_version")
        normalized_roots = tuple(sorted({_canonical_package_name(name) for name in self.roots}))
        if normalized_roots != self.roots or not self.roots:
            raise ValueError("dependency-lock roots must be sorted, unique, and canonical")
        normalized_packages = tuple(
            sorted((_canonical_package_name(name), version) for name, version in self.packages)
        )
        if normalized_packages != self.packages:
            raise ValueError("dependency-lock packages must be sorted and canonical")
        names = tuple(name for name, _ in self.packages)
        if len(names) != len(set(names)):
            raise ValueError("dependency-lock package names must be unique")
        if any(not version or version.strip() != version for _, version in self.packages):
            raise ValueError("dependency-lock package versions must be non-empty and trimmed")
        if any(root not in set(names) for root in self.roots):
            raise ValueError("dependency-lock closure is missing a required root package")
        for name, value in (
            ("python_version", self.python_version),
            ("implementation", self.implementation),
            ("platform_system", self.platform_system),
            ("platform_machine", self.platform_machine),
        ):
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty trimmed string")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "roots": list(self.roots),
            "packages": [
                {"name": name, "version": version} for name, version in self.packages
            ],
            "python_version": self.python_version,
            "implementation": self.implementation,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
        }

    def to_json(self) -> str:
        return _canonical_json(self.payload())

    @classmethod
    def from_json(cls, value: str) -> RuntimeDependencyLock:
        parsed = _parse_strict_json(value)
        payload = _require_object(
            parsed,
            "dependency lock",
            {
                "schema_version",
                "roots",
                "packages",
                "python_version",
                "implementation",
                "platform_system",
                "platform_machine",
            },
        )
        roots_value = payload["roots"]
        packages_value = payload["packages"]
        if not isinstance(roots_value, list) or not all(
            isinstance(item, str) for item in roots_value
        ):
            raise TypeError("dependency-lock roots must be a list of strings")
        if not isinstance(packages_value, list):
            raise TypeError("dependency-lock packages must be a list")
        packages: list[tuple[str, str]] = []
        for index, item in enumerate(packages_value):
            record = _require_object(item, f"packages[{index}]", {"name", "version"})
            packages.append(
                (
                    _require_string(record["name"], f"packages[{index}].name"),
                    _require_string(record["version"], f"packages[{index}].version"),
                )
            )
        lock = cls(
            schema_version=_require_integer(payload["schema_version"], "schema_version"),
            roots=tuple(roots_value),
            packages=tuple(packages),
            python_version=_require_string(payload["python_version"], "python_version"),
            implementation=_require_string(payload["implementation"], "implementation"),
            platform_system=_require_string(payload["platform_system"], "platform_system"),
            platform_machine=_require_string(payload["platform_machine"], "platform_machine"),
        )
        if value != lock.to_json():
            raise ValueError("dependency-lock JSON is not in canonical form")
        return lock

    def verify_current_environment(self) -> None:
        """Reject any interpreter, platform, package, or closure mismatch."""

        current = capture_runtime_dependency_lock(self.roots)
        if current != self:
            expected = dict(self.packages)
            observed = dict(current.packages)
            differences = sorted(
                name
                for name in set(expected) | set(observed)
                if expected.get(name) != observed.get(name)
            )
            detail = ", ".join(differences[:8]) or "interpreter/platform identity"
            raise ValueError(f"active environment differs from dependency lock: {detail}")


def capture_runtime_dependency_lock(
    roots: tuple[str, ...] = TRAINING_DEPENDENCY_ROOTS,
) -> RuntimeDependencyLock:
    """Capture the installed transitive closure for the selected training roots."""

    normalized_roots = tuple(sorted({_canonical_package_name(name) for name in roots}))
    if not normalized_roots:
        raise ValueError("at least one dependency root is required")
    pending = list(reversed(normalized_roots))
    observed: dict[str, str] = {}
    while pending:
        name = pending.pop()
        if name in observed:
            continue
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            raise ValueError(f"required distribution is not installed: {name}") from error
        canonical_name = _canonical_package_name(distribution.metadata["Name"] or name)
        if canonical_name != name and name in normalized_roots:
            raise ValueError(f"dependency root resolved to an unexpected distribution: {name}")
        observed[canonical_name] = distribution.version
        for requirement_text in distribution.requires or ():
            requirement = Requirement(requirement_text)
            if requirement.marker is not None and not requirement.marker.evaluate():
                continue
            dependency = _canonical_package_name(requirement.name)
            if dependency not in observed:
                pending.append(dependency)
    return RuntimeDependencyLock(
        roots=normalized_roots,
        packages=tuple(sorted(observed.items())),
        python_version=platform.python_version(),
        implementation=sys.implementation.name,
        platform_system=platform.system(),
        platform_machine=platform.machine(),
    )


def write_runtime_dependency_lock(path: str | Path) -> RuntimeDependencyLock:
    """Write a new canonical lock without overwriting an existing artifact."""

    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite dependency lock: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock = capture_runtime_dependency_lock()
    with destination.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(lock.to_json() + "\n")
    return lock


def verify_runtime_dependency_lock(path: str | Path) -> str:
    """Verify a regular canonical lock against the active environment and return its hash."""

    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"dependency lock must not be a symlink: {source}")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"dependency lock must be a regular file: {resolved}")
    raw = resolved.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("dependency lock must be UTF-8") from error
    if not text.endswith("\n") or text.count("\n") != 1:
        raise ValueError("dependency lock must contain one canonical JSON line")
    lock = RuntimeDependencyLock.from_json(text[:-1])
    lock.verify_current_environment()
    return hashlib.sha256(raw).hexdigest()


def _canonical_package_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("package name must be a non-empty string")
    return _PACKAGE_SEPARATOR.sub("-", value.strip()).lower()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _parse_strict_json(value: str) -> Any:
    if not isinstance(value, str):
        raise TypeError("dependency-lock JSON must be a string")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        return json.loads(
            value,
            object_pairs_hook=object_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is forbidden: {item}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError("dependency lock is not valid JSON") from error


def _require_object(value: Any, name: str, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} has missing or unexpected fields")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _require_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value
