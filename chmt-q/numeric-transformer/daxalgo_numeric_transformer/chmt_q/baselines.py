"""Simple mask-correct baselines fitted on the same CHMT-Q targets."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ChmtQConfig
from .contracts import ChmtQBatch
from .output import ChmtQOutput


@dataclass(frozen=True, slots=True)
class EmpiricalBaseline:
    """Training-only empirical marginal quantiles and smoothed presence rates."""

    quantiles: tuple[float, ...]
    presence_logits: tuple[float, ...]
    target_counts: tuple[int, ...]
    presence_counts: tuple[int, ...]
    source_example_set_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        cfg = ChmtQConfig.reference()
        if self.schema_version != 1:
            raise ValueError("unsupported empirical baseline schema_version")
        expected_targets = cfg.horizons * cfg.targets
        expected_quantiles = expected_targets * cfg.quantiles
        expected_presence = cfg.horizons * cfg.presence_targets
        if len(self.quantiles) != expected_quantiles:
            raise ValueError("empirical baseline quantile size is invalid")
        if len(self.presence_logits) != expected_presence:
            raise ValueError("empirical baseline presence size is invalid")
        if len(self.target_counts) != expected_targets:
            raise ValueError("empirical baseline target-count size is invalid")
        if len(self.presence_counts) != expected_presence:
            raise ValueError("empirical baseline presence-count size is invalid")
        if any(not math.isfinite(value) for value in self.quantiles):
            raise ValueError("empirical baseline quantiles must be finite")
        if any(not math.isfinite(value) for value in self.presence_logits):
            raise ValueError("empirical baseline presence logits must be finite")
        if any(value <= 0 for value in (*self.target_counts, *self.presence_counts)):
            raise ValueError("every empirical baseline cell requires training observations")
        if (
            not isinstance(self.source_example_set_sha256, str)
            or len(self.source_example_set_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.source_example_set_sha256)
        ):
            raise ValueError("source_example_set_sha256 must be lowercase SHA-256")

    @classmethod
    def fit(
        cls,
        batches: Iterable[ChmtQBatch],
        *,
        source_example_set_sha256: str,
        config: ChmtQConfig | None = None,
    ) -> EmpiricalBaseline:
        cfg = config or ChmtQConfig.reference()
        target_cell_count = cfg.horizons * cfg.targets
        presence_cell_count = cfg.horizons * cfg.presence_targets
        presence_counts = [0 for _ in range(presence_cell_count)]
        presence_positives = [0 for _ in range(presence_cell_count)]
        seen = False
        with _DiskBackedTargetStatistics(target_cell_count) as statistics:
            for batch in batches:
                seen = True
                batch.validate(cfg, require_targets=True, require_presence=True)
                assert batch.targets is not None and batch.target_mask is not None
                assert batch.presence_labels is not None
                assert batch.presence_mask is not None
                targets = batch.targets.detach().cpu().numpy()
                target_mask = batch.target_mask.detach().cpu().numpy()
                presence = batch.presence_labels.detach().cpu().numpy()
                presence_mask = batch.presence_mask.detach().cpu().numpy()
                for horizon in range(cfg.horizons):
                    for target in range(cfg.targets):
                        values = targets[:, horizon, target][
                            target_mask[:, horizon, target]
                        ]
                        statistics.add(
                            horizon * cfg.targets + target,
                            (float(value) for value in values),
                        )
                    for target in range(cfg.presence_targets):
                        values = presence[:, horizon, target][
                            presence_mask[:, horizon, target]
                        ]
                        cell = horizon * cfg.presence_targets + target
                        presence_counts[cell] += int(values.size)
                        presence_positives[cell] += sum(
                            int(float(value)) for value in values
                        )
                statistics.commit_batch()
            if not seen:
                raise ValueError("at least one training batch is required")
            if any(count <= 0 for count in (*statistics.counts, *presence_counts)):
                raise ValueError(
                    "every target and presence cell requires training observations"
                )
            quantiles = statistics.quantiles(cfg.quantile_levels)
            target_counts = statistics.counts

        logits: list[float] = []
        for positives, count in zip(
            presence_positives, presence_counts, strict=True
        ):
            probability = (positives + 0.5) / (count + 1.0)
            logits.append(math.log(probability / (1.0 - probability)))
        return cls(
            quantiles=quantiles,
            presence_logits=tuple(logits),
            target_counts=target_counts,
            presence_counts=tuple(presence_counts),
            source_example_set_sha256=source_example_set_sha256,
        )

    def predict(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        config: ChmtQConfig | None = None,
    ) -> ChmtQOutput:
        cfg = config or ChmtQConfig.reference()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        quantiles = torch.tensor(
            self.quantiles, dtype=torch.float32, device=device
        ).reshape(1, cfg.horizons, cfg.targets, cfg.quantiles)
        presence = torch.tensor(
            self.presence_logits, dtype=torch.float32, device=device
        ).reshape(1, cfg.horizons, cfg.presence_targets)
        return ChmtQOutput(
            quantiles.expand(batch_size, -1, -1, -1).clone(),
            presence.expand(batch_size, -1, -1).clone(),
        ).validate(cfg)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_example_set_sha256": self.source_example_set_sha256,
            "quantiles": list(self.quantiles),
            "presence_logits": list(self.presence_logits),
            "target_counts": list(self.target_counts),
            "presence_counts": list(self.presence_counts),
        }

    @property
    def artifact_sha256(self) -> str:
        encoded = _canonical_json(self.payload())
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json(
            {"artifact_sha256": self.artifact_sha256, "payload": self.payload()}
        )

    @classmethod
    def from_json(cls, value: str) -> EmpiricalBaseline:
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("empirical baseline is not valid JSON") from error
        if not isinstance(document, dict) or set(document) != {
            "artifact_sha256",
            "payload",
        }:
            raise ValueError("empirical baseline JSON has an invalid field set")
        payload = document["payload"]
        expected = {
            "schema_version",
            "source_example_set_sha256",
            "quantiles",
            "presence_logits",
            "target_counts",
            "presence_counts",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("empirical baseline payload has an invalid field set")
        baseline = cls(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            source_example_set_sha256=_string(
                payload["source_example_set_sha256"], "source_example_set_sha256"
            ),
            quantiles=_float_tuple(payload["quantiles"], "quantiles"),
            presence_logits=_float_tuple(
                payload["presence_logits"], "presence_logits"
            ),
            target_counts=_integer_tuple(payload["target_counts"], "target_counts"),
            presence_counts=_integer_tuple(
                payload["presence_counts"], "presence_counts"
            ),
        )
        if document["artifact_sha256"] != baseline.artifact_sha256:
            raise ValueError("empirical baseline SHA-256 does not match its payload")
        if value not in {baseline.to_json(), baseline.to_json() + "\n"}:
            raise ValueError("empirical baseline JSON is not in canonical form")
        return baseline


@dataclass(frozen=True, slots=True)
class ZeroResidualBaseline:
    """Persistence-equivalent zero residuals with train-only presence marginals."""

    presence_logits: tuple[float, ...]
    source_example_set_sha256: str

    def __post_init__(self) -> None:
        cfg = ChmtQConfig.reference()
        if len(self.presence_logits) != cfg.horizons * cfg.presence_targets:
            raise ValueError("zero-residual presence size is invalid")
        if any(not math.isfinite(value) for value in self.presence_logits):
            raise ValueError("zero-residual presence logits must be finite")
        if (
            not isinstance(self.source_example_set_sha256, str)
            or len(self.source_example_set_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in self.source_example_set_sha256)
        ):
            raise ValueError("source_example_set_sha256 must be lowercase SHA-256")

    @classmethod
    def from_empirical(cls, baseline: EmpiricalBaseline) -> ZeroResidualBaseline:
        if not isinstance(baseline, EmpiricalBaseline):
            raise TypeError("baseline must be an EmpiricalBaseline")
        return cls(
            presence_logits=baseline.presence_logits,
            source_example_set_sha256=baseline.source_example_set_sha256,
        )

    def predict(
        self,
        batch_size: int,
        *,
        device: torch.device | str = "cpu",
        config: ChmtQConfig | None = None,
    ) -> ChmtQOutput:
        cfg = config or ChmtQConfig.reference()
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        quantiles = torch.zeros(
            batch_size,
            cfg.horizons,
            cfg.targets,
            cfg.quantiles,
            dtype=torch.float32,
            device=device,
        )
        presence = torch.tensor(
            self.presence_logits, dtype=torch.float32, device=device
        ).reshape(1, cfg.horizons, cfg.presence_targets)
        return ChmtQOutput(
            quantiles,
            presence.expand(batch_size, -1, -1).clone(),
        ).validate(cfg)


class _DiskBackedTargetStatistics:
    """Exact ordered target statistics with fixed-size memory and disk spooling."""

    def __init__(self, cell_count: int) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="chmtq-empirical-baseline-"
        )
        database = Path(self._temporary_directory.name) / "targets.sqlite3"
        try:
            self._connection = sqlite3.connect(database)
            self._connection.execute("PRAGMA cache_size = -2048")
            self._connection.execute("PRAGMA mmap_size = 0")
            self._connection.execute("PRAGMA journal_mode = OFF")
            self._connection.execute("PRAGMA synchronous = OFF")
            self._connection.execute("PRAGMA temp_store = FILE")
            self._connection.execute(
                "CREATE TABLE target_values (cell INTEGER NOT NULL, value REAL NOT NULL)"
            )
            self._connection.execute(
                "CREATE INDEX target_values_cell_value "
                "ON target_values (cell, value)"
            )
        except Exception:
            self._temporary_directory.cleanup()
            raise
        self._counts = [0 for _ in range(cell_count)]
        self._closed = False

    @property
    def counts(self) -> tuple[int, ...]:
        return tuple(self._counts)

    def __enter__(self) -> _DiskBackedTargetStatistics:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def add(self, cell: int, values: Iterable[float]) -> None:
        if cell < 0 or cell >= len(self._counts):
            raise IndexError("target statistics cell is out of range")
        count = 0

        def rows() -> Iterable[tuple[int, float]]:
            nonlocal count
            for value in values:
                converted = float(value)
                if not math.isfinite(converted):
                    raise ValueError("target statistics require finite values")
                count += 1
                yield cell, converted

        self._connection.executemany(
            "INSERT INTO target_values (cell, value) VALUES (?, ?)", rows()
        )
        self._counts[cell] += count

    def commit_batch(self) -> None:
        self._connection.commit()

    def quantiles(self, levels: tuple[float, ...]) -> tuple[float, ...]:
        self._connection.commit()
        result: list[float] = []
        for cell, count in enumerate(self._counts):
            if count <= 0:
                raise ValueError("target statistics cell has no observations")
            result.extend(
                _exact_linear_quantile(self._connection, cell, count, level)
                for level in levels
            )
        return tuple(result)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._connection.close()
        self._temporary_directory.cleanup()


def _exact_linear_quantile(
    connection: sqlite3.Connection,
    cell: int,
    count: int,
    level: float,
) -> float:
    virtual_index = (count - 1) * level
    lower_rank = math.floor(virtual_index)
    upper_rank = math.ceil(virtual_index)
    lower = _ordered_target_value(connection, cell, lower_rank)
    if lower_rank == upper_rank:
        return lower
    upper = _ordered_target_value(connection, cell, upper_rank)
    fraction = virtual_index - lower_rank
    difference = upper - lower
    # Match NumPy's stable linear interpolation branch so existing artifact
    # values remain bit-for-bit identical without retaining the sample array.
    if fraction >= 0.5:
        return upper - difference * (1.0 - fraction)
    return lower + difference * fraction


def _ordered_target_value(
    connection: sqlite3.Connection, cell: int, rank: int
) -> float:
    row = connection.execute(
        "SELECT value FROM target_values INDEXED BY target_values_cell_value "
        "WHERE cell = ? ORDER BY value LIMIT 1 OFFSET ?",
        (cell, rank),
    ).fetchone()
    if row is None:
        raise RuntimeError("target statistics rank is unavailable")
    return float(row[0])


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"{name} entries must be numbers")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValueError(f"{name} entries must be finite")
        result.append(converted)
    return tuple(result)


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return tuple(_integer(item, f"{name} entry") for item in value)
