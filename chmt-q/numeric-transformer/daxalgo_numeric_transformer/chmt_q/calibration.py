"""Chronological split-conformal interval expansion for CHMT-Q marginals."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .config import ChmtQConfig
from .contracts import ChmtQBatch
from .output import ChmtQOutput


@dataclass(frozen=True, slots=True)
class ConformalCalibration:
    """Frozen per-horizon/target nonconformity expansion from calibration data only."""

    offsets: tuple[float, ...]
    counts: tuple[int, ...]
    alpha: float
    source_calibration_set_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        cfg = ChmtQConfig.reference()
        expected = cfg.horizons * cfg.targets
        if self.schema_version != 1:
            raise ValueError("unsupported conformal calibration schema_version")
        if len(self.offsets) != expected or len(self.counts) != expected:
            raise ValueError("conformal arrays must have one value per horizon and target")
        if any(not math.isfinite(value) or value < 0.0 for value in self.offsets):
            raise ValueError("conformal offsets must be finite and non-negative")
        if any(isinstance(value, bool) or value <= 0 for value in self.counts):
            raise ValueError("every conformal cell requires calibration observations")
        if not math.isfinite(self.alpha) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        if re.fullmatch(r"[0-9a-f]{64}", self.source_calibration_set_sha256) is None:
            raise ValueError("source_calibration_set_sha256 must be lowercase SHA-256")

    @classmethod
    def fit(
        cls,
        examples: Iterable[tuple[ChmtQOutput, ChmtQBatch]],
        *,
        source_calibration_set_sha256: str,
        alpha: float = 0.2,
        config: ChmtQConfig | None = None,
    ) -> ConformalCalibration:
        cfg = config or ChmtQConfig.reference()
        if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be finite and in (0, 1)")
        scores: list[list[float]] = [
            [] for _ in range(cfg.horizons * cfg.targets)
        ]
        seen = False
        for output, batch in examples:
            seen = True
            output.validate(cfg)
            batch.validate(cfg, require_targets=True)
            assert batch.targets is not None and batch.target_mask is not None
            if output.quantiles.shape[0] != batch.batch_size:
                raise ValueError("calibration output and batch sizes differ")
            targets = batch.targets.detach().cpu().numpy()
            mask = batch.target_mask.detach().cpu().numpy()
            lower = output.q10.detach().cpu().numpy()
            upper = output.q90.detach().cpu().numpy()
            for horizon in range(cfg.horizons):
                for target in range(cfg.targets):
                    valid = mask[:, horizon, target]
                    y = targets[:, horizon, target][valid]
                    cell_lower = lower[:, horizon, target][valid]
                    cell_upper = upper[:, horizon, target][valid]
                    cell_scores = np.maximum(
                        0.0, np.maximum(cell_lower - y, y - cell_upper)
                    )
                    scores[horizon * cfg.targets + target].extend(
                        float(value) for value in cell_scores
                    )
        if not seen:
            raise ValueError("at least one calibration batch is required")
        if any(not values for values in scores):
            raise ValueError("every conformal cell requires calibration observations")
        offsets = tuple(_finite_sample_quantile(values, alpha) for values in scores)
        return cls(
            offsets=offsets,
            counts=tuple(len(values) for values in scores),
            alpha=alpha,
            source_calibration_set_sha256=source_calibration_set_sha256,
        )

    def apply(
        self, output: ChmtQOutput, config: ChmtQConfig | None = None
    ) -> ChmtQOutput:
        cfg = config or ChmtQConfig.reference()
        output.validate(cfg)
        offset = output.quantiles.new_tensor(self.offsets).reshape(
            1, cfg.horizons, cfg.targets
        )
        lower = output.q10 - offset
        median = output.median
        upper = output.q90 + offset
        delta_index = cfg.delta_target_index
        lower[..., delta_index] = lower[..., delta_index].clamp(-1.0, 1.0)
        upper[..., delta_index] = upper[..., delta_index].clamp(-1.0, 1.0)
        calibrated = torch.stack((lower, median, upper), dim=-1)
        return ChmtQOutput(calibrated, output.presence_logits).validate(cfg)

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "alpha": self.alpha,
            "offsets": list(self.offsets),
            "counts": list(self.counts),
            "source_calibration_set_sha256": self.source_calibration_set_sha256,
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
    def from_json(cls, value: str) -> ConformalCalibration:
        try:
            document = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("conformal calibration is not valid JSON") from error
        if not isinstance(document, dict) or set(document) != {
            "artifact_sha256",
            "payload",
        }:
            raise ValueError("conformal calibration JSON has an invalid field set")
        payload = document["payload"]
        expected = {
            "schema_version",
            "source_calibration_set_sha256",
            "alpha",
            "offsets",
            "counts",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("conformal calibration payload has an invalid field set")
        artifact = cls(
            schema_version=_integer(payload["schema_version"], "schema_version"),
            source_calibration_set_sha256=_string(
                payload["source_calibration_set_sha256"],
                "source_calibration_set_sha256",
            ),
            alpha=_number(payload["alpha"], "alpha"),
            offsets=_float_tuple(payload["offsets"], "offsets"),
            counts=_integer_tuple(payload["counts"], "counts"),
        )
        if document["artifact_sha256"] != artifact.artifact_sha256:
            raise ValueError("conformal calibration SHA-256 does not match its payload")
        if value not in {artifact.to_json(), artifact.to_json() + "\n"}:
            raise ValueError("conformal calibration JSON is not in canonical form")
        return artifact


def _finite_sample_quantile(values: list[float], alpha: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    rank = min(int(math.ceil((ordered.size + 1) * (1.0 - alpha))), ordered.size)
    return float(ordered[rank - 1])


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


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def _float_tuple(value: Any, name: str) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return tuple(_number(item, f"{name} entry") for item in value)


def _integer_tuple(value: Any, name: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    return tuple(_integer(item, f"{name} entry") for item in value)
