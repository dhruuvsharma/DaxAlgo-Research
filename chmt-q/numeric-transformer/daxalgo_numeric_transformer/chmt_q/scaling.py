"""Exact robust scaling with deterministic disk spooling and bounded working memory."""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .config import ChmtQConfig
from .contracts import ChmtQBatch
from .output import ChmtQOutput

MAD_MULTIPLIER = 1.4826
FIT_METHOD = "exact-disk-spooled-radix-median-mad-v1"
PARTITION_BINDING = "caller-supplied-batches-unverified-v1"
FIT_CAVEAT = (
    "Exact fitting streams batches into temporary float64 disk spools and uses bounded-chunk "
    "radix order-statistic selection; spool location and chunk size do not affect statistics. "
    "ChmtQBatch has no example IDs, so this fitter cannot verify split-manifest membership."
)
DEFAULT_SCALER_CHUNK_VALUES = 131_072
MAX_SCALER_CHUNK_VALUES = 1_048_576
_FLOAT64_BYTES = 8
_SIGN_BIT = 1 << 63
_UINT64_MASK = (1 << 64) - 1

DEFAULT_IDENTITY_INDICES: dict[str, tuple[int, ...]] = {
    "bars": (9, 13, 14, 15),
    "l1": (5, 13, 14, 15),
    "l2": (4, 5, 6, 9, 10, 11),
    "quality": tuple(range(16)),
    "targets": (4,),
}


@dataclass(frozen=True, slots=True)
class ScaleBlock:
    """Flattened robust statistics for one named tensor's statistic axes."""

    name: str
    stat_shape: tuple[int, ...]
    centers: tuple[float, ...]
    scales: tuple[float, ...]
    fitted: tuple[bool, ...]
    valid_counts: tuple[int, ...]
    identity_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scale block name must not be empty")
        if not self.stat_shape or any(value <= 0 for value in self.stat_shape):
            raise ValueError("stat_shape must contain positive dimensions")
        size = math.prod(self.stat_shape)
        fields = {
            "centers": self.centers,
            "scales": self.scales,
            "fitted": self.fitted,
            "valid_counts": self.valid_counts,
        }
        for name, values in fields.items():
            if len(values) != size:
                raise ValueError(f"{name} must contain exactly {size} entries")
        if any(not math.isfinite(value) for value in self.centers):
            raise ValueError("centers must be finite")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.scales):
            raise ValueError("scales must be finite and positive")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.valid_counts
        ):
            raise ValueError("valid_counts must be non-negative integers")
        feature_count = self.stat_shape[-1]
        if tuple(sorted(set(self.identity_indices))) != self.identity_indices:
            raise ValueError("identity_indices must be sorted and unique")
        if any(index < 0 or index >= feature_count for index in self.identity_indices):
            raise ValueError("identity index is outside the final statistic axis")

        for flat_index in range(size):
            feature_index = flat_index % feature_count
            if feature_index in self.identity_indices and (
                self.centers[flat_index] != 0.0
                or self.scales[flat_index] != 1.0
                or not self.fitted[flat_index]
            ):
                raise ValueError("identity-scaled fields must use center=0, scale=1, fitted=true")

    def center_tensor(self, reference: Tensor) -> Tensor:
        return reference.new_tensor(self.centers).reshape(self.stat_shape)

    def scale_tensor(self, reference: Tensor) -> Tensor:
        return reference.new_tensor(self.scales).reshape(self.stat_shape)

    def fitted_tensor(self, reference: Tensor) -> Tensor:
        return torch.tensor(
            self.fitted, dtype=torch.bool, device=reference.device
        ).reshape(self.stat_shape)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stat_shape": list(self.stat_shape),
            "centers": list(self.centers),
            "scales": list(self.scales),
            "fitted": list(self.fitted),
            "valid_counts": list(self.valid_counts),
            "identity_indices": list(self.identity_indices),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ScaleBlock:
        expected = {
            "name",
            "stat_shape",
            "centers",
            "scales",
            "fitted",
            "valid_counts",
            "identity_indices",
        }
        if set(payload) != expected:
            raise ValueError("scale block payload has missing or unexpected fields")
        return cls(
            name=str(payload["name"]),
            stat_shape=tuple(int(value) for value in payload["stat_shape"]),
            centers=tuple(float(value) for value in payload["centers"]),
            scales=tuple(float(value) for value in payload["scales"]),
            fitted=tuple(bool(value) for value in payload["fitted"]),
            valid_counts=tuple(int(value) for value in payload["valid_counts"]),
            identity_indices=tuple(int(value) for value in payload["identity_indices"]),
        )


@dataclass(frozen=True, slots=True)
class RobustScalerArtifact:
    """Frozen statistics for caller-supplied batches with explicit provenance limits."""

    source_batch_set_hash: str
    minimum_scale: float
    bars: ScaleBlock
    l1: ScaleBlock
    l2: ScaleBlock
    quality: ScaleBlock
    targets: ScaleBlock
    schema_version: int = 1
    mad_multiplier: float = MAD_MULTIPLIER
    clip_min: float = -8.0
    clip_max: float = 8.0
    fit_method: str = FIT_METHOD
    fit_caveat: str = FIT_CAVEAT
    partition_binding: str = PARTITION_BINDING

    def __post_init__(self) -> None:
        _validate_sha256(self.source_batch_set_hash, "source_batch_set_hash")
        if self.schema_version != 1:
            raise ValueError("unsupported scaler schema_version")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be finite and positive")
        if self.mad_multiplier != MAD_MULTIPLIER:
            raise ValueError(f"mad_multiplier is fixed at {MAD_MULTIPLIER}")
        if (self.clip_min, self.clip_max) != (-8.0, 8.0):
            raise ValueError("CHMT-Q scaling clip is fixed at [-8, 8]")
        if self.fit_method != FIT_METHOD or self.fit_caveat != FIT_CAVEAT:
            raise ValueError("fit method and bounded-memory caveat are versioned constants")
        if self.partition_binding != PARTITION_BINDING:
            raise ValueError("scaler partition-binding limitation is a versioned constant")
        expected = {
            "bars": (16,),
            "l1": (16,),
            "l2": (12,),
            "quality": (16,),
            "targets": (5, 5),
        }
        for name, shape in expected.items():
            block = getattr(self, name)
            if block.name != name or block.stat_shape != shape:
                raise ValueError(f"{name} scale block must use statistic shape {shape}")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_batch_set_hash": self.source_batch_set_hash,
            "minimum_scale": self.minimum_scale,
            "mad_multiplier": self.mad_multiplier,
            "clip_min": self.clip_min,
            "clip_max": self.clip_max,
            "fit_method": self.fit_method,
            "fit_caveat": self.fit_caveat,
            "partition_binding": self.partition_binding,
            "blocks": {
                "bars": self.bars.to_payload(),
                "l1": self.l1.to_payload(),
                "l2": self.l2.to_payload(),
                "quality": self.quality.to_payload(),
                "targets": self.targets.to_payload(),
            },
        }

    @property
    def artifact_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return _canonical_json(
            {"artifact_sha256": self.artifact_sha256, "payload": self.payload()}
        )

    @classmethod
    def from_json(cls, value: str) -> RobustScalerArtifact:
        parsed = json.loads(value)
        if not isinstance(parsed, dict) or set(parsed) != {"artifact_sha256", "payload"}:
            raise ValueError("scaler JSON must contain artifact_sha256 and payload")
        payload = parsed["payload"]
        if not isinstance(payload, dict):
            raise ValueError("scaler payload must be an object")
        expected_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        if parsed["artifact_sha256"] != expected_hash:
            raise ValueError("scaler artifact SHA-256 does not match its payload")
        expected_keys = {
            "schema_version",
            "source_batch_set_hash",
            "minimum_scale",
            "mad_multiplier",
            "clip_min",
            "clip_max",
            "fit_method",
            "fit_caveat",
            "partition_binding",
            "blocks",
        }
        if set(payload) != expected_keys or not isinstance(payload["blocks"], dict):
            raise ValueError("scaler payload has missing or unexpected fields")
        blocks = payload["blocks"]
        if set(blocks) != {"bars", "l1", "l2", "quality", "targets"}:
            raise ValueError("scaler payload has an invalid block set")
        artifact = cls(
            schema_version=int(payload["schema_version"]),
            source_batch_set_hash=str(payload["source_batch_set_hash"]),
            minimum_scale=float(payload["minimum_scale"]),
            mad_multiplier=float(payload["mad_multiplier"]),
            clip_min=float(payload["clip_min"]),
            clip_max=float(payload["clip_max"]),
            fit_method=str(payload["fit_method"]),
            fit_caveat=str(payload["fit_caveat"]),
            partition_binding=str(payload["partition_binding"]),
            bars=ScaleBlock.from_payload(blocks["bars"]),
            l1=ScaleBlock.from_payload(blocks["l1"]),
            l2=ScaleBlock.from_payload(blocks["l2"]),
            quality=ScaleBlock.from_payload(blocks["quality"]),
            targets=ScaleBlock.from_payload(blocks["targets"]),
        )
        if artifact.artifact_sha256 != expected_hash:
            raise ValueError("reconstructed scaler identity differs from serialized payload")
        return artifact

    def normalize_batch(
        self, batch: ChmtQBatch, config: ChmtQConfig | None = None
    ) -> ChmtQBatch:
        """Apply frozen statistics and zero every invalid value after normalization."""

        cfg = config or ChmtQConfig()
        batch.validate(cfg)
        normalized_targets = None
        if batch.targets is not None and batch.target_mask is not None:
            normalized_targets = self._normalize(
                batch.targets, batch.target_mask, self.targets
            )
        normalized = ChmtQBatch(
            bars=self._normalize(
                batch.bars, batch.bars_mask.unsqueeze(-1).expand_as(batch.bars), self.bars
            ),
            bars_mask=batch.bars_mask,
            l1=self._normalize(
                batch.l1, batch.l1_mask.unsqueeze(-1).expand_as(batch.l1), self.l1
            ),
            l1_mask=batch.l1_mask,
            l2=self._normalize(
                batch.l2,
                batch.l2_level_mask.unsqueeze(-1).expand_as(batch.l2),
                self.l2,
            ),
            l2_level_mask=batch.l2_level_mask,
            l2_snapshot_mask=batch.l2_snapshot_mask,
            quality=self._normalize(
                batch.quality, torch.ones_like(batch.quality, dtype=torch.bool), self.quality
            ),
            targets=normalized_targets,
            target_mask=batch.target_mask,
            presence_labels=batch.presence_labels,
            presence_mask=batch.presence_mask,
        )
        return normalized.validate(cfg)

    def denormalize_targets(self, values: Tensor) -> Tensor:
        """Invert target scaling for a tensor whose trailing axes are ``[5,5]``."""

        if tuple(values.shape[-2:]) != self.targets.stat_shape:
            raise ValueError("target tensor must have trailing shape [5,5]")
        if not values.is_floating_point():
            raise TypeError("target tensor must use a floating-point dtype")
        center = self.targets.center_tensor(values)
        scale = self.targets.scale_tensor(values)
        return values * scale + center

    def denormalize_output(
        self, output: ChmtQOutput, config: ChmtQConfig | None = None
    ) -> ChmtQOutput:
        """Invert target scaling for every quantile while preserving presence logits."""

        cfg = config or ChmtQConfig.reference()
        output.validate(cfg)
        quantile_first = output.quantiles.permute(0, 3, 1, 2)
        physical = self.denormalize_targets(quantile_first).permute(0, 2, 3, 1)
        return ChmtQOutput(physical, output.presence_logits).validate(cfg)

    def _normalize(self, values: Tensor, valid: Tensor, block: ScaleBlock) -> Tensor:
        if valid.shape != values.shape or valid.dtype != torch.bool:
            raise ValueError("normalization validity mask must match values and be boolean")
        center = block.center_tensor(values)
        scale = block.scale_tensor(values)
        fitted = block.fitted_tensor(values)
        if bool((valid & ~fitted).any().item()):
            raise ValueError(f"{block.name} contains a valid value without training statistics")
        safe_values = torch.where(valid, values, center)
        normalized = torch.clamp(
            (safe_values - center) / scale, min=self.clip_min, max=self.clip_max
        )
        return torch.where(valid, normalized, torch.zeros_like(normalized))


def fit_robust_scaler(
    batches_to_fit: Iterable[ChmtQBatch],
    *,
    source_batch_set_hash: str,
    minimum_scale: float = 1.0e-6,
    config: ChmtQConfig | None = None,
    identity_indices: dict[str, tuple[int, ...]] | None = None,
    spool_directory: str | Path | None = None,
    max_chunk_values: int = DEFAULT_SCALER_CHUNK_VALUES,
) -> RobustScalerArtifact:
    """Fit exact statistics with bounded working memory and temporary disk spools.

    The caller must bind ``source_batch_set_hash`` and the supplied batches to exact
    example IDs from a validated split manifest before treating the result as train-only.
    ``ChmtQBatch`` itself carries no example IDs, so this function cannot enforce that
    association. Batches are consumed once and are never retained as a collection.

    ``max_chunk_values`` caps every NumPy work chunk and is deliberately absent from the
    artifact identity: exact statistics are invariant to this operational setting.
    """

    cfg = config or ChmtQConfig()
    _validate_sha256(source_batch_set_hash, "source_batch_set_hash")
    if not math.isfinite(minimum_scale) or minimum_scale <= 0.0:
        raise ValueError("minimum_scale must be finite and positive")
    _validate_max_chunk_values(max_chunk_values)
    identities = dict(DEFAULT_IDENTITY_INDICES)
    if identity_indices is not None:
        unknown = sorted(set(identity_indices) - set(identities))
        if unknown:
            raise ValueError(f"unknown identity-index block(s): {', '.join(unknown)}")
        identities.update(identity_indices)

    spool_parent = None if spool_directory is None else Path(spool_directory)
    if spool_parent is not None and not spool_parent.is_dir():
        raise ValueError("spool_directory must name an existing directory")

    with TemporaryDirectory(prefix="chmtq-scaler-", dir=spool_parent) as temporary:
        root = Path(temporary)
        spools = {
            "bars": _create_spool_cells(
                root, "bars", (cfg.bar_features,), identities["bars"]
            ),
            "l1": _create_spool_cells(root, "l1", (cfg.l1_features,), identities["l1"]),
            "l2": _create_spool_cells(root, "l2", (cfg.l2_features,), identities["l2"]),
            "quality": _create_spool_cells(
                root, "quality", (cfg.quality_features,), identities["quality"]
            ),
            "targets": _create_spool_cells(
                root,
                "targets",
                (cfg.horizons, cfg.targets),
                identities["targets"],
            ),
        }
        observed_batch = False
        try:
            for batch in batches_to_fit:
                observed_batch = True
                batch.validate(cfg)
                _append_rows(
                    spools["bars"], batch.bars, batch.bars_mask, max_chunk_values
                )
                _append_rows(spools["l1"], batch.l1, batch.l1_mask, max_chunk_values)
                _append_rows(
                    spools["l2"], batch.l2, batch.l2_level_mask, max_chunk_values
                )
                quality_mask = torch.ones(
                    batch.quality.shape[:-1], dtype=torch.bool, device=batch.device
                )
                _append_rows(
                    spools["quality"], batch.quality, quality_mask, max_chunk_values
                )
                if batch.targets is not None and batch.target_mask is not None:
                    _append_target_cells(
                        spools["targets"],
                        batch.targets,
                        batch.target_mask,
                        cfg,
                        max_chunk_values,
                    )
        finally:
            for cells in spools.values():
                for cell in cells:
                    cell.close()

        if not observed_batch:
            raise ValueError("at least one batch is required")

        return RobustScalerArtifact(
            source_batch_set_hash=source_batch_set_hash,
            minimum_scale=minimum_scale,
            bars=_fit_spooled_block(
                "bars",
                (cfg.bar_features,),
                spools["bars"],
                identities["bars"],
                minimum_scale,
                max_chunk_values,
                root,
            ),
            l1=_fit_spooled_block(
                "l1",
                (cfg.l1_features,),
                spools["l1"],
                identities["l1"],
                minimum_scale,
                max_chunk_values,
                root,
            ),
            l2=_fit_spooled_block(
                "l2",
                (cfg.l2_features,),
                spools["l2"],
                identities["l2"],
                minimum_scale,
                max_chunk_values,
                root,
            ),
            quality=_fit_spooled_block(
                "quality",
                (cfg.quality_features,),
                spools["quality"],
                identities["quality"],
                minimum_scale,
                max_chunk_values,
                root,
            ),
            targets=_fit_spooled_block(
                "targets",
                (cfg.horizons, cfg.targets),
                spools["targets"],
                identities["targets"],
                minimum_scale,
                max_chunk_values,
                root,
            ),
        )


@dataclass(slots=True)
class _SpoolCell:
    path: Path | None
    count: int = 0
    _handle: Any = None

    def append(self, values: np.ndarray) -> None:
        self.count += int(values.size)
        if self.path is None or values.size == 0:
            return
        if self._handle is None:
            self._handle = self.path.open("wb")
        _write_float64_chunk(self._handle, values)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _create_spool_cells(
    root: Path,
    name: str,
    stat_shape: tuple[int, ...],
    identity_indices: tuple[int, ...],
) -> list[_SpoolCell]:
    feature_count = stat_shape[-1]
    identities = frozenset(identity_indices)
    return [
        _SpoolCell(
            None
            if flat_index % feature_count in identities
            else root / f"{name}-{flat_index:04d}.f64"
        )
        for flat_index in range(math.prod(stat_shape))
    ]


def _append_rows(
    cells: list[_SpoolCell],
    values: Tensor,
    mask: Tensor,
    max_chunk_values: int,
) -> None:
    feature_count = values.shape[-1]
    array = values.detach().cpu().numpy().reshape(-1, feature_count)
    valid = mask.detach().cpu().numpy().reshape(-1)
    for start in range(0, array.shape[0], max_chunk_values):
        stop = min(start + max_chunk_values, array.shape[0])
        valid_chunk = valid[start:stop]
        valid_count = int(np.count_nonzero(valid_chunk))
        for index, cell in enumerate(cells):
            if cell.path is None:
                cell.count += valid_count
                continue
            selected = array[start:stop, index][valid_chunk]
            cell.append(selected)


def _append_target_cells(
    cells: list[_SpoolCell],
    values: Tensor,
    mask: Tensor,
    config: ChmtQConfig,
    max_chunk_values: int,
) -> None:
    cell_count = config.horizons * config.targets
    array = values.detach().cpu().numpy().reshape(-1, cell_count)
    valid = mask.detach().cpu().numpy().reshape(-1, cell_count)
    for start in range(0, array.shape[0], max_chunk_values):
        stop = min(start + max_chunk_values, array.shape[0])
        for index, cell in enumerate(cells):
            valid_chunk = valid[start:stop, index]
            if cell.path is None:
                cell.count += int(np.count_nonzero(valid_chunk))
                continue
            selected = array[start:stop, index][valid_chunk]
            cell.append(selected)


def _fit_spooled_block(
    name: str,
    stat_shape: tuple[int, ...],
    cells: list[_SpoolCell],
    identity_indices: tuple[int, ...],
    minimum_scale: float,
    max_chunk_values: int,
    root: Path,
) -> ScaleBlock:
    feature_count = stat_shape[-1]
    identities = tuple(sorted(set(identity_indices)))
    centers: list[float] = []
    scales: list[float] = []
    fitted: list[bool] = []
    valid_counts: list[int] = []
    deviation_path = root / f".{name}-deviation.f64"
    for flat_index, cell in enumerate(cells):
        feature_index = flat_index % feature_count
        count = cell.count
        valid_counts.append(count)
        if feature_index in identities:
            centers.append(0.0)
            scales.append(1.0)
            fitted.append(True)
            continue
        if count == 0:
            centers.append(0.0)
            scales.append(1.0)
            fitted.append(False)
            continue
        if cell.path is None:
            raise RuntimeError("non-identity statistic is missing its disk spool")
        median = _exact_spooled_median(cell.path, count, max_chunk_values)
        with deviation_path.open("wb") as deviation_handle:
            for chunk in _read_float64_chunks(cell.path, max_chunk_values):
                _write_float64_chunk(deviation_handle, np.abs(chunk - median))
        mad = _exact_spooled_median(deviation_path, count, max_chunk_values)
        deviation_path.unlink()
        centers.append(median)
        scales.append(max(MAD_MULTIPLIER * mad, minimum_scale))
        fitted.append(True)
    return ScaleBlock(
        name=name,
        stat_shape=stat_shape,
        centers=tuple(centers),
        scales=tuple(scales),
        fitted=tuple(fitted),
        valid_counts=tuple(valid_counts),
        identity_indices=identities,
    )


def _exact_spooled_median(path: Path, count: int, max_chunk_values: int) -> float:
    if count <= 0:
        raise ValueError("exact median requires at least one value")
    expected_size = count * _FLOAT64_BYTES
    if path.stat().st_size != expected_size:
        raise RuntimeError("float64 spool size differs from its valid-value count")
    if count <= max_chunk_values:
        values = np.fromfile(path, dtype="<f8", count=count)
        if values.size != count:
            raise RuntimeError("float64 spool ended before its valid-value count")
        median = float(np.median(values))
        return 0.0 if median == 0.0 else median
    ranks = ((count - 1) // 2, count // 2)
    lower, upper = _select_float64_order_statistics(
        path, ranks, max_chunk_values=max_chunk_values
    )
    median = float((np.float64(lower) + np.float64(upper)) / np.float64(2.0))
    return 0.0 if median == 0.0 else median


def _select_float64_order_statistics(
    path: Path,
    ranks: tuple[int, ...],
    *,
    max_chunk_values: int,
) -> tuple[float, ...]:
    """Select exact zero-based ranks in eight bounded scans of a float64 spool."""

    states = [{"prefix": 0, "rank": rank} for rank in ranks]
    for processed_bits in range(0, 64, 8):
        shift = 56 - processed_bits
        prefixes = {int(state["prefix"]) for state in states}
        histograms = {
            prefix: np.zeros(256, dtype=np.uint64) for prefix in prefixes
        }
        for chunk in _read_float64_chunks(path, max_chunk_values):
            keys = chunk.view(np.uint64)
            negative = (keys & np.uint64(_SIGN_BIT)) != 0
            keys[negative] = ~keys[negative]
            keys[~negative] ^= np.uint64(_SIGN_BIT)
            next_byte = ((keys >> np.uint64(shift)) & np.uint64(0xFF)).astype(
                np.intp,
                copy=False,
            )
            if processed_bits == 0:
                histograms[0] += np.bincount(next_byte, minlength=256).astype(
                    np.uint64,
                    copy=False,
                )
                continue
            observed_prefix = keys >> np.uint64(64 - processed_bits)
            for prefix, histogram in histograms.items():
                selected = next_byte[observed_prefix == prefix]
                histogram += np.bincount(selected, minlength=256).astype(
                    np.uint64,
                    copy=False,
                )

        for state in states:
            prefix = int(state["prefix"])
            rank = int(state["rank"])
            cumulative = np.cumsum(histograms[prefix], dtype=np.uint64)
            if not cumulative.size or rank >= int(cumulative[-1]):
                raise RuntimeError("order-statistic rank exceeds the selected spool prefix")
            bucket = int(np.searchsorted(cumulative, rank, side="right"))
            preceding = 0 if bucket == 0 else int(cumulative[bucket - 1])
            state["prefix"] = (prefix << 8) | bucket
            state["rank"] = rank - preceding

    return tuple(_sortable_key_to_float64(int(state["prefix"])) for state in states)


def _sortable_key_to_float64(key: int) -> float:
    bits = key ^ _SIGN_BIT if key & _SIGN_BIT else (~key) & _UINT64_MASK
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def _read_float64_chunks(path: Path, max_chunk_values: int) -> Iterable[np.ndarray]:
    with path.open("rb") as handle:
        while True:
            chunk = np.fromfile(handle, dtype="<f8", count=max_chunk_values)
            if chunk.size == 0:
                break
            yield chunk


def _write_float64_chunk(handle: Any, values: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(values, dtype="<f8")
    contiguous.tofile(handle)


def _validate_max_chunk_values(value: int) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value <= 0
        or value > MAX_SCALER_CHUNK_VALUES
    ):
        raise ValueError(
            f"max_chunk_values must be between 1 and {MAX_SCALER_CHUNK_VALUES}"
        )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _validate_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256")
