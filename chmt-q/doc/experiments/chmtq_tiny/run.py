"""Train and evaluate the 282,838-parameter CHMT-Q Tiny pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import sys
import time
from typing import Callable, Iterator

import numpy as np
import pyarrow.ipc as ipc

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor
import torch.nn.functional as F

from model import ChmtQTiny, TinyConfig, ordered_quantiles, parameter_count


SHAPES = {
    "bars": (64, 16),
    "bars_mask": (64,),
    "l1": (240, 16),
    "l1_mask": (240,),
    "l2": (240, 2, 10, 12),
    "l2_level_mask": (240, 2, 10),
    "l2_snapshot_mask": (240,),
    "quality": (16,),
    "targets": (5, 5),
    "target_mask": (5, 5),
    "presence_labels": (5, 2),
    "presence_mask": (5, 2),
}
FLOAT_COLUMNS = {"bars", "l1", "l2", "quality", "targets", "presence_labels"}
MASK_COLUMNS = set(SHAPES) - FLOAT_COLUMNS
DATE_PATTERN = re.compile(r"(\d{8})T")
QUANTILE_LEVELS = np.array([0.1, 0.5, 0.9], dtype=np.float64)
HORIZON_WEIGHTS = np.array([1.0, 1 / math.sqrt(2), 1 / math.sqrt(3), 0.5, 1 / math.sqrt(5)])
TARGET_WEIGHTS = np.array([1.0, 1.0, 1.0, 0.5, 0.5])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def date_from_id(example_id: str) -> str:
    match = DATE_PATTERN.search(example_id)
    if not match:
        raise ValueError(f"unrecognized example ID: {example_id}")
    return match.group(1)


class Partition:
    def __init__(
        self,
        name: str,
        predicate: Callable[[str], bool],
        lower_id: str,
        upper_id: str,
        expected_count: int | None = None,
    ) -> None:
        self.name = name
        self.predicate = predicate
        self.lower_id = lower_id
        self.upper_id = upper_id
        self.expected_count = expected_count


class ArrowCorpus:
    def __init__(self, dataset_dir: Path, batch_size: int) -> None:
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        manifest_path = dataset_dir / "dataset.json"
        self.manifest_path = manifest_path
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.payload = self.manifest["payload"]
        self.shards = self.payload["shards"]

    def verify(self) -> dict:
        expected_dataset_id = canonical_payload_sha256(self.payload)
        if self.manifest["dataset_id"] != expected_dataset_id:
            raise ValueError("dataset manifest payload hash does not match dataset_id")
        if sum(shard["row_count"] for shard in self.shards) != self.payload["row_count"]:
            raise ValueError("shard row counts do not match dataset row_count")
        expected_schema = self.payload["arrow_schema_sha256"]
        total_bytes = 0
        for shard in self.shards:
            path = self.dataset_dir / shard["path"]
            if path.stat().st_size != shard["byte_size"]:
                raise ValueError(f"shard size mismatch: {path}")
            if sha256_file(path) != shard["sha256"]:
                raise ValueError(f"shard hash mismatch: {path}")
            with path.open("rb") as handle:
                schema_hash = hashlib.sha256(
                    ipc.open_file(handle).schema.serialize().to_pybytes()
                ).hexdigest()
            if schema_hash != expected_schema or schema_hash != shard["schema_sha256"]:
                raise ValueError(f"shard schema mismatch: {path}")
            total_bytes += shard["byte_size"]
        return {
            "verified_shards": len(self.shards),
            "verified_bytes": total_bytes,
            "row_count": self.payload["row_count"],
            "arrow_schema_sha256": expected_schema,
        }

    def relevant_shards(self, partition: Partition) -> list[dict]:
        return [
            shard
            for shard in self.shards
            if shard["last_example_id"] >= partition.lower_id
            and shard["first_example_id"] <= partition.upper_id
        ]

    @staticmethod
    def _fixed_array(record_batch, name: str) -> np.ndarray:
        array = record_batch.column(record_batch.schema.get_field_index(name))
        dtype = np.float32 if name in FLOAT_COLUMNS else np.bool_
        values = array.values.to_numpy(zero_copy_only=False).astype(dtype, copy=False)
        return values.reshape(len(array), *SHAPES[name])

    def batches(
        self,
        partition: Partition,
        *,
        epoch: int,
        seed: int,
        include_features: bool = True,
        shuffle: bool = False,
        limit_batches: int | None = None,
    ) -> Iterator[dict[str, Tensor | list[str]]]:
        rng = np.random.default_rng(seed + 1009 * epoch)
        shards = self.relevant_shards(partition)
        if shuffle:
            rng.shuffle(shards)
        yielded = 0
        for shard in shards:
            path = self.dataset_dir / shard["path"]
            with path.open("rb") as handle:
                reader = ipc.open_file(handle)
                record_indices = list(range(reader.num_record_batches))
                if shuffle:
                    rng.shuffle(record_indices)
                for record_index in record_indices:
                    record_batch = reader.get_batch(record_index)
                    ids = record_batch.column(record_batch.schema.get_field_index("example_id")).to_pylist()
                    selected = np.fromiter(
                        (index for index, example_id in enumerate(ids) if partition.predicate(example_id)),
                        dtype=np.int64,
                    )
                    if not selected.size:
                        continue
                    if shuffle:
                        rng.shuffle(selected)
                    columns = list(SHAPES) if include_features else [
                        "targets",
                        "target_mask",
                        "presence_labels",
                        "presence_mask",
                    ]
                    arrays = {name: self._fixed_array(record_batch, name) for name in columns}
                    for start in range(0, selected.size, self.batch_size):
                        indices = selected[start : start + self.batch_size]
                        batch: dict[str, Tensor | list[str]] = {
                            name: torch.from_numpy(np.ascontiguousarray(array[indices]))
                            for name, array in arrays.items()
                        }
                        batch["example_id"] = [ids[index] for index in indices]
                        yield batch
                        yielded += 1
                        if limit_batches is not None and yielded >= limit_batches:
                            return


class RobustScaler:
    def __init__(self, path: Path, device: torch.device) -> None:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if canonical_payload_sha256(artifact["payload"]) != artifact["artifact_sha256"]:
            raise ValueError("scaler artifact payload hash mismatch")
        self.artifact = artifact
        blocks = artifact["payload"]["inner_scaler"]["payload"]["blocks"]
        self.clip_min = float(artifact["payload"]["inner_scaler"]["payload"]["clip_min"])
        self.clip_max = float(artifact["payload"]["inner_scaler"]["payload"]["clip_max"])
        self.centers: dict[str, Tensor] = {}
        self.scales: dict[str, Tensor] = {}
        for name in ("bars", "l1", "l2", "quality", "targets"):
            shape = (5, 5) if name == "targets" else (-1,)
            self.centers[name] = torch.tensor(blocks[name]["centers"], device=device).reshape(shape)
            self.scales[name] = torch.tensor(blocks[name]["scales"], device=device).reshape(shape)

    def transform(self, name: str, values: Tensor, mask: Tensor | None = None) -> Tensor:
        normalized = ((values - self.centers[name]) / self.scales[name]).clamp(
            self.clip_min, self.clip_max
        )
        if mask is not None:
            normalized = normalized * mask.unsqueeze(-1) if normalized.ndim == mask.ndim + 1 else normalized * mask
        return normalized

    def inverse_targets(self, values: np.ndarray) -> np.ndarray:
        center = self.centers["targets"].detach().cpu().numpy()
        scale = self.scales["targets"].detach().cpu().numpy()
        return values * scale[..., None] + center[..., None]


def move_and_scale(batch: dict, scaler: RobustScaler, device: torch.device) -> dict[str, Tensor]:
    moved = {
        name: value.to(device, non_blocking=True)
        for name, value in batch.items()
        if name != "example_id"
    }
    moved["bars"] = scaler.transform("bars", moved["bars"], moved["bars_mask"])
    moved["l1"] = scaler.transform("l1", moved["l1"], moved["l1_mask"])
    moved["l2"] = scaler.transform("l2", moved["l2"], moved["l2_level_mask"])
    moved["quality"] = scaler.transform("quality", moved["quality"])
    moved["normalized_targets"] = scaler.transform("targets", moved["targets"], moved["target_mask"])
    return moved


def model_forward(model: ChmtQTiny, batch: dict[str, Tensor]):
    return model(
        batch["bars"],
        batch["bars_mask"],
        batch["l1"],
        batch["l1_mask"],
        batch["l2"],
        batch["l2_level_mask"],
        batch["l2_snapshot_mask"],
        batch["quality"],
    )


def base_loss(raw: Tensor, batch: dict[str, Tensor], presence_weight: float) -> tuple[Tensor, dict[str, float]]:
    prediction = ordered_quantiles(raw)
    target = batch["normalized_targets"].unsqueeze(-1)
    levels = torch.tensor((0.1, 0.5, 0.9), device=raw.device)
    error = target - prediction
    pinball = torch.maximum(levels * error, (levels - 1.0) * error)
    horizon = torch.tensor(HORIZON_WEIGHTS, device=raw.device, dtype=raw.dtype)[:, None, None]
    targets = torch.tensor(TARGET_WEIGHTS, device=raw.device, dtype=raw.dtype)[None, :, None]
    weights = batch["target_mask"].unsqueeze(-1) * horizon * targets
    quantile_loss = (pinball * weights).sum() / (3.0 * weights.sum()).clamp_min(1e-6)

    median_error = batch["normalized_targets"] - prediction[..., 1]
    huber = F.huber_loss(prediction[..., 1], batch["normalized_targets"], reduction="none", delta=1.0)
    median_weights = batch["target_mask"] * horizon.squeeze(-1) * targets.squeeze(-1)
    huber_loss = (huber * median_weights).sum() / median_weights.sum().clamp_min(1e-6)

    presence = F.binary_cross_entropy_with_logits(
        raw[..., 15:17], batch["presence_labels"], reduction="none"
    )
    presence_loss = (presence * batch["presence_mask"]).sum() / batch["presence_mask"].sum().clamp_min(1)
    total = quantile_loss + 0.25 * huber_loss + presence_weight * presence_loss
    return total, {
        "pinball": float(quantile_loss.detach()),
        "huber": float(huber_loss.detach()),
        "presence": float(presence_loss.detach()),
        "median_abs_error": float(
            (median_error.abs() * batch["target_mask"]).sum().detach()
            / batch["target_mask"].sum().clamp_min(1)
        ),
    }


def auxiliary_weight(progress: float) -> float:
    if progress <= 0.60:
        return 0.25
    if progress < 0.90:
        return 0.25 - (0.20 / 0.30) * (progress - 0.60)
    return 0.05


def fit_empirical_baseline(
    corpus: ArrowCorpus,
    partition: Partition,
    scaler: RobustScaler,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    values: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    presence_values: list[np.ndarray] = []
    presence_masks: list[np.ndarray] = []
    count = 0
    for batch in corpus.batches(partition, epoch=0, seed=seed, include_features=False):
        targets = batch["targets"].to(device)
        target_mask = batch["target_mask"].to(device)
        normalized = scaler.transform("targets", targets, target_mask)
        values.append(normalized.cpu().numpy())
        masks.append(target_mask.cpu().numpy())
        presence_values.append(batch["presence_labels"].numpy())
        presence_masks.append(batch["presence_mask"].numpy())
        count += len(batch["example_id"])
    y = np.concatenate(values)
    valid = np.concatenate(masks).astype(bool)
    baseline = np.zeros((5, 5, 3), dtype=np.float32)
    for horizon in range(5):
        for target in range(5):
            baseline[horizon, target] = np.quantile(y[:, horizon, target][valid[:, horizon, target]], QUANTILE_LEVELS)

    labels = np.concatenate(presence_values)
    label_mask = np.concatenate(presence_masks).astype(bool)
    valid_count = label_mask.sum(axis=0)
    positive = (labels * label_mask).sum(axis=0)
    rate = np.divide(positive, valid_count, out=np.zeros_like(positive), where=valid_count > 0)
    minority_rate = np.minimum(rate, 1.0 - rate)
    presence_weight = 0.0 if np.any(minority_rate < 0.01) else 0.20
    return baseline, {
        "train_examples": count,
        "presence_positive_rate": rate.tolist(),
        "presence_valid_count": valid_count.tolist(),
        "presence_weight": presence_weight,
    }


def metric_report(
    prediction: np.ndarray,
    target_normalized: np.ndarray,
    target_physical: np.ndarray,
    mask: np.ndarray,
    scaler: RobustScaler,
) -> dict:
    error = target_normalized[..., None] - prediction
    pinball = np.maximum(QUANTILE_LEVELS * error, (QUANTILE_LEVELS - 1.0) * error)
    weights = (
        mask[..., None]
        * HORIZON_WEIGHTS[None, :, None, None]
        * TARGET_WEIGHTS[None, None, :, None]
    )
    weighted_pinball = float((pinball * weights).sum() / (3.0 * weights.sum()))
    per_cell = np.divide(
        (pinball * mask[..., None]).sum(axis=(0, 3)),
        (mask.sum(axis=0) * 3),
        out=np.full((5, 5), np.nan),
        where=mask.sum(axis=0) > 0,
    )
    prediction_physical = scaler.inverse_targets(prediction)
    median = prediction_physical[..., 1]
    physical_error = median - target_physical
    valid_count = mask.sum(axis=0)
    mae = np.divide(
        (np.abs(physical_error) * mask).sum(axis=0),
        valid_count,
        out=np.full((5, 5), np.nan),
        where=valid_count > 0,
    )
    bias = np.divide(
        (physical_error * mask).sum(axis=0),
        valid_count,
        out=np.full((5, 5), np.nan),
        where=valid_count > 0,
    )
    covered = (target_physical >= prediction_physical[..., 0]) & (
        target_physical <= prediction_physical[..., 2]
    )
    coverage = float((covered * mask).sum() / mask.sum())
    width = prediction_physical[..., 2] - prediction_physical[..., 0]
    mean_width = float((width * mask).sum() / mask.sum())
    return {
        "weighted_normalized_pinball": weighted_pinball,
        "per_horizon_target_pinball": per_cell.tolist(),
        "median_mae_physical": mae.tolist(),
        "median_bias_physical": bias.tolist(),
        "q10_q90_coverage": coverage,
        "q10_q90_mean_width_physical": mean_width,
        "valid_labels": int(mask.sum()),
    }


@torch.inference_mode()
def evaluate(
    model: ChmtQTiny,
    corpus: ArrowCorpus,
    partition: Partition,
    scaler: RobustScaler,
    device: torch.device,
    baseline: np.ndarray,
    seed: int,
    limit_batches: int | None,
) -> tuple[dict, dict]:
    model.eval()
    predictions: list[np.ndarray] = []
    normalized_targets: list[np.ndarray] = []
    physical_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    ids: list[str] = []
    for batch in corpus.batches(
        partition,
        epoch=0,
        seed=seed,
        include_features=True,
        limit_batches=limit_batches,
    ):
        ids.extend(batch["example_id"])
        scaled = move_and_scale(batch, scaler, device)
        raw, _, _ = model_forward(model, scaled)
        predictions.append(ordered_quantiles(raw).cpu().numpy())
        normalized_targets.append(scaled["normalized_targets"].cpu().numpy())
        physical_targets.append(scaled["targets"].cpu().numpy())
        masks.append(scaled["target_mask"].cpu().numpy().astype(bool))
    prediction = np.concatenate(predictions)
    normalized = np.concatenate(normalized_targets)
    physical = np.concatenate(physical_targets)
    mask = np.concatenate(masks)
    model_metrics = metric_report(prediction, normalized, physical, mask, scaler)
    baseline_prediction = np.broadcast_to(baseline, prediction.shape)
    baseline_metrics = metric_report(baseline_prediction, normalized, physical, mask, scaler)
    return {
        "partition": partition.name,
        "first_example_id": ids[0],
        "last_example_id": ids[-1],
        "examples": len(ids),
        "model": model_metrics,
        "empirical_marginal_baseline": baseline_metrics,
        "relative_pinball_improvement_percent": 100.0
        * (baseline_metrics["weighted_normalized_pinball"] - model_metrics["weighted_normalized_pinball"])
        / baseline_metrics["weighted_normalized_pinball"],
    }, {"ids": ids}


def learning_rate(step: int, total_steps: int, base: float, minimum: float, warmup_fraction: float) -> float:
    warmup = max(1, round(total_steps * warmup_fraction))
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total_steps - warmup - 1)
    return minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def make_partitions(split_path: Path) -> tuple[Partition, Partition, Partition, str, str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if canonical_payload_sha256(split["payload"]) != split["artifact_sha256"]:
        raise ValueError("split artifact payload hash mismatch")
    fold = split["payload"]["folds"][0]["payload"]
    train_ids = frozenset(fold["train_example_ids"])
    validation_ids = frozenset(fold["validation_example_ids"])
    train = Partition(
        "train-2026-06-01--18",
        train_ids.__contains__,
        min(train_ids),
        max(train_ids),
        len(train_ids),
    )
    validation = Partition(
        "validation-2026-06-19--20",
        validation_ids.__contains__,
        min(validation_ids),
        max(validation_ids),
        len(validation_ids),
    )
    test_lower = "okx-btc-usdt-20260621T000000Z"
    test_upper = "okx-btc-usdt-20260624T999999Z"
    test = Partition(
        "development-forward-2026-06-21--24",
        lambda example_id: "20260621" <= date_from_id(example_id) <= "20260624",
        test_lower,
        test_upper,
        5724,
    )
    return (
        train,
        validation,
        test,
        split["artifact_sha256"],
        split["payload"]["source_dataset_hash"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-v1"),
    )
    parser.add_argument(
        "--study",
        type=Path,
        default=Path(r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-tiny-retrospective-v1"),
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-eval-batches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("this pilot requires a CUDA device")
    device = torch.device("cuda:0")
    output_dir = args.output / f"seed-{args.seed}"
    if (output_dir / "best-model.pt").exists() or (output_dir / "result.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run artifacts in {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_path = args.study / "split.json"
    scaler_path = args.study / "fold-01-scaler.json"
    (
        train_partition,
        validation_partition,
        test_partition,
        split_hash,
        split_dataset_id,
    ) = make_partitions(split_path)
    corpus = ArrowCorpus(args.dataset, args.batch_size)
    corpus_verification = corpus.verify()
    scaler = RobustScaler(scaler_path, device)
    if split_dataset_id != corpus.manifest["dataset_id"]:
        raise ValueError("split is not bound to the selected dataset")
    scaler_payload = scaler.artifact["payload"]
    if scaler_payload["dataset_id"] != corpus.manifest["dataset_id"]:
        raise ValueError("scaler is not bound to the selected dataset")
    if scaler_payload["split_manifest_sha256"] != split_hash:
        raise ValueError("scaler is not bound to the selected split")
    baseline, label_report = fit_empirical_baseline(corpus, train_partition, scaler, device, args.seed)
    presence_weight = float(label_report["presence_weight"])

    model = ChmtQTiny(TinyConfig()).to(device)
    count = parameter_count(model)
    if not 200_000 <= count <= 500_000:
        raise RuntimeError(f"Tiny parameter budget violated: {count}")
    decay_parameters: list[Tensor] = []
    no_decay_parameters: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2 and name.endswith(".weight"):
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_parameters, "weight_decay": args.weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    batches_per_epoch = args.limit_train_batches or math.ceil(
        train_partition.expected_count / args.batch_size
    )
    updates_per_epoch = math.ceil(batches_per_epoch / args.accumulation_steps)
    total_updates = updates_per_epoch * args.epochs
    best_validation = math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    update = 0
    started = time.perf_counter()
    checkpoint_path = output_dir / "best-model.pt"
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_pinball = 0.0
        train_batches = 0
        for batch_index, batch in enumerate(
            corpus.batches(
                train_partition,
                epoch=epoch,
                seed=args.seed,
                include_features=True,
                shuffle=True,
                limit_batches=args.limit_train_batches,
            ),
            start=1,
        ):
            scaled = move_and_scale(batch, scaler, device)
            raw, auxiliary_raw, _ = model_forward(model, scaled)
            primary, components = base_loss(raw, scaled, presence_weight)
            auxiliary, _ = base_loss(auxiliary_raw, scaled, presence_weight)
            progress = update / max(total_updates - 1, 1)
            loss = primary + auxiliary_weight(progress) * auxiliary
            (loss / args.accumulation_steps).backward()
            train_pinball += components["pinball"]
            train_batches += 1
            should_step = batch_index % args.accumulation_steps == 0
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                lr = learning_rate(
                    update,
                    total_updates,
                    args.learning_rate,
                    args.minimum_learning_rate,
                    0.05,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update += 1
        if train_batches % args.accumulation_steps:
            remainder = train_batches % args.accumulation_steps
            correction = args.accumulation_steps / remainder
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            lr = learning_rate(
                update,
                total_updates,
                args.learning_rate,
                args.minimum_learning_rate,
                0.05,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

        validation, _ = evaluate(
            model,
            corpus,
            validation_partition,
            scaler,
            device,
            baseline,
            args.seed,
            args.limit_eval_batches,
        )
        validation_score = validation["model"]["weighted_normalized_pinball"]
        epoch_record = {
            "epoch": epoch,
            "mean_train_batch_pinball": train_pinball / max(train_batches, 1),
            "validation_pinball": validation_score,
            "validation_baseline_pinball": validation["empirical_marginal_baseline"][
                "weighted_normalized_pinball"
            ],
            "updates": update,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record), flush=True)
        if validation_score < best_validation:
            best_validation = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(model.config),
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_pinball": validation_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation, _ = evaluate(
        model,
        corpus,
        validation_partition,
        scaler,
        device,
        baseline,
        args.seed,
        args.limit_eval_batches,
    )
    test, _ = evaluate(
        model,
        corpus,
        test_partition,
        scaler,
        device,
        baseline,
        args.seed,
        args.limit_eval_batches,
    )

    sample = next(
        corpus.batches(
            test_partition,
            epoch=0,
            seed=args.seed,
            include_features=True,
            limit_batches=1,
        )
    )
    for name, value in list(sample.items()):
        if name != "example_id":
            sample[name] = value[:1]
    sample_scaled = move_and_scale(sample, scaler, device)
    model.eval()
    with torch.inference_mode():
        first_raw, _, _ = model_forward(model, sample_scaled)
        second_raw, _, _ = model_forward(model, sample_scaled)
        torch.cuda.synchronize()
        latency_start = time.perf_counter()
        for _ in range(20):
            model_forward(model, sample_scaled)
        torch.cuda.synchronize()
    deterministic_max_abs_difference = float((first_raw - second_raw).abs().max().cpu())
    latency_ms = (time.perf_counter() - latency_start) * 1000.0 / 20

    report = {
        "study": "CHMT-Q Tiny retrospective engineering pilot",
        "interpretation": "Development evidence only; no confirmatory or economic claim.",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "model_config": asdict(model.config),
        "trainable_parameters": count,
        "dataset_id": corpus.manifest["dataset_id"],
        "dataset_manifest_sha256": sha256_file(corpus.manifest_path),
        "corpus_verification": corpus_verification,
        "split_manifest_sha256": split_hash,
        "scaler_artifact_sha256": scaler.artifact["artifact_sha256"],
        "label_report": label_report,
        "presence_head_status": (
            "inactive and unvalidated because every valid training label is positive"
            if presence_weight == 0.0
            else "trained"
        ),
        "selected_epoch": best_epoch,
        "optimizer_updates": update,
        "history": history,
        "validation": validation,
        "development_forward_test": test,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "batch_one_latency_ms": latency_ms,
        "deterministic_rerun_max_abs_difference": deterministic_max_abs_difference,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "model_source_sha256": sha256_file(Path(__file__).with_name("model.py")),
            "runner_source_sha256": sha256_file(Path(__file__)),
        },
        "wall_seconds": time.perf_counter() - started,
        "limits": {
            "train_batches": args.limit_train_batches,
            "evaluation_batches": args.limit_eval_batches,
        },
    }
    report_path = output_dir / "result.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(report_path), "test": test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
