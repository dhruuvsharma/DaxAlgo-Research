"""Review-safe commands for the legacy prototype and selected CHMT-Q implementation."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .chmt_q.config import ChmtQConfig
from .chmt_q.data import (
    ArrowDataset,
    FoldBoundScalerArtifact,
    LocalResearchAuthorization,
    audit_lse_tick_parquet,
    fit_fold_bound_scaler,
)
from .chmt_q.dependencies import (
    verify_runtime_dependency_lock,
    write_runtime_dependency_lock,
)
from .chmt_q.experiment import ExperimentConfig, source_tree_sha256
from .chmt_q.model import ChmtQTransformer
from .chmt_q.qualification import (
    run_random_target_negative_control,
    run_tiny_overfit_control,
)
from .chmt_q.scaling import DEFAULT_SCALER_CHUNK_VALUES
from .chmt_q.synthetic import make_chmt_q_synthetic_batch
from .chmt_q.training import (
    ChmtQTrainer,
    TrainingConfig,
    WalkForwardManifest,
    build_expanding_walk_forward,
    evaluate_development_checkpoint,
    evaluate_loss,
    run_development_experiment,
)
from .config import ModelConfig
from .export import export_onnx
from .loss import masked_huber_loss
from .model.numeric_transformer import NumericMarketTransformer
from .synthetic import make_synthetic_batch


@dataclass(frozen=True)
class SmokeResult:
    """Machine-readable summary of one synthetic optimization smoke run."""

    device: str
    seed: int
    steps: int
    parameter_count: int
    initial_loss: float
    final_loss: float

    def as_dict(self) -> dict[str, int | float | str]:
        return {
            "device": self.device,
            "seed": self.seed,
            "steps": self.steps,
            "parameter_count": self.parameter_count,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
        }


@dataclass(frozen=True)
class ChmtQSmokeResult:
    """Machine-readable summary of a bounded CHMT-Q optimization smoke run."""

    device: str
    seed: int
    steps: int
    parameter_count: int
    initial_loss: float
    final_loss: float

    def as_dict(self) -> dict[str, int | float | str | bool]:
        return {
            "architecture": "CHMT-Q-v1",
            "device": self.device,
            "seed": self.seed,
            "steps": self.steps,
            "parameter_count": self.parameter_count,
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "promotable": False,
        }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _default_source_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return device


def _add_experiment_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scaler", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--dependency-lock", type=Path, required=True)
    _add_local_research_authorization_argument(parser)


def _add_local_research_authorization_argument(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--local-research-authorization",
        type=Path,
        help="content-addressed local-only research authorization sidecar",
    )


def train_smoke(*, device: str = "cpu", seed: int = 17, steps: int = 3):
    """Optimize a tiny model against one fixed synthetic batch and return it with metrics."""

    if steps < 1:
        raise ValueError("steps must be at least 1")

    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    config = ModelConfig.tiny()
    model = NumericMarketTransformer(config).to(device=resolved_device, dtype=torch.float32)
    batch = make_synthetic_batch(config, batch_size=2, seed=seed, device=resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.0)

    model.train()
    with torch.no_grad():
        initial_loss = masked_huber_loss(model(batch), batch.targets, batch.target_mask)

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_huber_loss(model(batch), batch.targets, batch.target_mask)
        if not torch.isfinite(loss):
            raise RuntimeError("synthetic smoke loss became non-finite")
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("synthetic smoke produced no gradients")
        if any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients):
            raise RuntimeError("synthetic smoke produced a non-finite gradient")
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_loss = masked_huber_loss(model(batch), batch.targets, batch.target_mask)

    result = SmokeResult(
        device=str(resolved_device),
        seed=seed,
        steps=steps,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        initial_loss=float(initial_loss.detach().cpu()),
        final_loss=float(final_loss.detach().cpu()),
    )
    return model, result


def train_chmt_q_smoke(*, device: str = "cpu", seed: int = 17, steps: int = 3):
    """Run a bounded deterministic optimization without reading or training on market data."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    _seed_everything(seed)
    resolved_device = _resolve_device(device)
    model_config = ChmtQConfig.tiny(decoder_layers=2)
    model = ChmtQTransformer(model_config)
    batch = make_chmt_q_synthetic_batch(
        model_config,
        batch_size=2,
        seed=seed,
        device=resolved_device,
    )
    initial_loss = evaluate_loss(model, [batch], device=resolved_device)
    training_config = TrainingConfig(
        seed=seed,
        micro_batch_size=2,
        accumulation_steps=1,
        mixed_precision=False,
    )
    trainer = ChmtQTrainer(
        model,
        total_optimizer_steps=steps,
        config=training_config,
        device=resolved_device,
    )
    trainer.fit([batch])
    final_loss = evaluate_loss(model, [batch], device=resolved_device)
    result = ChmtQSmokeResult(
        device=str(resolved_device),
        seed=seed,
        steps=steps,
        parameter_count=model.parameter_count,
        initial_loss=initial_loss,
        final_loss=final_loss,
    )
    return model, result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daxalgo-numeric-transformer",
        description="Pro-private numeric transformer research and data-readiness tools",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke", help="run a deterministic short synthetic training loop")
    smoke.add_argument("--device", default="cpu", help="PyTorch device (default: cpu)")
    smoke.add_argument("--seed", type=int, default=17)
    smoke.add_argument("--steps", type=_positive_int, default=3)

    export = subparsers.add_parser(
        "export-smoke",
        help="train a tiny synthetic model and export a non-promotable fixed-shape ONNX graph",
    )
    export.add_argument(
        "--device",
        default="cpu",
        help="training device; export is always CPU FP32",
    )
    export.add_argument("--seed", type=int, default=17)
    export.add_argument("--steps", type=_positive_int, default=1)
    export.add_argument("--output", type=Path, default=Path("artifacts/smoke.onnx"))

    chmt_q_smoke = subparsers.add_parser(
        "chmtq-smoke",
        help="run a bounded synthetic optimization through the selected CHMT-Q model",
    )
    chmt_q_smoke.add_argument("--device", default="cpu")
    chmt_q_smoke.add_argument("--seed", type=int, default=17)
    chmt_q_smoke.add_argument("--steps", type=_positive_int, default=3)

    subparsers.add_parser(
        "chmtq-inspect",
        help="validate and print the selected reference parameter ledger",
    )

    audit = subparsers.add_parser(
        "audit-data",
        help="read-only audit of the observed BTC/USD Parquet tick corpus",
    )
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.add_argument(
        "--full-json",
        action="store_true",
        help="print the complete report instead of the concise machine summary",
    )

    verify = subparsers.add_parser(
        "verify-dataset",
        help="verify a compiled Arrow dataset, its content identity, and training gates",
    )
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--dataset-id", required=True)
    verify.add_argument("--instrument-id", required=True)
    _add_local_research_authorization_argument(verify)

    authorize = subparsers.add_parser(
        "authorize-local-research",
        help="authorize non-promotable local train/validation without changing dataset gates",
    )
    authorize.add_argument("--dataset", type=Path, required=True)
    authorize.add_argument("--dataset-id", required=True)
    authorize.add_argument("--authorizer", required=True)
    authorize.add_argument("--reason", required=True)
    authorize.add_argument("--confirm-rights", action="store_true")
    authorize.add_argument("--acknowledge-incomplete-readiness", action="store_true")
    authorize.add_argument("--acknowledge-incomplete-provenance", action="store_true")
    authorize.add_argument("--output", type=Path, required=True)

    lock = subparsers.add_parser(
        "lock-environment",
        help="write the canonical active CHMT-Q runtime dependency closure",
    )
    lock.add_argument("--output", type=Path, required=True)

    split = subparsers.add_parser(
        "build-split",
        help="build a content-addressed expanding walk-forward manifest",
    )
    split.add_argument("--dataset", type=Path, required=True)
    split.add_argument("--dataset-id", required=True)
    split.add_argument("--initial-train-sessions", type=_positive_int, required=True)
    split.add_argument("--validation-sessions", type=_positive_int, required=True)
    split.add_argument("--outer-holdout-sessions", type=_positive_int, required=True)
    split.add_argument("--embargo-ns", type=_non_negative_int, default=0)
    split.add_argument("--output", type=Path, required=True)
    _add_local_research_authorization_argument(split)

    scaler = subparsers.add_parser(
        "fit-scaler",
        help="fit an exact review scaler bound to one verified training fold",
    )
    scaler.add_argument("--dataset", type=Path, required=True)
    scaler.add_argument("--dataset-id", required=True)
    scaler.add_argument("--split", type=Path, required=True)
    scaler.add_argument("--fold", type=_positive_int, required=True)
    scaler.add_argument("--batch-size", type=_positive_int, default=128)
    scaler.add_argument("--spool-directory", type=Path)
    scaler.add_argument(
        "--max-chunk-values",
        type=_positive_int,
        default=DEFAULT_SCALER_CHUNK_VALUES,
    )
    scaler.add_argument("--output", type=Path, required=True)
    _add_local_research_authorization_argument(scaler)

    initialize = subparsers.add_parser(
        "init-experiment",
        help="freeze a reference-model experiment configuration and run identity",
    )
    initialize.add_argument("--dataset", type=Path, required=True)
    initialize.add_argument("--dataset-id", required=True)
    initialize.add_argument("--instrument-id", required=True)
    initialize.add_argument("--split", type=Path, required=True)
    initialize.add_argument("--scaler", type=Path, required=True)
    initialize.add_argument("--fold", type=_positive_int, required=True)
    initialize.add_argument("--source-root", type=Path, default=_default_source_root())
    initialize.add_argument("--dependency-lock", type=Path, required=True)
    initialize.add_argument("--seed", type=_non_negative_int, default=17)
    initialize.add_argument("--micro-batch-size", type=_positive_int, default=8)
    initialize.add_argument("--accumulation-steps", type=_positive_int, default=16)
    initialize.add_argument("--mixed-precision", action="store_true")
    initialize.add_argument("--device", default="cpu")
    initialize.add_argument("--attention-logit-threshold", type=float, default=80.0)
    initialize.add_argument("--max-epochs", type=_positive_int, default=50)
    initialize.add_argument("--early-stop-patience", type=_positive_int, default=8)
    initialize.add_argument("--checkpoint-every", type=_positive_int, default=1)
    initialize.add_argument(
        "--primary-metric",
        choices=("weighted_pinball", "one_step_poc_mae"),
        default="weighted_pinball",
    )
    initialize.add_argument("--minimum-improvement", type=float, default=0.0)
    initialize.add_argument("--output", type=Path, required=True)
    _add_local_research_authorization_argument(initialize)

    train = subparsers.add_parser(
        "train",
        help="train one verified development fold; the outer holdout remains inaccessible",
    )
    _add_experiment_artifact_arguments(train)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--device")
    train.add_argument("--resume", type=Path)
    train.add_argument(
        "--progress",
        action="store_true",
        help="print live bounded progress during baseline, training, and validation",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="read-only validation evaluation of one verified development checkpoint",
    )
    _add_experiment_artifact_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")

    subparsers.add_parser(
        "qualify",
        help="run bounded non-promotable overfit and random-target trainer controls",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code = 0
    try:
        if args.command in {"smoke", "export-smoke"}:
            model, result = train_smoke(device=args.device, seed=args.seed, steps=args.steps)
            payload = result.as_dict()
            payload["promotable"] = False
        elif args.command == "chmtq-smoke":
            _, result = train_chmt_q_smoke(device=args.device, seed=args.seed, steps=args.steps)
            payload = result.as_dict()
        elif args.command == "chmtq-inspect":
            model = ChmtQTransformer(ChmtQConfig.reference())
            model.assert_reference_parameter_count()
            payload = {
                "architecture": "CHMT-Q-v1",
                "parameter_count": model.parameter_count,
                "expected_parameter_count": model.config.reference_parameter_count,
                "raw_output_shape": [1, 5, 17],
                "parameter_breakdown": model.parameter_breakdown(),
            }
        elif args.command == "audit-data":
            report = audit_lse_tick_parquet(args.input)
            if args.output is not None:
                _write_audit_report(args.output, report.to_json() + "\n")
            if args.full_json:
                print(report.to_json())
                return 0 if report.chmt_q_training_ready else 3
            payload = {
                "file_count": report.aggregate["file_count"],
                "total_rows": report.aggregate["total_rows"],
                "nominal_endpoint_span_ratio": report.aggregate["nominal_endpoint_span_ratio"],
                "exact_duplicate_surplus": report.aggregate["within_file_exact_duplicate_surplus"],
                "timestamp_collision_surplus": report.aggregate[
                    "within_file_timestamp_collision_surplus"
                ],
                "source_manifest_sha256": report.aggregate["source_manifest_sha256"],
                "chmt_q_training_ready": report.chmt_q_training_ready,
                "training_blockers": list(report.training_blockers),
                "report_path": str(args.output.resolve()) if args.output else None,
            }
            if not report.chmt_q_training_ready:
                exit_code = 3
        elif args.command == "lock-environment":
            lock = write_runtime_dependency_lock(args.output)
            payload = {
                "dependency_lock_sha256": verify_runtime_dependency_lock(args.output),
                "package_count": len(lock.packages),
                "roots": list(lock.roots),
                "output": str(args.output.resolve()),
            }
        elif args.command == "authorize-local-research":
            dataset = ArrowDataset.open(
                args.dataset,
                expected_dataset_id=args.dataset_id,
                require_training_capabilities=False,
            )
            authorization = LocalResearchAuthorization.create(
                dataset.manifest,
                dataset.root,
                authorizer=args.authorizer,
                reason=args.reason,
                rights_confirmed=args.confirm_rights,
                incomplete_readiness_acknowledged=(args.acknowledge_incomplete_readiness),
                incomplete_provenance_acknowledged=(args.acknowledge_incomplete_provenance),
            )
            authorization.write_new(args.output)
            payload = {
                "authorization_sha256": authorization.authorization_sha256,
                "research_dataset_id": authorization.research_dataset_id,
                "source_dataset_id": authorization.dataset_id,
                "scope": "local-development-train-validation-only",
                "promotable": False,
                "outer_holdout_allowed": False,
                "output": str(args.output.resolve()),
            }
        elif args.command == "verify-dataset":
            authorization = _read_local_research_authorization(args.local_research_authorization)
            dataset = ArrowDataset.open(
                args.input,
                expected_dataset_id=args.dataset_id,
                expected_instrument_id=args.instrument_id,
                require_training_capabilities=False,
                local_research_authorization=authorization,
            )
            capabilities = dataset.manifest.capabilities
            ready = (
                capabilities.training_ready
                and capabilities.rights_confirmed
                and capabilities.provenance_complete
            )
            payload = {
                "dataset_id": dataset.manifest.dataset_id,
                "instrument_id": dataset.manifest.instrument_id,
                "row_count": dataset.manifest.row_count,
                "shard_count": len(dataset.manifest.shards),
                "first_decision_ns": dataset.examples[0].decision_ns,
                "last_decision_ns": dataset.examples[-1].decision_ns,
                "tensor_schema_sha256": dataset.manifest.tensor_schema_sha256,
                "capabilities": capabilities.to_payload(),
                "chmt_q_training_ready": ready,
                "local_research_authorized": authorization is not None,
                "research_dataset_id": (
                    authorization.research_dataset_id if authorization else None
                ),
                "promotable": ready and authorization is None,
            }
            if not ready:
                exit_code = 3
        elif args.command == "build-split":
            dataset = _open_development_dataset(
                args,
                expected_dataset_id=args.dataset_id,
            )
            split = build_expanding_walk_forward(
                dataset.dependencies(),
                source_dataset_hash=dataset.manifest.dataset_id,
                initial_train_session_count=args.initial_train_sessions,
                validation_session_count=args.validation_sessions,
                outer_holdout_session_count=args.outer_holdout_sessions,
                embargo_ns=args.embargo_ns,
            )
            _write_audit_report(args.output, split.to_json())
            payload = {
                "dataset_id": dataset.manifest.dataset_id,
                "split_manifest_sha256": split.artifact_sha256,
                "fold_count": len(split.folds),
                "outer_holdout_example_count": len(split.outer_holdout.example_ids),
                "output": str(args.output.resolve()),
            }
        elif args.command == "fit-scaler":
            dataset = _open_development_dataset(
                args,
                expected_dataset_id=args.dataset_id,
            )
            split = _read_walk_forward(args.split)
            split.validate(dataset.dependencies())
            if split.source_dataset_hash != dataset.manifest.dataset_id:
                raise ValueError("split manifest is bound to a different dataset")
            fold = split.fold(args.fold)
            fold_id = _fold_id(args.fold)
            scaler = fit_fold_bound_scaler(
                dataset,
                dataset_id=dataset.manifest.dataset_id,
                split_manifest_sha256=split.artifact_sha256,
                fold_id=fold_id,
                ordered_train_example_ids=fold.train_example_ids,
                batch_size=args.batch_size,
                config=ChmtQConfig.reference(),
                spool_directory=args.spool_directory,
                max_chunk_values=args.max_chunk_values,
            )
            _write_audit_report(args.output, scaler.to_json())
            payload = {
                "dataset_id": dataset.manifest.dataset_id,
                "split_manifest_sha256": split.artifact_sha256,
                "fold_id": fold_id,
                "train_example_count": scaler.train_example_count,
                "scaler_artifact_sha256": scaler.artifact_sha256,
                "output": str(args.output.resolve()),
            }
        elif args.command == "init-experiment":
            dataset = _open_development_dataset(
                args,
                expected_dataset_id=args.dataset_id,
                expected_instrument_id=args.instrument_id,
            )
            if args.instrument_id != dataset.manifest.instrument_id:
                raise ValueError("requested instrument_id differs from the dataset manifest")
            split = _read_walk_forward(args.split)
            split.validate(dataset.dependencies())
            if split.source_dataset_hash != dataset.manifest.dataset_id:
                raise ValueError("split manifest is bound to a different dataset")
            fold = split.fold(args.fold)
            fold_id = _fold_id(args.fold)
            scaler = FoldBoundScalerArtifact.from_json(
                _read_regular_text(args.scaler),
                expected_dataset_id=dataset.manifest.dataset_id,
                expected_split_manifest_sha256=split.artifact_sha256,
                expected_fold_id=fold_id,
                ordered_train_example_ids=fold.train_example_ids,
            )
            training = TrainingConfig(
                seed=args.seed,
                micro_batch_size=args.micro_batch_size,
                accumulation_steps=args.accumulation_steps,
                mixed_precision=args.mixed_precision,
            )
            experiment = ExperimentConfig(
                dataset_id=dataset.manifest.dataset_id,
                instrument_id=dataset.manifest.instrument_id,
                split_manifest_sha256=split.artifact_sha256,
                scaler_artifact_sha256=scaler.artifact_sha256,
                source_tree_sha256=source_tree_sha256(args.source_root),
                dependency_lock_sha256=verify_runtime_dependency_lock(args.dependency_lock),
                fold_id=fold_id,
                seed=args.seed,
                model=ChmtQConfig.reference(),
                training=training,
                local_research_authorization_sha256=(
                    dataset.local_research_authorization.authorization_sha256
                    if dataset.local_research_authorization
                    else None
                ),
                device=args.device,
                attention_logit_threshold=args.attention_logit_threshold,
                max_epochs=args.max_epochs,
                early_stop_patience=args.early_stop_patience,
                checkpoint_every_epochs=args.checkpoint_every,
                primary_metric=args.primary_metric,
                minimum_improvement=args.minimum_improvement,
            )
            experiment.write_new(args.output)
            payload = {
                "run_id": experiment.run_id,
                "config_sha256": experiment.config_sha256,
                "instrument_id": experiment.instrument_id,
                "scaler_artifact_sha256": experiment.scaler_artifact_sha256,
                "effective_batch_size": training.effective_batch_size,
                "parameter_count": experiment.model.reference_parameter_count,
                "local_research_only": experiment.local_research_only,
                "promotable": not experiment.local_research_only,
                "output": str(args.output.resolve()),
            }
        elif args.command == "train":
            dataset, experiment, split, scaler, fold_number = _load_experiment_bundle(args)
            result = run_development_experiment(
                dataset,
                split,
                scaler,
                experiment,
                fold_number=fold_number,
                output_root=args.output_root,
                source_root=args.source_root,
                dependency_lock=args.dependency_lock,
                device=args.device,
                resume_checkpoint=args.resume,
                progress_callback=(_print_training_progress if args.progress else None),
            )
            payload = result.to_dict()
            payload["outer_holdout_accessed"] = False
            payload["local_research_only"] = experiment.local_research_only
            payload["promotable"] = not experiment.local_research_only
        elif args.command == "evaluate":
            dataset, experiment, split, scaler, fold_number = _load_experiment_bundle(args)
            result = evaluate_development_checkpoint(
                dataset,
                split,
                scaler,
                experiment,
                fold_number=fold_number,
                checkpoint=args.checkpoint,
                source_root=args.source_root,
                dependency_lock=args.dependency_lock,
                device=args.device,
            )
            payload = result.to_dict()
            payload["local_research_only"] = experiment.local_research_only
            payload["promotable"] = not experiment.local_research_only
        elif args.command == "qualify":
            payload = {
                "overfit": asdict(run_tiny_overfit_control()),
                "random_target": asdict(run_random_target_negative_control()),
                "promotable": False,
            }
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(f"unsupported command: {args.command}")

        if args.command == "export-smoke":
            output = export_onnx(model, args.output, config=ModelConfig.tiny(), seed=args.seed)
            payload["onnx_path"] = str(output)

        print(json.dumps(payload, sort_keys=True))
        return exit_code
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}")
        return 2


def _read_walk_forward(path: Path) -> WalkForwardManifest:
    return WalkForwardManifest.from_json(_read_regular_text(path))


def _load_experiment_bundle(args):
    experiment = ExperimentConfig.from_json(_read_regular_text(args.config))
    dataset = _open_development_dataset(
        args,
        expected_dataset_id=experiment.dataset_id,
        expected_instrument_id=experiment.instrument_id,
    )
    observed_authorization_sha256 = (
        dataset.local_research_authorization.authorization_sha256
        if dataset.local_research_authorization
        else None
    )
    if observed_authorization_sha256 != experiment.local_research_authorization_sha256:
        raise ValueError("experiment local-research authorization differs from the opened dataset")
    if dataset.manifest.instrument_id != experiment.instrument_id:
        raise ValueError("experiment instrument identity does not match the dataset")
    split = _read_walk_forward(args.split)
    split.validate(dataset.dependencies())
    if split.artifact_sha256 != experiment.split_manifest_sha256:
        raise ValueError("experiment configuration and split manifest differ")
    fold_number = _fold_number(experiment.fold_id)
    fold = split.fold(fold_number)
    scaler = FoldBoundScalerArtifact.from_json(
        _read_regular_text(args.scaler),
        expected_dataset_id=experiment.dataset_id,
        expected_split_manifest_sha256=split.artifact_sha256,
        expected_fold_id=experiment.fold_id,
        ordered_train_example_ids=fold.train_example_ids,
    )
    if scaler.artifact_sha256 != experiment.scaler_artifact_sha256:
        raise ValueError("experiment configuration and scaler artifact differ")
    return dataset, experiment, split, scaler, fold_number


def _read_local_research_authorization(
    path: Path | None,
) -> LocalResearchAuthorization | None:
    return LocalResearchAuthorization.read(path) if path is not None else None


def _open_development_dataset(
    args,
    *,
    expected_dataset_id: str,
    expected_instrument_id: str | None = None,
) -> ArrowDataset:
    authorization = _read_local_research_authorization(args.local_research_authorization)
    return ArrowDataset.open(
        args.dataset,
        expected_dataset_id=expected_dataset_id,
        expected_instrument_id=expected_instrument_id,
        local_research_authorization=authorization,
    )


def _print_training_progress(event: dict[str, object]) -> None:
    if event.get("phase") == "optimizer_window":
        window = int(event["window"])
        total = int(event["windows_per_epoch"])
        if window not in {1, total} and window % 10 != 0:
            return
    print("progress " + json.dumps(event, sort_keys=True), flush=True)


def _read_regular_text(path: Path) -> str:
    if path.is_symlink():
        raise ValueError(f"artifact must not be a symlink: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"artifact must be a regular non-symlink file: {resolved}")
    return resolved.read_text(encoding="utf-8")


def _fold_id(fold_number: int) -> str:
    return f"fold-{fold_number:02d}"


def _fold_number(fold_id: str) -> int:
    if not fold_id.startswith("fold-") or not fold_id[5:].isdigit():
        raise ValueError("experiment fold_id does not use the fold-NN convention")
    value = int(fold_id[5:])
    if value <= 0 or _fold_id(value) != fold_id:
        raise ValueError("experiment fold_id is not canonical")
    return value


def _write_audit_report(path: Path, content: str) -> None:
    resolved = path.resolve()
    if resolved.exists():
        raise FileExistsError(f"audit report already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"audit report staging path already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(resolved)


if __name__ == "__main__":
    raise SystemExit(main())
