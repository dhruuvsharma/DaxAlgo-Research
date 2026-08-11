"""Deterministic CHMT-Q training configuration and optimizer helpers."""

from .checkpoint import (
    CheckpointVerificationError,
    load_checkpoint,
    save_checkpoint,
)
from .config import TrainingConfig
from .optim import (
    apply_learning_rate,
    build_adamw,
    build_parameter_groups,
    learning_rate_at_step,
    seed_everything,
    validate_training_runtime,
)
from .runner import (
    DevelopmentEvaluationResult,
    ExperimentRunResult,
    evaluate_development_checkpoint,
    run_development_experiment,
)
from .trainer import (
    ChmtQEvaluationResult,
    ChmtQTrainer,
    TrainingStepResult,
    TrainingWindowResult,
    evaluate_loss,
    evaluate_model,
)
from .walk_forward import (
    FoldPurge,
    OuterHoldout,
    OuterHoldoutAccess,
    WalkForwardFold,
    WalkForwardManifest,
    build_expanding_walk_forward,
    iter_fold_batches,
    iter_outer_holdout_batches,
    outer_holdout_consumption_ledger_path,
)

__all__ = [
    "CheckpointVerificationError",
    "ChmtQEvaluationResult",
    "ChmtQTrainer",
    "DevelopmentEvaluationResult",
    "ExperimentRunResult",
    "FoldPurge",
    "OuterHoldout",
    "OuterHoldoutAccess",
    "TrainingConfig",
    "TrainingStepResult",
    "TrainingWindowResult",
    "WalkForwardFold",
    "WalkForwardManifest",
    "apply_learning_rate",
    "build_adamw",
    "build_parameter_groups",
    "build_expanding_walk_forward",
    "evaluate_model",
    "evaluate_development_checkpoint",
    "evaluate_loss",
    "iter_fold_batches",
    "iter_outer_holdout_batches",
    "outer_holdout_consumption_ledger_path",
    "learning_rate_at_step",
    "load_checkpoint",
    "save_checkpoint",
    "seed_everything",
    "validate_training_runtime",
    "run_development_experiment",
]
