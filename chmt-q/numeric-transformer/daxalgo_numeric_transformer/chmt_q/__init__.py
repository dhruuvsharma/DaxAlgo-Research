"""Selected CHMT-Q v1 model, contracts, data gates, and training primitives."""

from .config import ChmtQConfig
from .contracts import ChmtQBatch, NoAvailableModalitiesError
from .data import (
    ArrowDataset,
    DatasetManifest,
    FoldBoundScalerArtifact,
    ParquetAuditReport,
    audit_lse_tick_parquet,
    fit_fold_bound_scaler,
)
from .experiment import ExperimentConfig
from .export import ChmtQExportWrapper, export_chmt_q_onnx
from .loss import (
    ChmtQLoss,
    ChmtQLossTerms,
    NoSupervisedTermsError,
    auxiliary_loss_weight,
    chmt_q_loss,
)
from .model import ChmtQTransformer
from .output import ChmtQOutput, transform_raw_output
from .qualification import (
    QualificationControlError,
    run_random_target_negative_control,
    run_tiny_overfit_control,
)
from .scaling import RobustScalerArtifact, fit_robust_scaler
from .splits import ExampleDependency, SplitManifest, build_chronological_split
from .synthetic import make_chmt_q_synthetic_batch
from .training import (
    ChmtQTrainer,
    ExperimentRunResult,
    TrainingConfig,
    WalkForwardManifest,
    build_expanding_walk_forward,
    run_development_experiment,
)

__all__ = [
    "ChmtQBatch",
    "ChmtQConfig",
    "ChmtQExportWrapper",
    "ChmtQLoss",
    "ChmtQLossTerms",
    "ChmtQOutput",
    "ChmtQTrainer",
    "ChmtQTransformer",
    "ExampleDependency",
    "ExperimentConfig",
    "ExperimentRunResult",
    "ArrowDataset",
    "DatasetManifest",
    "FoldBoundScalerArtifact",
    "NoAvailableModalitiesError",
    "NoSupervisedTermsError",
    "ParquetAuditReport",
    "QualificationControlError",
    "RobustScalerArtifact",
    "SplitManifest",
    "TrainingConfig",
    "WalkForwardManifest",
    "audit_lse_tick_parquet",
    "auxiliary_loss_weight",
    "chmt_q_loss",
    "build_chronological_split",
    "build_expanding_walk_forward",
    "export_chmt_q_onnx",
    "fit_robust_scaler",
    "fit_fold_bound_scaler",
    "make_chmt_q_synthetic_batch",
    "run_development_experiment",
    "run_random_target_negative_control",
    "run_tiny_overfit_control",
    "transform_raw_output",
]
