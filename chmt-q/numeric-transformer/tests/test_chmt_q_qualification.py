from __future__ import annotations

import math

from daxalgo_numeric_transformer.chmt_q.qualification import (
    NON_PROMOTABLE_REASON,
    run_random_target_negative_control,
    run_tiny_overfit_control,
)


def test_tiny_overfit_control_substantially_reduces_fixed_training_loss() -> None:
    result = run_tiny_overfit_control()

    assert result.passed
    assert result.disposition == "non-promotable"
    assert result.non_promotable_reason == NON_PROMOTABLE_REASON
    assert result.optimizer_steps == 12
    assert result.final_training_loss < result.initial_training_loss
    assert result.relative_loss_reduction >= result.minimum_required_reduction
    assert result.minimum_required_reduction >= 0.30
    assert math.isfinite(result.minimum_gradient_norm)
    assert math.isfinite(result.maximum_gradient_norm)


def test_random_target_control_rejects_spurious_held_out_signal() -> None:
    result = run_random_target_negative_control()

    assert result.passed
    assert result.disposition == "non-promotable"
    assert result.non_promotable_reason == NON_PROMOTABLE_REASON
    assert result.training_label_seed != result.validation_label_seed
    assert len(result.validation_loss_trace) == result.optimizer_steps + 1
    assert result.best_validation_loss == min(result.validation_loss_trace)
    assert result.final_validation_loss == result.validation_loss_trace[-1]
    assert (
        result.claimed_validation_improvement
        <= result.maximum_allowed_validation_improvement
    )
    assert result.maximum_allowed_validation_improvement <= 0.10
