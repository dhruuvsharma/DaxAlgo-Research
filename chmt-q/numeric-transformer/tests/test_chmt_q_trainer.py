from __future__ import annotations

import copy

import pytest
import torch

import daxalgo_numeric_transformer.chmt_q.output as output_module
from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.contracts import ChmtQBatch
from daxalgo_numeric_transformer.chmt_q.loss import chmt_q_loss
from daxalgo_numeric_transformer.chmt_q.model import ChmtQTransformer
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch
from daxalgo_numeric_transformer.chmt_q.training import (
    ChmtQTrainer,
    TrainingConfig,
    evaluate_loss,
    load_checkpoint,
    save_checkpoint,
)
from daxalgo_numeric_transformer.chmt_q.training.trainer import evaluate_model


def _setup(seed: int = 83):
    torch.manual_seed(seed)
    model_config = ChmtQConfig.tiny()
    model = ChmtQTransformer(model_config)
    batch = make_chmt_q_synthetic_batch(
        model_config, batch_size=2, seed=seed, device="cpu"
    )
    training_config = TrainingConfig(
        seed=seed,
        micro_batch_size=2,
        accumulation_steps=2,
        mixed_precision=False,
    )
    return model, batch, training_config


def test_trainer_accumulates_then_updates_with_finite_gradients() -> None:
    model, batch, config = _setup()
    trainer = ChmtQTrainer(model, total_optimizer_steps=1, config=config)

    window = trainer.train_accumulation_window((batch, batch))
    first, second = window.micro_batches

    assert not first.optimizer_updated
    assert first.gradient_norm is None
    assert second.optimizer_updated
    assert second.optimizer_step == 1
    assert second.gradient_norm is not None and torch.isfinite(
        torch.tensor(second.gradient_norm)
    )
    assert all(torch.isfinite(torch.tensor([second.loss, second.learning_rate])))
    assert window.optimizer_step == 1
    assert torch.isfinite(torch.tensor(window.loss))
    assert torch.isfinite(torch.tensor(window.update_norm))
    assert window.update_norm > 0.0


def test_mixed_precision_trainer_handles_autocast_promoted_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from daxalgo_numeric_transformer.chmt_q.training import trainer as trainer_module

    class IdentityGradScaler:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def scale(self, value: torch.Tensor) -> torch.Tensor:
            return value

        def unscale_(self, _optimizer: torch.optim.Optimizer) -> None:
            pass

        def step(self, optimizer: torch.optim.Optimizer) -> None:
            optimizer.step()

        def update(self) -> None:
            pass

    original_softplus = output_module.F.softplus

    def promote_softplus(value: torch.Tensor) -> torch.Tensor:
        return original_softplus(value.float())

    monkeypatch.setattr(
        trainer_module,
        "validate_training_runtime",
        lambda _config, _device: torch.device("cpu"),
    )
    monkeypatch.setattr(trainer_module.torch.amp, "GradScaler", IdentityGradScaler)
    monkeypatch.setattr(output_module.F, "softplus", promote_softplus)
    model_config = ChmtQConfig.tiny()
    model = ChmtQTransformer(model_config)
    batch = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=87, device="cpu"
    )
    trainer = ChmtQTrainer(
        model,
        total_optimizer_steps=1,
        config=TrainingConfig(
            seed=87,
            micro_batch_size=1,
            accumulation_steps=1,
            mixed_precision=True,
        ),
        device="cpu",
    )

    result = trainer.train_micro_batch(batch)

    assert result.optimizer_updated
    assert torch.isfinite(torch.tensor(result.loss))
    assert result.gradient_norm is not None
    assert torch.isfinite(torch.tensor(result.gradient_norm))


def test_two_reference_runs_are_exactly_deterministic() -> None:
    first_model, batch, config = _setup(seed=89)
    initial = copy.deepcopy(first_model.state_dict())
    first = ChmtQTrainer(first_model, total_optimizer_steps=1, config=config)
    first.fit([batch, batch])

    second_model = ChmtQTransformer(ChmtQConfig.tiny())
    second_model.load_state_dict(initial)
    second = ChmtQTrainer(second_model, total_optimizer_steps=1, config=config)
    second.fit([batch, batch])

    for name, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[name], rtol=0, atol=0)


def test_evaluation_requires_data_and_returns_finite_value() -> None:
    model, batch, _ = _setup(seed=97)
    original_device = next(model.parameters()).device
    model.train()

    value = evaluate_loss(model, [batch])

    assert torch.isfinite(torch.tensor(value))
    assert model.training
    assert next(model.parameters()).device == original_device
    model.eval()
    with pytest.raises(ValueError, match="no batches"):
        evaluate_loss(model, [])
    assert not model.training
    assert next(model.parameters()).device == original_device


def test_trainer_rejects_a_batch_that_violates_configured_micro_batch_size() -> None:
    model, _, config = _setup(seed=101)
    short_batch = make_chmt_q_synthetic_batch(
        model.config, batch_size=1, seed=101, device="cpu"
    )
    trainer = ChmtQTrainer(model, total_optimizer_steps=1, config=config)

    with pytest.raises(ValueError, match="micro_batch_size"):
        trainer.train_accumulation_window((short_batch, short_batch))

    assert trainer.micro_step == 0
    assert trainer.optimizer_step == 0


def test_checkpoint_resume_matches_uninterrupted_training_exactly(tmp_path) -> None:
    seed = 109
    torch.manual_seed(seed)
    model_config = ChmtQConfig.tiny(dropout=0.20)
    model = ChmtQTransformer(model_config)
    batch = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=seed, device="cpu"
    )
    training_config = TrainingConfig(
        seed=seed,
        micro_batch_size=1,
        accumulation_steps=2,
        mixed_precision=False,
    )
    uninterrupted = ChmtQTrainer(
        model,
        total_optimizer_steps=2,
        config=training_config,
    )
    with pytest.raises(RuntimeError, match="train_accumulation_window"):
        uninterrupted.train_micro_batch(batch)
    uninterrupted.train_accumulation_window((batch, batch))
    trainer_state = uninterrupted.state_dict()
    assert trainer_state["sampler_position"] == "external"
    checkpoint = save_checkpoint(
        tmp_path,
        step=uninterrupted.optimizer_step,
        model=uninterrupted.model,
        optimizer=uninterrupted.optimizer,
        run_identity={"dataset": "synthetic", "config": "resume-v1"},
        trainer_state=trainer_state,
    )

    uninterrupted.train_accumulation_window((batch, batch))

    resumed_model = ChmtQTransformer(model_config)
    resumed = ChmtQTrainer(
        resumed_model,
        total_optimizer_steps=2,
        config=training_config,
    )
    restored = load_checkpoint(
        checkpoint,
        model=resumed.model,
        optimizer=resumed.optimizer,
        expected_identity={"config": "resume-v1", "dataset": "synthetic"},
    )
    resumed.load_state_dict(restored["trainer_state"])
    resumed.train_accumulation_window((batch, batch))

    assert resumed.micro_step == uninterrupted.micro_step == 4
    assert resumed.optimizer_step == uninterrupted.optimizer_step == 2
    for name, value in uninterrupted.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)


def test_unequal_mask_accumulation_matches_one_effective_batch() -> None:
    seed = 131
    torch.manual_seed(seed)
    model_config = ChmtQConfig.tiny()
    first_model = ChmtQTransformer(model_config)
    initial = copy.deepcopy(first_model.state_dict())
    second_model = ChmtQTransformer(model_config)
    second_model.load_state_dict(initial)
    first_batch = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=seed, device="cpu"
    )
    second_batch = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=seed + 1, device="cpu"
    )
    first_batch.target_mask.zero_()
    first_batch.target_mask[0, 0, 0] = True
    first_batch.presence_mask.zero_()
    first_batch.presence_mask[0, 0, 0] = True
    second_batch.target_mask.zero_()
    second_batch.target_mask[0, :3, :4] = True
    second_batch.presence_mask.zero_()
    second_batch.presence_mask[0, :, :] = True
    combined = _concatenate_batches(first_batch, second_batch, model_config)

    accumulated = ChmtQTrainer(
        first_model,
        total_optimizer_steps=1,
        config=TrainingConfig(
            seed=seed,
            micro_batch_size=1,
            accumulation_steps=2,
        ),
    )
    effective = ChmtQTrainer(
        second_model,
        total_optimizer_steps=1,
        config=TrainingConfig(
            seed=seed,
            micro_batch_size=2,
            accumulation_steps=1,
        ),
    )

    window = accumulated.train_accumulation_window((first_batch, second_batch))
    single = effective.train_micro_batch(combined)

    assert window.loss == pytest.approx(single.loss, rel=1.0e-6, abs=1.0e-7)
    for name, value in accumulated.model.state_dict().items():
        torch.testing.assert_close(
            value, effective.model.state_dict()[name], rtol=2.0e-6, atol=2.0e-7
        )


def test_evaluation_aggregates_full_set_statistics_not_batch_means() -> None:
    seed = 137
    torch.manual_seed(seed)
    model_config = ChmtQConfig.tiny()
    model = ChmtQTransformer(model_config).eval()
    first = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=seed, device="cpu"
    )
    second = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=seed + 1, device="cpu"
    )
    first.target_mask.zero_()
    first.target_mask[0, 0, 0] = True
    first.presence_mask.zero_()
    first.presence_mask[0, 0, 0] = True
    second.target_mask[0, 4, :] = False
    second.presence_mask[0, 4, :] = False
    combined = _concatenate_batches(first, second, model_config)

    result = evaluate_model(model, [first, second])
    with torch.no_grad():
        prediction = model.forward_training(combined)
        direct = chmt_q_loss(
            prediction.final,
            targets=combined.targets,
            target_mask=combined.target_mask,
            presence_labels=combined.presence_labels,
            presence_mask=combined.presence_mask,
            config=model_config,
        )

    assert result.loss == pytest.approx(float(direct.total), rel=1.0e-6)
    assert result.pinball == pytest.approx(float(direct.pinball), rel=1.0e-6)
    assert result.metrics.weighted_primary_metric == pytest.approx(
        result.pinball, rel=1.0e-6
    )
    assert result.statistics.forecast_counts.equal(
        combined.target_mask.sum(dim=0, dtype=torch.int64)
    )
    assert result.metrics.forecast_counts.equal(result.statistics.forecast_counts.cpu())


def test_evaluation_does_not_materialize_per_batch_loss_statistics(
    monkeypatch,
) -> None:
    seed = 149
    torch.manual_seed(seed)
    model_config = ChmtQConfig.tiny()
    model = ChmtQTransformer(model_config).eval()
    batch = make_chmt_q_synthetic_batch(
        model_config, batch_size=1, seed=seed, device="cpu"
    )
    from daxalgo_numeric_transformer.chmt_q.training import trainer as trainer_module

    def reject_materialized_aggregation(*_args, **_kwargs):
        raise AssertionError("evaluation materialized per-batch loss statistics")

    monkeypatch.setattr(
        trainer_module,
        "aggregate_chmt_q_loss_statistics",
        reject_materialized_aggregation,
    )
    result = evaluate_model(model, (batch for _ in range(32)))

    assert result.statistics.pinball_numerator.device.type == "cpu"
    assert not result.statistics.pinball_numerator.requires_grad
    assert result.statistics.forecast_counts.equal(
        batch.target_mask.sum(dim=0, dtype=torch.int64) * 32
    )
    assert result.statistics.presence_counts.equal(
        batch.presence_mask.sum(dim=0, dtype=torch.int64) * 32
    )


def _concatenate_batches(
    first: ChmtQBatch, second: ChmtQBatch, config: ChmtQConfig
) -> ChmtQBatch:
    def combine(name: str):
        left = getattr(first, name)
        right = getattr(second, name)
        if left is None or right is None:
            return None
        return torch.cat((left, right), dim=0)

    return ChmtQBatch(
        bars=combine("bars"),
        bars_mask=combine("bars_mask"),
        l1=combine("l1"),
        l1_mask=combine("l1_mask"),
        l2=combine("l2"),
        l2_level_mask=combine("l2_level_mask"),
        l2_snapshot_mask=combine("l2_snapshot_mask"),
        quality=combine("quality"),
        targets=combine("targets"),
        target_mask=combine("target_mask"),
        presence_labels=combine("presence_labels"),
        presence_mask=combine("presence_mask"),
    ).validate(config, require_supervision=True)
