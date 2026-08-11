from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch

from daxalgo_numeric_transformer.cli import train_smoke
from daxalgo_numeric_transformer.config import ModelConfig
from daxalgo_numeric_transformer.contracts import NoValidModalitiesError
from daxalgo_numeric_transformer.export import ExportWrapper
from daxalgo_numeric_transformer.loss import NoValidTargetsError, masked_huber_loss
from daxalgo_numeric_transformer.model.numeric_transformer import NumericMarketTransformer


def test_fully_missing_l2_is_finite_and_masked_values_are_invariant(
    tiny_config: ModelConfig, market_batch
) -> None:
    torch.manual_seed(31)
    model = NumericMarketTransformer(tiny_config).eval()
    level_mask = torch.zeros_like(market_batch.l2_level_mask)
    snapshot_mask = torch.zeros_like(market_batch.l2_snapshot_mask)
    zeros = replace(
        market_batch,
        l2=torch.zeros_like(market_batch.l2),
        l2_level_mask=level_mask,
        l2_snapshot_mask=snapshot_mask,
    )
    garbage = replace(
        zeros,
        l2=torch.full_like(market_batch.l2, 1.0e6),
    )

    with torch.no_grad():
        zero_result = model(zeros)
        garbage_result = model(garbage)
        gates = model.modality_gates(zeros)

    assert torch.isfinite(zero_result).all()
    torch.testing.assert_close(zero_result, garbage_result, rtol=0, atol=0)
    assert gates.shape == (2, 3)
    assert gates[:, 2].count_nonzero() == 0


def test_one_sided_l2_remains_finite(tiny_config: ModelConfig, market_batch) -> None:
    torch.manual_seed(37)
    model = NumericMarketTransformer(tiny_config).eval()
    level_mask = torch.zeros_like(market_batch.l2_level_mask)
    level_mask[:, :, 0, :3] = market_batch.l2_snapshot_mask.unsqueeze(-1)
    one_sided = replace(market_batch, l2_level_mask=level_mask)

    with torch.no_grad():
        result = model(one_sided)

    assert result.shape == (2, 5, 5)
    assert torch.isfinite(result).all()


def test_all_masked_loss_requests_controlled_batch_skip(market_batch) -> None:
    predictions = torch.randn_like(market_batch.targets, requires_grad=True)
    no_targets = torch.zeros_like(market_batch.target_mask)

    with pytest.raises(NoValidTargetsError, match="no valid target"):
        masked_huber_loss(predictions, market_batch.targets, no_targets)


def test_zero_weight_valid_targets_request_controlled_batch_skip(market_batch) -> None:
    predictions = torch.randn_like(market_batch.targets)
    delta_only = torch.zeros_like(market_batch.target_mask)
    delta_only[:, :, 4] = True

    with pytest.raises(NoValidTargetsError, match="positive loss weight"):
        masked_huber_loss(
            predictions,
            market_batch.targets,
            delta_only,
            head_weights=[1.0, 1.0, 1.0, 1.0, 0.0],
        )


def test_all_missing_modalities_abstain_before_decoding(
    tiny_config: ModelConfig, market_batch
) -> None:
    model = NumericMarketTransformer(tiny_config).eval()
    empty = replace(
        market_batch,
        bars_mask=torch.zeros_like(market_batch.bars_mask),
        l1_mask=torch.zeros_like(market_batch.l1_mask),
        l2_level_mask=torch.zeros_like(market_batch.l2_level_mask),
        l2_snapshot_mask=torch.zeros_like(market_batch.l2_snapshot_mask),
    )

    with torch.no_grad():
        gates = model.modality_gates(empty)
    assert gates.count_nonzero() == 0
    with pytest.raises(NoValidModalitiesError, match="at least one valid modality"):
        model(empty)


def test_export_wrapper_clamps_identity_scaled_features(
    tiny_config: ModelConfig, market_batch
) -> None:
    torch.manual_seed(41)
    wrapper = ExportWrapper(NumericMarketTransformer(tiny_config)).eval()
    clipped_bars = market_batch.bars.clone()
    extreme_bars = market_batch.bars.clone()
    clipped_bars[:, -1, 0] = 8.0
    extreme_bars[:, -1, 0] = 100.0
    clipped = replace(market_batch, bars=clipped_bars)
    extreme = replace(market_batch, bars=extreme_bars)

    with torch.no_grad():
        clipped_result = wrapper(*clipped.model_inputs())
        extreme_result = wrapper(*extreme.model_inputs())

    torch.testing.assert_close(clipped_result, extreme_result, rtol=0, atol=0)


def test_forward_and_backward_are_deterministic(tiny_config: ModelConfig, market_batch) -> None:
    torch.manual_seed(43)
    first = NumericMarketTransformer(tiny_config).eval()
    second = copy.deepcopy(first).eval()

    first_output = first(market_batch)
    first_loss = masked_huber_loss(first_output, market_batch.targets, market_batch.target_mask)
    first_loss.backward()
    second_output = second(market_batch)
    second_loss = masked_huber_loss(second_output, market_batch.targets, market_batch.target_mask)
    second_loss.backward()

    torch.testing.assert_close(first_output, second_output, rtol=0, atol=0)
    for (first_name, first_parameter), (second_name, second_parameter) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert first_name == second_name
        if first_parameter.grad is None or second_parameter.grad is None:
            assert first_parameter.grad is second_parameter.grad
        else:
            torch.testing.assert_close(first_parameter.grad, second_parameter.grad, rtol=0, atol=0)


def test_short_synthetic_update_is_finite_and_reduces_loss() -> None:
    _, result = train_smoke(device="cpu", seed=47, steps=3)

    assert torch.isfinite(torch.tensor([result.initial_loss, result.final_loss])).all()
    assert result.final_loss < result.initial_loss


def test_reference_configuration_stays_within_parameter_ceiling() -> None:
    model = NumericMarketTransformer(ModelConfig.reference())
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    assert 0 < parameter_count <= 2_250_000

