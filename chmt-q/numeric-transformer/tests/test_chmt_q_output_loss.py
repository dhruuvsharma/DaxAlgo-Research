from __future__ import annotations

import math

import pytest
import torch

import daxalgo_numeric_transformer.chmt_q.output as output_module
from daxalgo_numeric_transformer.chmt_q import (
    ChmtQConfig,
    ChmtQOutput,
    NoSupervisedTermsError,
    auxiliary_loss_weight,
    chmt_q_loss,
    transform_raw_output,
)
from daxalgo_numeric_transformer.chmt_q.loss import (
    ChmtQLossStatisticsAccumulator,
    aggregate_chmt_q_loss_statistics,
    chmt_q_objective_from_statistics,
)


def _zero_output(batch_size: int = 1) -> ChmtQOutput:
    return ChmtQOutput(
        quantiles=torch.zeros(batch_size, 5, 5, 3),
        presence_logits=torch.zeros(batch_size, 5, 2),
    ).validate()


def test_raw_output_transform_is_ordered_bounded_and_preserves_logits() -> None:
    raw = torch.zeros(2, 5, 17)
    raw[..., 15] = 2.0
    raw[..., 16] = -3.0

    output = transform_raw_output(raw)
    gap = math.log(2.0)

    assert output.quantiles.shape == (2, 5, 5, 3)
    torch.testing.assert_close(output.q10[..., :4], torch.full((2, 5, 4), -gap))
    torch.testing.assert_close(output.median[..., :4], torch.zeros(2, 5, 4))
    torch.testing.assert_close(output.q90[..., :4], torch.full((2, 5, 4), gap))
    torch.testing.assert_close(
        output.q10[..., 4], torch.full((2, 5), -math.tanh(gap))
    )
    torch.testing.assert_close(output.median[..., 4], torch.zeros(2, 5))
    torch.testing.assert_close(
        output.q90[..., 4], torch.full((2, 5), math.tanh(gap))
    )
    torch.testing.assert_close(output.presence_logits, raw[..., 15:])
    assert torch.all(output.q10 <= output.median)
    assert torch.all(output.median <= output.q90)


def test_output_transform_is_differentiable_and_rejects_nonfinite_raw_values() -> None:
    raw = torch.randn(2, 5, 17, requires_grad=True)
    output = transform_raw_output(raw)
    (output.quantiles.sum() + output.presence_logits.sum()).backward()

    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    invalid = raw.detach().clone()
    invalid[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        transform_raw_output(invalid)


def test_single_supervised_forecast_cell_has_exact_weighted_objective() -> None:
    output = _zero_output()
    targets = torch.zeros(1, 5, 5)
    target_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    targets[0, 0, 0] = 1.0
    target_mask[0, 0, 0] = True

    terms = chmt_q_loss(output, targets=targets, target_mask=target_mask)

    torch.testing.assert_close(terms.pinball, torch.tensor(0.5))
    torch.testing.assert_close(terms.median_huber, torch.tensor(0.5))
    torch.testing.assert_close(terms.presence, torch.tensor(0.0))
    torch.testing.assert_close(terms.total, torch.tensor(0.625))


def test_presence_only_supervision_zeroes_absent_forecast_terms() -> None:
    output = _zero_output()
    labels = torch.zeros(1, 5, 2)
    mask = torch.zeros(1, 5, 2, dtype=torch.bool)
    labels[0, 0, 0] = 1.0
    mask[0, 0, 0] = True

    terms = chmt_q_loss(output, presence_labels=labels, presence_mask=mask)

    torch.testing.assert_close(terms.pinball, torch.tensor(0.0))
    torch.testing.assert_close(terms.median_huber, torch.tensor(0.0))
    torch.testing.assert_close(terms.presence, torch.tensor(math.log(2.0)))
    torch.testing.assert_close(terms.total, torch.tensor(0.2 * math.log(2.0)))


def test_no_supervised_term_is_a_controlled_rejection() -> None:
    output = _zero_output()
    targets = torch.zeros(1, 5, 5)
    target_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    labels = torch.zeros(1, 5, 2)
    presence_mask = torch.zeros(1, 5, 2, dtype=torch.bool)

    with pytest.raises(NoSupervisedTermsError, match="no supervised"):
        chmt_q_loss(
            output,
            targets=targets,
            target_mask=target_mask,
            presence_labels=labels,
            presence_mask=presence_mask,
        )


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.25), (60, 0.25), (75, 0.15), (90, 0.05), (150, 0.05)],
)
def test_auxiliary_weight_schedule(step: int, expected: float) -> None:
    assert auxiliary_loss_weight(step, 100) == pytest.approx(expected)


def test_auxiliary_objective_repeats_all_available_final_terms() -> None:
    output = _zero_output()
    targets = torch.zeros(1, 5, 5)
    target_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    targets[0, 0, 0] = 1.0
    target_mask[0, 0, 0] = True

    terms = chmt_q_loss(
        output,
        targets=targets,
        target_mask=target_mask,
        auxiliary_output=output,
        step=0,
        total_steps=100,
    )

    torch.testing.assert_close(terms.final_objective, torch.tensor(0.625))
    torch.testing.assert_close(terms.auxiliary_objective, torch.tensor(0.625))
    assert terms.auxiliary_scale == pytest.approx(0.25)
    torch.testing.assert_close(terms.total, torch.tensor(0.78125))


def test_combined_objective_backpropagates_finite_gradients() -> None:
    raw = torch.randn(2, 5, 17, requires_grad=True)
    auxiliary_raw = torch.randn(2, 5, 17, requires_grad=True)
    output = transform_raw_output(raw)
    auxiliary = transform_raw_output(auxiliary_raw)
    targets = torch.randn(2, 5, 5)
    target_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    labels = torch.randint(0, 2, (2, 5, 2)).float()
    presence_mask = torch.ones(2, 5, 2, dtype=torch.bool)

    terms = chmt_q_loss(
        output,
        targets=targets,
        target_mask=target_mask,
        presence_labels=labels,
        presence_mask=presence_mask,
        auxiliary_output=auxiliary,
        step=50,
        total_steps=100,
    )
    terms.total.backward()

    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert auxiliary_raw.grad is not None and torch.isfinite(auxiliary_raw.grad).all()
    assert torch.isfinite(terms.total)


def test_bfloat16_predictions_accept_float32_labels_and_reduce_in_float32() -> None:
    raw = torch.randn(2, 5, 17, dtype=torch.bfloat16, requires_grad=True)
    output = transform_raw_output(raw)
    targets = torch.randn(2, 5, 5, dtype=torch.float32)
    target_mask = torch.ones(2, 5, 5, dtype=torch.bool)
    labels = torch.randint(0, 2, (2, 5, 2), dtype=torch.int64).float()
    presence_mask = torch.ones(2, 5, 2, dtype=torch.bool)

    terms = chmt_q_loss(
        output,
        targets=targets,
        target_mask=target_mask,
        presence_labels=labels,
        presence_mask=presence_mask,
    )
    terms.total.backward()

    assert output.quantiles.dtype == torch.float32
    assert output.presence_logits.dtype == torch.float32
    assert terms.total.dtype == torch.float32
    assert terms.pinball.dtype == torch.float32
    assert terms.median_huber.dtype == torch.float32
    assert terms.presence.dtype == torch.float32
    assert raw.grad is not None and raw.grad.dtype == torch.bfloat16
    assert torch.isfinite(raw.grad).all()


def test_autocast_promoted_quantiles_also_promote_presence_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_softplus = output_module.F.softplus

    def promote_softplus(value: torch.Tensor) -> torch.Tensor:
        return original_softplus(value.float())

    monkeypatch.setattr(output_module.F, "softplus", promote_softplus)
    raw = torch.randn(2, 5, 17, dtype=torch.bfloat16, requires_grad=True)

    output = transform_raw_output(raw)
    (output.quantiles.sum() + output.presence_logits.sum()).backward()

    assert output.quantiles.dtype == torch.float32
    assert output.presence_logits.dtype == torch.float32
    assert raw.grad is not None and raw.grad.dtype == torch.bfloat16
    assert torch.isfinite(raw.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires a CUDA device",
)
def test_cuda_autocast_output_and_float32_supervision_share_a_stable_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = torch.device("cuda:0")
    projection = torch.nn.Linear(4, 5 * 17, device=device)
    features = torch.randn(2, 4, device=device)
    targets = torch.randn(2, 5, 5, device=device)
    target_mask = torch.ones(2, 5, 5, dtype=torch.bool, device=device)
    labels = torch.randint(0, 2, (2, 5, 2), device=device).float()
    presence_mask = torch.ones(2, 5, 2, dtype=torch.bool, device=device)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        raw = projection(features).reshape(2, 5, 17)
        output = transform_raw_output(raw)
    terms = chmt_q_loss(
        output,
        targets=targets,
        target_mask=target_mask,
        presence_labels=labels,
        presence_mask=presence_mask,
    )
    terms.total.backward()

    assert raw.dtype == torch.bfloat16
    assert output.quantiles.dtype == torch.float32
    assert output.presence_logits.dtype == torch.float32
    assert output.quantiles.device == output.presence_logits.device == device
    assert terms.total.dtype == torch.float32
    assert projection.weight.grad is not None
    assert torch.isfinite(projection.weight.grad).all()


def test_output_contract_rejects_crossed_or_unbounded_delta_quantiles() -> None:
    crossed = _zero_output().quantiles.clone()
    crossed[0, 0, 0] = torch.tensor([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="q10"):
        ChmtQOutput(crossed, torch.zeros(1, 5, 2)).validate()

    unbounded = _zero_output().quantiles.clone()
    unbounded[0, 0, 4] = torch.tensor([-1.1, 0.0, 0.5])
    with pytest.raises(ValueError, match="delta-fraction"):
        ChmtQOutput(unbounded, torch.zeros(1, 5, 2)).validate()


def test_selected_reference_count_includes_both_attention_parameter_families() -> None:
    config = ChmtQConfig()

    assert config.reference_parameter_count == 2_089_518 + 36


def test_every_raw_position_has_the_declared_target_major_meaning() -> None:
    raw = torch.zeros(1, 5, 17)
    expected_quantiles = torch.empty(1, 5, 5, 3)
    for horizon in range(5):
        for target in range(5):
            center = float(10 * horizon + target) / 10.0
            lower_gap = 0.25 + 0.05 * target
            upper_gap = 0.4 + 0.03 * horizon
            offset = 3 * target
            raw[0, horizon, offset] = center
            raw[0, horizon, offset + 1] = math.log(math.expm1(lower_gap))
            raw[0, horizon, offset + 2] = math.log(math.expm1(upper_gap))
            ordered = torch.tensor([center - lower_gap, center, center + upper_gap])
            expected_quantiles[0, horizon, target] = (
                torch.tanh(ordered) if target == 4 else ordered
            )
        raw[0, horizon, 15] = 100.0 + horizon
        raw[0, horizon, 16] = -100.0 - horizon

    output = transform_raw_output(raw)

    torch.testing.assert_close(output.quantiles, expected_quantiles)
    torch.testing.assert_close(output.presence_logits, raw[..., 15:17])


def test_loss_statistics_add_exactly_under_unequal_masks() -> None:
    config = ChmtQConfig()
    first_targets = torch.zeros(1, 5, 5)
    first_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    first_targets[0, 0, 0] = 2.0
    first_mask[0, 0, 0] = True
    second_targets = torch.zeros(1, 5, 5)
    second_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    second_targets[0, 0, 0] = -1.0
    second_targets[0, 1, 4] = 0.5
    second_mask[0, 0, 0] = True
    second_mask[0, 1, 4] = True
    first_presence = torch.ones(1, 5, 2)
    first_presence_mask = torch.zeros(1, 5, 2, dtype=torch.bool)
    first_presence_mask[0, 0, 0] = True
    second_presence = torch.zeros(1, 5, 2)
    second_presence_mask = torch.zeros(1, 5, 2, dtype=torch.bool)
    second_presence_mask[0, 0, 0] = True
    second_presence_mask[0, 2, 1] = True

    first = chmt_q_loss(
        _zero_output(),
        targets=first_targets,
        target_mask=first_mask,
        presence_labels=first_presence,
        presence_mask=first_presence_mask,
    )
    second = chmt_q_loss(
        _zero_output(),
        targets=second_targets,
        target_mask=second_mask,
        presence_labels=second_presence,
        presence_mask=second_presence_mask,
    )
    aggregate = aggregate_chmt_q_loss_statistics(
        (first.statistics, second.statistics)
    )

    manual_pinball_numerator = 0.0
    manual_pinball_denominator = 0.0
    manual_huber_numerator = 0.0
    manual_huber_denominator = 0.0
    supervised = ((0, 0, 2.0), (0, 0, -1.0), (1, 4, 0.5))
    for horizon, target, value in supervised:
        weight = config.horizon_weights[horizon] * config.target_weights[target]
        for level in config.quantile_levels:
            error = value
            manual_pinball_numerator += weight * max(
                level * error, (level - 1.0) * error
            )
            manual_pinball_denominator += weight
        absolute = abs(value)
        huber = 0.5 * value * value if absolute <= 1.0 else absolute - 0.5
        manual_huber_numerator += weight * huber
        manual_huber_denominator += weight

    assert float(aggregate.pinball_numerator) == pytest.approx(
        manual_pinball_numerator
    )
    assert float(aggregate.pinball_denominator) == pytest.approx(
        manual_pinball_denominator
    )
    assert float(aggregate.median_huber_numerator) == pytest.approx(
        manual_huber_numerator
    )
    assert float(aggregate.median_huber_denominator) == pytest.approx(
        manual_huber_denominator
    )
    assert float(aggregate.presence_numerator) == pytest.approx(3.0 * math.log(2.0))
    assert float(aggregate.presence_denominator) == 3.0
    assert aggregate.forecast_counts[0, 0] == 2
    assert aggregate.forecast_counts[1, 4] == 1
    assert aggregate.forecast_counts[4, 4] == 0
    assert aggregate.presence_counts[0, 0] == 2
    assert aggregate.presence_counts[2, 1] == 1
    assert aggregate.presence_counts[4, 1] == 0
    reconstructed = chmt_q_objective_from_statistics(aggregate, config)
    expected = (
        manual_pinball_numerator / manual_pinball_denominator
        + config.median_huber_weight
        * manual_huber_numerator
        / manual_huber_denominator
        + config.presence_weight * math.log(2.0)
    )
    assert float(reconstructed) == pytest.approx(expected)


def test_online_loss_statistics_match_materialized_aggregation() -> None:
    targets = torch.linspace(-1.0, 1.0, steps=50).reshape(2, 5, 5)
    target_mask = torch.ones_like(targets, dtype=torch.bool)
    presence = torch.tensor(
        [0.0, 1.0] * 10, dtype=torch.float32
    ).reshape(2, 5, 2)
    presence_mask = torch.ones_like(presence, dtype=torch.bool)
    statistics = []
    for offset in range(17):
        terms = chmt_q_loss(
            _zero_output(batch_size=2),
            targets=targets + offset * 0.03125,
            target_mask=target_mask,
            presence_labels=presence,
            presence_mask=presence_mask,
        )
        statistics.append(terms.statistics)

    materialized = aggregate_chmt_q_loss_statistics(statistics)
    accumulator = ChmtQLossStatisticsAccumulator()
    for value in statistics:
        accumulator.update(value)
    online = accumulator.finalize()

    assert not hasattr(accumulator, "__dict__")
    assert all(
        not isinstance(getattr(accumulator, name), list)
        for name in ChmtQLossStatisticsAccumulator.__slots__
    )
    for name in (
        "pinball_numerator",
        "pinball_denominator",
        "median_huber_numerator",
        "median_huber_denominator",
        "presence_numerator",
        "presence_denominator",
    ):
        torch.testing.assert_close(
            getattr(online, name), getattr(materialized, name), rtol=1.0e-6, atol=1.0e-5
        )
    assert torch.equal(online.forecast_counts, materialized.forecast_counts)
    assert torch.equal(online.presence_counts, materialized.presence_counts)
    assert online.pinball_numerator.device.type == "cpu"
    assert not online.pinball_numerator.requires_grad
