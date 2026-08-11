from __future__ import annotations

import math

import pytest
import torch

from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.metrics import ChmtQMetricsAccumulator
from daxalgo_numeric_transformer.chmt_q.output import ChmtQOutput


def test_metrics_match_independent_unequal_mask_reference() -> None:
    config = ChmtQConfig()
    quantiles = torch.zeros(2, 5, 5, 3)
    quantiles[:, 0, 0] = torch.tensor([0.0, 1.0, 2.0])
    quantiles[:, 1, 2] = torch.tensor([-1.0, 0.0, 1.0])
    logits = torch.zeros(2, 5, 2)
    logits[1, 0, 0] = math.log(3.0)
    output = ChmtQOutput(quantiles, logits).validate(config)

    targets = torch.zeros(2, 5, 5)
    target_mask = torch.zeros(2, 5, 5, dtype=torch.bool)
    targets[0, 0, 0] = 1.0
    targets[1, 0, 0] = 3.0
    targets[0, 1, 2] = 0.0
    target_mask[0, 0, 0] = True
    target_mask[1, 0, 0] = True
    target_mask[0, 1, 2] = True
    labels = torch.zeros(2, 5, 2)
    presence_mask = torch.zeros(2, 5, 2, dtype=torch.bool)
    labels[0, 0, 0] = 1.0
    labels[1, 0, 0] = 0.0
    presence_mask[:, 0, 0] = True

    accumulator = ChmtQMetricsAccumulator(config)
    for index in range(2):
        accumulator.update(
            ChmtQOutput(
                output.quantiles[index : index + 1],
                output.presence_logits[index : index + 1],
            ),
            targets=targets[index : index + 1],
            target_mask=target_mask[index : index + 1],
            presence_labels=labels[index : index + 1],
            presence_mask=presence_mask[index : index + 1],
        )
    report = accumulator.finalize()

    assert report.forecast_counts[0, 0] == 2
    assert report.pinball[0, 0] == pytest.approx(0.4)
    assert report.median_mae[0, 0] == pytest.approx(1.0)
    assert report.median_rmse[0, 0] == pytest.approx(math.sqrt(2.0))
    assert report.median_bias[0, 0] == pytest.approx(-1.0)
    assert report.interval_coverage[0, 0] == pytest.approx(0.5)
    assert report.interval_width[0, 0] == pytest.approx(2.0)
    assert report.forecast_counts[1, 2] == 1
    assert report.pinball[1, 2] == pytest.approx(0.1 / 3.0 + 0.1 / 3.0)
    assert report.forecast_counts[4, 4] == 0
    assert report.pinball[4, 4] == 0.0
    assert report.median_mae[4, 4] == 0.0
    assert report.interval_coverage[4, 4] == 0.0

    assert report.presence_counts[0, 0] == 2
    assert report.presence_log_loss[0, 0] == pytest.approx(
        (math.log(2.0) + math.log(4.0)) / 2.0
    )
    assert report.presence_brier[0, 0] == pytest.approx((0.25 + 0.75**2) / 2.0)
    assert report.presence_counts[4, 1] == 0
    assert report.presence_log_loss[4, 1] == 0.0
    assert report.presence_brier[4, 1] == 0.0

    weighted_denominator = 6.0 + 3.0 / math.sqrt(2.0)
    weighted_numerator = 2.4 + 0.2 / math.sqrt(2.0)
    assert report.weighted_primary_metric == pytest.approx(
        weighted_numerator / weighted_denominator
    )


def test_metrics_reject_nonfinite_supervised_values() -> None:
    config = ChmtQConfig()
    output = ChmtQOutput(
        torch.zeros(1, 5, 5, 3), torch.zeros(1, 5, 2)
    ).validate(config)
    targets = torch.zeros(1, 5, 5)
    target_mask = torch.zeros(1, 5, 5, dtype=torch.bool)
    targets[0, 0, 0] = torch.nan
    target_mask[0, 0, 0] = True

    with pytest.raises(ValueError, match="NaN or infinity"):
        ChmtQMetricsAccumulator(config).update(
            output,
            targets=targets,
            target_mask=target_mask,
            presence_labels=None,
            presence_mask=None,
        )
