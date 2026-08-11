from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from daxalgo_numeric_transformer.chmt_q import baselines as baseline_module
from daxalgo_numeric_transformer.chmt_q.baselines import (
    EmpiricalBaseline,
    ZeroResidualBaseline,
)
from daxalgo_numeric_transformer.chmt_q.calibration import ConformalCalibration
from daxalgo_numeric_transformer.chmt_q.config import ChmtQConfig
from daxalgo_numeric_transformer.chmt_q.output import ChmtQOutput
from daxalgo_numeric_transformer.chmt_q.synthetic import make_chmt_q_synthetic_batch


def test_empirical_baseline_is_mask_correct_and_ordered() -> None:
    config = ChmtQConfig.tiny()
    batch = make_chmt_q_synthetic_batch(config, batch_size=8, seed=211)
    baseline = EmpiricalBaseline.fit(
        [batch], source_example_set_sha256="a" * 64, config=config
    )
    output = baseline.predict(3, config=config)
    assert output.quantiles.shape == (3, 5, 5, 3)
    assert bool((output.q10 <= output.median).all())
    assert bool((output.median <= output.q90).all())
    assert all(count > 0 for count in baseline.target_counts)
    assert len(baseline.artifact_sha256) == 64
    assert EmpiricalBaseline.from_json(baseline.to_json()) == baseline

    tampered = json.loads(baseline.to_json())
    tampered["payload"]["quantiles"][0] += 1.0
    with pytest.raises(ValueError, match="SHA-256"):
        EmpiricalBaseline.from_json(
            json.dumps(tampered, separators=(",", ":"), sort_keys=True)
        )

    zero = ZeroResidualBaseline.from_empirical(baseline).predict(3, config=config)
    assert torch.count_nonzero(zero.quantiles) == 0
    torch.testing.assert_close(zero.presence_logits, output.presence_logits)


def test_empirical_baseline_disk_statistics_match_materialized_reference() -> None:
    config = ChmtQConfig.tiny()
    batches = [
        make_chmt_q_synthetic_batch(config, batch_size=7, seed=seed)
        for seed in (227, 229, 233)
    ]

    baseline = EmpiricalBaseline.fit(
        iter(batches), source_example_set_sha256="c" * 64, config=config
    )

    expected_quantiles: list[float] = []
    expected_target_counts: list[int] = []
    for horizon in range(config.horizons):
        for target in range(config.targets):
            values = [
                float(value)
                for batch in batches
                for value in batch.targets[:, horizon, target][
                    batch.target_mask[:, horizon, target]
                ]
            ]
            expected_target_counts.append(len(values))
            expected_quantiles.extend(
                float(value)
                for value in np.quantile(
                    np.asarray(values, dtype=np.float64), config.quantile_levels
                )
            )

    expected_presence_counts: list[int] = []
    expected_presence_logits: list[float] = []
    for horizon in range(config.horizons):
        for target in range(config.presence_targets):
            values = [
                float(value)
                for batch in batches
                for value in batch.presence_labels[:, horizon, target][
                    batch.presence_mask[:, horizon, target]
                ]
            ]
            expected_presence_counts.append(len(values))
            probability = (sum(values) + 0.5) / (len(values) + 1.0)
            expected_presence_logits.append(
                math.log(probability / (1.0 - probability))
            )

    assert baseline.quantiles == tuple(expected_quantiles)
    assert baseline.target_counts == tuple(expected_target_counts)
    assert baseline.presence_counts == tuple(expected_presence_counts)
    assert baseline.presence_logits == tuple(expected_presence_logits)


def test_empirical_baseline_spools_without_numpy_full_set_materialization(
    monkeypatch,
) -> None:
    config = ChmtQConfig.tiny()
    batch = make_chmt_q_synthetic_batch(config, batch_size=4, seed=239)
    real_connect = baseline_module.sqlite3.connect
    databases: list[Path] = []

    def tracked_connect(database, *args, **kwargs):
        databases.append(Path(database))
        return real_connect(database, *args, **kwargs)

    def reject_materialization(*_args, **_kwargs):
        raise AssertionError("full target-set materialization was attempted")

    monkeypatch.setattr(baseline_module.sqlite3, "connect", tracked_connect)
    monkeypatch.setattr(np, "asarray", reject_materialization)
    monkeypatch.setattr(np, "quantile", reject_materialization)

    baseline = EmpiricalBaseline.fit(
        (batch for _ in range(32)),
        source_example_set_sha256="d" * 64,
        config=config,
    )

    expected_counts = tuple(
        int(batch.target_mask[:, horizon, target].sum()) * 32
        for horizon in range(config.horizons)
        for target in range(config.targets)
    )
    assert baseline.target_counts == expected_counts
    assert databases and all(str(path) != ":memory:" for path in databases)
    assert all(not path.exists() for path in databases)


def test_conformal_calibration_uses_finite_sample_quantile_and_expands() -> None:
    config = ChmtQConfig.tiny()
    batch = make_chmt_q_synthetic_batch(config, batch_size=8, seed=223)
    assert batch.targets is not None
    center = batch.targets.clone()
    raw_quantiles = torch.stack((center - 0.1, center, center + 0.1), dim=-1)
    raw_quantiles[..., config.delta_target_index, :] = raw_quantiles[
        ..., config.delta_target_index, :
    ].clamp(-1.0, 1.0)
    output = ChmtQOutput(
        raw_quantiles,
        torch.zeros(8, config.horizons, config.presence_targets),
    ).validate(config)
    shifted = ChmtQOutput(
        output.quantiles - 0.2,
        output.presence_logits,
    )
    shifted_quantiles = shifted.quantiles.clone()
    shifted_quantiles[..., config.delta_target_index, :] = shifted_quantiles[
        ..., config.delta_target_index, :
    ].clamp(-1.0, 1.0)
    shifted = ChmtQOutput(shifted_quantiles, shifted.presence_logits).validate(config)
    artifact = ConformalCalibration.fit(
        [(shifted, batch)],
        source_calibration_set_sha256="b" * 64,
        config=config,
    )
    calibrated = artifact.apply(shifted, config)
    assert bool((calibrated.q10 <= shifted.q10).all())
    assert bool((calibrated.q90 >= shifted.q90).all())
    assert all(count > 0 for count in artifact.counts)
    assert len(artifact.artifact_sha256) == 64
    assert ConformalCalibration.from_json(artifact.to_json()) == artifact
