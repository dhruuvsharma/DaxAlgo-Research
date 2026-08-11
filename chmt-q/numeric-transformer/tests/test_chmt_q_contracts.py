from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from daxalgo_numeric_transformer.chmt_q import (
    ChmtQBatch,
    ChmtQConfig,
    NoAvailableModalitiesError,
    make_chmt_q_synthetic_batch,
)


def _batch(batch_size: int = 2) -> ChmtQBatch:
    l2_level_mask = torch.ones(batch_size, 240, 2, 10, dtype=torch.bool)
    targets = torch.randn(batch_size, 5, 5)
    targets[..., 4] = torch.rand(batch_size, 5) * 2.0 - 1.0
    bars = torch.randn(batch_size, 64, 16)
    bars[..., 9] = torch.sigmoid(bars[..., 9])
    bars[..., 13] = torch.sigmoid(bars[..., 13])
    bars[..., 14:16] = torch.tanh(bars[..., 14:16])
    l1 = torch.randn(batch_size, 240, 16)
    l1[..., 5] = torch.tanh(l1[..., 5])
    l1[..., 13] = torch.sigmoid(l1[..., 13])
    l1[..., 14:16] = torch.randint(0, 2, l1[..., 14:16].shape).float()
    l2 = torch.randn(batch_size, 240, 2, 10, 12)
    l2[..., 4:6] = torch.sigmoid(l2[..., 4:6])
    l2[..., 6] = torch.tanh(l2[..., 6])
    l2[..., 9] = torch.sigmoid(l2[..., 9])
    l2[..., 10:12] = torch.randint(0, 2, l2[..., 10:12].shape).float()
    return ChmtQBatch(
        bars=bars,
        bars_mask=torch.ones(batch_size, 64, dtype=torch.bool),
        l1=l1,
        l1_mask=torch.ones(batch_size, 240, dtype=torch.bool),
        l2=l2,
        l2_level_mask=l2_level_mask,
        l2_snapshot_mask=l2_level_mask.any(dim=(-1, -2)),
        quality=torch.rand(batch_size, 16),
        targets=targets,
        target_mask=torch.ones(batch_size, 5, 5, dtype=torch.bool),
        presence_labels=torch.randint(0, 2, (batch_size, 5, 2)).float(),
        presence_mask=torch.ones(batch_size, 5, 2, dtype=torch.bool),
    )


def test_batch_one_synthetic_default_exercises_present_two_sided_l2() -> None:
    batch = make_chmt_q_synthetic_batch(ChmtQConfig.tiny(), batch_size=1, seed=103)

    assert batch.l2_snapshot_mask.any()
    assert batch.l2_level_mask[0, :, 0, :].any()
    assert batch.l2_level_mask[0, :, 1, :].any()


def test_config_is_immutable_and_fixes_the_selected_contract() -> None:
    config = ChmtQConfig()

    assert config.reference_parameter_count == 2_089_554
    assert config.raw_output_features == config.targets * config.quantiles + 2
    assert config.TARGET_ORDER[-1] == "delta_fraction"
    assert config.d_model == 128 and config.ff_dim == 256
    assert config.memory_tokens == 184
    assert config.bar_ages_seconds == tuple(float(value) for value in range(3780, -1, -60))
    assert config.patch_ages_seconds == tuple(float(value) for value in range(118, -1, -2))
    with pytest.raises(FrozenInstanceError):
        config.bar_steps = 63  # type: ignore[misc]
    with pytest.raises(ValueError, match="bar_steps is fixed"):
        replace(config, bar_steps=63)
    with pytest.raises(ValueError, match="reference_parameter_count is fixed"):
        replace(config, reference_parameter_count=2_089_518)
    with pytest.raises(ValueError, match="bar_time_scales is fixed"):
        replace(config, bar_time_scales=(60.0,))


def test_tiny_factory_only_changes_validated_architecture_dimensions() -> None:
    tiny = ChmtQConfig.tiny(d_model=16, num_heads=4)

    assert tiny.d_model == 16
    assert tiny.bar_steps == 64
    assert tiny.raw_output_features == 17
    assert tiny.bar_time_scales == ChmtQConfig().bar_time_scales
    with pytest.raises(TypeError, match="forbidden: bar_steps"):
        ChmtQConfig.tiny(bar_steps=8)
    with pytest.raises(TypeError, match="unknown"):
        ChmtQConfig.tiny(not_a_setting=1)
    with pytest.raises(ValueError, match="divisible"):
        ChmtQConfig.tiny(d_model=15)
    with pytest.raises(ValueError, match="rms_epsilon"):
        replace(tiny, rms_epsilon=0.0)


def test_batch_validates_all_fixed_shapes_and_stable_model_inputs() -> None:
    batch = _batch().validate(
        require_targets=True,
        require_presence=True,
        require_supervision=True,
    )

    assert batch.bars.shape == (2, 64, 16)
    assert batch.l1.shape == (2, 240, 16)
    assert batch.l2.shape == (2, 240, 2, 10, 12)
    assert batch.quality.shape == (2, 16)
    assert batch.targets is not None and batch.targets.shape == (2, 5, 5)
    assert batch.presence_labels is not None
    assert batch.presence_labels.shape == (2, 5, 2)
    assert tuple(batch.as_model_kwargs()) == (
        "bars",
        "bars_mask",
        "l1",
        "l1_mask",
        "l2",
        "l2_level_mask",
        "l2_snapshot_mask",
        "quality",
    )


def test_forecast_and_presence_supervision_are_independent() -> None:
    batch = _batch()
    assert batch.target_mask is not None
    batch.target_mask.zero_()

    batch.validate(require_presence=True, require_supervision=True)
    assert batch.has_supervision()


def test_masked_unknown_labels_may_be_nonfinite_but_valid_labels_may_not() -> None:
    batch = _batch(batch_size=1)
    assert batch.targets is not None and batch.target_mask is not None
    batch.targets[0, 0, 0] = torch.nan
    batch.target_mask[0, 0, 0] = False
    batch.validate()

    batch.target_mask[0, 0, 0] = True
    with pytest.raises(ValueError, match="supervised cell"):
        batch.validate()


def test_nested_l2_mask_must_match_level_availability_exactly() -> None:
    batch = _batch(batch_size=1)
    batch.l2_snapshot_mask[0, 0] = False

    with pytest.raises(ValueError, match="exactly equal"):
        batch.validate()


def test_each_sample_requires_at_least_one_hard_available_modality() -> None:
    batch = _batch()
    batch.bars_mask[1].zero_()
    batch.l1_mask[1].zero_()
    batch.l2_level_mask[1].zero_()
    batch.l2_snapshot_mask[1].zero_()

    with pytest.raises(NoAvailableModalitiesError, match=r"missing=\[1\]"):
        batch.validate()


def test_contract_rejects_wrong_dtype_shape_and_nonbinary_presence() -> None:
    wrong_dtype = _batch(batch_size=1)
    wrong_dtype.quality = wrong_dtype.quality.double()
    with pytest.raises(TypeError, match="quality"):
        wrong_dtype.validate()

    wrong_shape = _batch(batch_size=1)
    wrong_shape.l1 = wrong_shape.l1[:, :-1]
    with pytest.raises(ValueError, match="l1 must have shape"):
        wrong_shape.validate()

    nonbinary = _batch(batch_size=1)
    assert nonbinary.presence_labels is not None
    nonbinary.presence_labels[0, 0, 0] = 0.5
    with pytest.raises(ValueError, match="binary"):
        nonbinary.validate()


def test_unscaled_feature_domains_and_invalid_rows_fail_closed() -> None:
    out_of_range = _batch(batch_size=1)
    out_of_range.bars[0, 0, 13] = -0.1
    with pytest.raises(ValueError, match="bars feature 13"):
        out_of_range.validate(require_feature_domains=True)

    nonbinary = _batch(batch_size=1)
    nonbinary.l1[0, 0, 14] = 0.5
    with pytest.raises(ValueError, match="l1 feature 14 must be binary"):
        nonbinary.validate(require_feature_domains=True)

    invalid_l2 = _batch(batch_size=1)
    invalid_l2.l2[0, 0, 0, 0, 6] = 1.1
    with pytest.raises(ValueError, match="l2 feature 6"):
        invalid_l2.validate(require_feature_domains=True)


@pytest.mark.parametrize("value", [-0.0001, 1.0001])
def test_quality_values_must_remain_bounded(value: float) -> None:
    batch = _batch(batch_size=1)
    batch.quality[0, 3] = value

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        batch.validate()


@pytest.mark.parametrize("value", [-1.0001, 1.0001])
def test_supervised_delta_fraction_must_remain_bounded(value: float) -> None:
    batch = _batch(batch_size=1)
    assert batch.targets is not None and batch.target_mask is not None
    batch.targets[0, 0, 4] = value

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        batch.validate()

    batch.target_mask[0, 0, 4] = False
    batch.validate()
