from __future__ import annotations

import math
from contextlib import contextmanager

import pytest
import torch
from torch import nn

from daxalgo_numeric_transformer.chmt_q.training import (
    TrainingConfig,
    build_parameter_groups,
    learning_rate_at_step,
    seed_everything,
    validate_training_runtime,
)


class _GroupingFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.embedding = nn.Parameter(torch.ones(3, 4))
        self.layer_scale = nn.Parameter(torch.ones(4))
        self.log_temperature = nn.Parameter(torch.zeros(2))


def test_training_config_is_strict_and_effective_batch_is_128() -> None:
    config = TrainingConfig()

    assert config.effective_batch_size == 128
    with pytest.raises(ValueError, match="unknown training"):
        TrainingConfig.from_mapping({"learnng_rate": 1.0e-3})


def test_runtime_preflight_rejects_unreproducible_cuda_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert validate_training_runtime(TrainingConfig(), "cpu") == torch.device("cpu")
    with pytest.raises(ValueError, match="requires a CUDA"):
        validate_training_runtime(TrainingConfig(mixed_precision=True), "cpu")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    native_bf16_queries: list[bool] = []

    def bf16_supported(*, including_emulation: bool = True) -> bool:
        native_bf16_queries.append(including_emulation)
        return including_emulation

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", bf16_supported)
    selected_devices: list[str] = []

    @contextmanager
    def selected_device(device: torch.device):
        selected_devices.append(str(device))
        yield

    monkeypatch.setattr(torch.cuda, "device", selected_device)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        validate_training_runtime(TrainingConfig(), "cuda:0")

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(ValueError, match="bf16"):
        validate_training_runtime(TrainingConfig(mixed_precision=True), "cuda:0")
    assert selected_devices == ["cuda:0"]
    assert native_bf16_queries == [False]
    with pytest.raises(ValueError, match="index"):
        validate_training_runtime(TrainingConfig(), "cuda:1")


def test_runtime_preflight_legacy_probe_rejects_emulated_bf16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    @contextmanager
    def selected_device(_device: torch.device):
        yield

    monkeypatch.setattr(torch.cuda, "device", selected_device)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (7, 5))

    with pytest.raises(ValueError, match="native bf16"):
        validate_training_runtime(TrainingConfig(mixed_precision=True), "cuda:0")

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 0))
    assert validate_training_runtime(
        TrainingConfig(mixed_precision=True), "cuda:0"
    ) == torch.device("cuda:0")


def test_parameter_groups_decay_only_plain_matrix_weights() -> None:
    model = _GroupingFixture()
    groups = build_parameter_groups(model, 0.05)
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    decay_names = {names[id(parameter)] for parameter in groups[0]["params"]}
    no_decay_names = {names[id(parameter)] for parameter in groups[1]["params"]}

    assert decay_names == {"projection.weight"}
    assert no_decay_names == {
        "projection.bias",
        "norm.weight",
        "norm.bias",
        "embedding",
        "layer_scale",
        "log_temperature",
    }
    assert groups[0]["weight_decay"] == 0.05
    assert groups[1]["weight_decay"] == 0.0


def test_reference_schedule_warms_then_cosine_decays_to_floor() -> None:
    config = TrainingConfig(warmup_fraction=0.10)
    rates = [learning_rate_at_step(step, 100, config) for step in range(101)]

    assert rates[0] == pytest.approx(config.learning_rate / 10)
    assert rates[9] == pytest.approx(config.learning_rate)
    assert rates[10] == pytest.approx(config.learning_rate)
    assert rates[-1] == pytest.approx(config.minimum_learning_rate)
    assert all(math.isfinite(rate) and rate > 0.0 for rate in rates)
    assert all(left >= right for left, right in zip(rates[10:], rates[11:]))


def test_seed_everything_repeats_cpu_randomness() -> None:
    seed_everything(71)
    first = torch.randn(5)
    seed_everything(71)
    second = torch.randn(5)

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
