from __future__ import annotations

import pytest
import torch

from daxalgo_numeric_transformer.config import ModelConfig
from daxalgo_numeric_transformer.synthetic import make_synthetic_batch


@pytest.fixture(scope="session")
def tiny_config() -> ModelConfig:
    return ModelConfig.tiny()


@pytest.fixture
def market_batch(tiny_config: ModelConfig):
    return make_synthetic_batch(tiny_config, batch_size=2, seed=41, device=torch.device("cpu"))

