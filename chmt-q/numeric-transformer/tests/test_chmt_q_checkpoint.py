from __future__ import annotations

import random

import numpy as np
import pytest
import torch
from torch import nn

from daxalgo_numeric_transformer.chmt_q.training.checkpoint import (
    CheckpointVerificationError,
    load_checkpoint,
    save_checkpoint,
)


def _components() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(nn.Linear(3, 4), nn.SiLU(), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    return model, optimizer


def test_checkpoint_restores_weights_optimizer_and_rng(tmp_path) -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    model, optimizer = _components()
    inputs = torch.randn(5, 3)
    model(inputs).square().mean().backward()
    optimizer.step()
    expected_output = model(inputs).detach().clone()
    path = save_checkpoint(
        tmp_path,
        step=9,
        model=model,
        optimizer=optimizer,
        run_identity={"dataset": "abc", "config": "def"},
        trainer_state={"epoch": 2, "best": 0.75},
    )
    expected_random = (random.random(), float(np.random.random()), torch.rand(1).item())

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(100.0)
    random.random()
    np.random.random()
    torch.rand(1)
    restored = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_identity={"config": "def", "dataset": "abc"},
    )

    torch.testing.assert_close(model(inputs), expected_output, rtol=0, atol=0)
    assert restored == {"step": 9, "trainer_state": {"best": 0.75, "epoch": 2}}
    actual_random = (random.random(), float(np.random.random()), torch.rand(1).item())
    assert actual_random == pytest.approx(expected_random)


def test_checkpoint_rejects_tampering_before_loading(tmp_path) -> None:
    model, optimizer = _components()
    path = save_checkpoint(
        tmp_path,
        step=1,
        model=model,
        optimizer=optimizer,
        run_identity={"dataset": "abc"},
    )
    with (path / "weights.npz").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(CheckpointVerificationError, match="hash mismatch"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_identity={"dataset": "abc"},
        )


def test_checkpoint_never_overwrites_and_checks_identity(tmp_path) -> None:
    model, optimizer = _components()
    path = save_checkpoint(
        tmp_path,
        step=3,
        model=model,
        optimizer=optimizer,
        run_identity={"dataset": "abc"},
    )
    with pytest.raises(FileExistsError):
        save_checkpoint(
            tmp_path,
            step=3,
            model=model,
            optimizer=optimizer,
            run_identity={"dataset": "abc"},
        )
    with pytest.raises(CheckpointVerificationError, match="identity"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_identity={"dataset": "different"},
        )
