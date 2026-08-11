from __future__ import annotations

import math

from test_chmt_q_runner import _ready_run

from daxalgo_numeric_transformer.chmt_q.training.runner import (
    run_development_experiment,
)


def test_development_runner_emits_deterministic_progress_events(tmp_path) -> None:
    dataset, split, scaler, config, source_root, dependency_lock = _ready_run(
        tmp_path, max_epochs=1
    )
    events: list[dict[str, object]] = []

    result = run_development_experiment(
        dataset,
        split,
        scaler,
        config,
        fold_number=1,
        output_root=tmp_path / "runs",
        source_root=source_root,
        dependency_lock=dependency_lock,
        progress_callback=events.append,
    )

    assert [(event["phase"], event["status"]) for event in events] == [
        ("baseline", "start"),
        ("epoch", "start"),
        ("optimizer_window", "progress"),
        ("optimizer_window", "progress"),
        ("validation", "complete"),
        ("checkpoint", "committed"),
        ("epoch", "committed"),
    ]
    common = {"run_id": config.run_id, "fold_id": config.fold_id}
    assert events[0] == {
        "phase": "baseline",
        "status": "start",
        **common,
        "max_epochs": 1,
    }
    assert events[1] == {
        "phase": "epoch",
        "status": "start",
        **common,
        "epoch": 1,
        "max_epochs": 1,
        "optimizer_step": 0,
    }
    assert events[2:4] == [
        {
            "phase": "optimizer_window",
            "status": "progress",
            **common,
            "epoch": 1,
            "max_epochs": 1,
            "windows_per_epoch": 2,
            "window": window,
            "optimizer_step": window,
        }
        for window in (1, 2)
    ]

    validation = events[4]
    assert math.isfinite(validation["primary_metric"])
    assert validation == {
        "phase": "validation",
        "status": "complete",
        **common,
        "epoch": 1,
        "max_epochs": 1,
        "optimizer_step": 2,
        "primary_metric_name": config.primary_metric,
        "primary_metric": validation["primary_metric"],
        "improved": True,
    }
    assert events[5] == {
        "phase": "checkpoint",
        "status": "committed",
        **common,
        "epoch": 1,
        "max_epochs": 1,
        "optimizer_step": 2,
        "checkpoint": str(result.best_checkpoint),
        "improved": True,
    }
    assert events[6] == {
        "phase": "epoch",
        "status": "committed",
        **common,
        "epoch": 1,
        "max_epochs": 1,
        "optimizer_step": 2,
        "checkpoint_due": True,
        "improved": True,
        "stopped_early": False,
    }
