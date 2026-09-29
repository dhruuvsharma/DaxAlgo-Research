"""Render training/testing dashboards from the frozen CHMT-Q study artifacts.

Read-only over the local study directory. No artifact is modified and no
checkpoint, dataset, or holdout state is touched. All figures are derived
from immutable run telemetry, evaluation, backtest, and failure-analysis
artifacts of the frozen June 2026 fold-01 seed-17 study.

Usage:
    python visualize_chmtq_study.py --study <frozen-study-dir> --output <dir>

Requires matplotlib (not part of the reviewed training environment).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

TARGET_NAMES = (
    "total POC",
    "buy POC",
    "sell POC",
    "log-vol residual",
    "delta fraction",
)
HORIZON_LABELS = ("H1", "H2", "H3", "H4", "H5")
HORIZON_MINUTES = (1, 2, 3, 4, 5)
COST_SCENARIOS = ("frictionless", "optimistic", "base", "stress")
COST_BPS = {"frictionless": 0.0, "optimistic": 9.0, "base": 12.0, "stress": 20.0}
PARTITION_LABELS = {
    "in_sample_train_replay": "In-sample train replay",
    "selected_validation": "Selected validation",
    "untouched_development_forward": "Development-forward (test)",
}
POLICY_LABELS = {
    "median_long_flat": "Median long/flat",
    "median_long_short": "Median long/short",
    "interval_long_flat": "Interval long/flat",
    "interval_long_short": "Interval long/short",
    "base_cost_hurdle_long_flat": "Cost-hurdle long/flat",
}
GRID_BLUE = "#1f6f8b"
GRID_TEAL = "#3aa17e"
GRID_AMBER = "#e0a458"
GRID_RED = "#c65d7b"
GRID_GRAY = "#8a8f98"
GRID_DARK = "#22303c"
GRID_BG = "#f7f8fa"


def load_json(path: Path) -> Any:
    """Load a JSON artifact tolerating the UTF encodings used by local tooling."""

    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise ValueError(f"cannot decode JSON artifact: {path}")


def _numbers(values: Any) -> np.ndarray:
    """Convert string rows ("a b c") or lists into a float array."""

    if isinstance(values[0], str):
        rows = [line.split() for line in values]
        return np.asarray(rows, dtype=np.float64)
    return np.asarray(values, dtype=np.float64)


class Study:
    """Read-only accessor over the frozen study directory."""

    def __init__(self, root: Path) -> None:
        self.root = root
        run_dirs = sorted((root / "runs").glob("*"))
        if not run_dirs:
            raise FileNotFoundError(f"no run directory under {root / 'runs'}")
        self.run_dir = run_dirs[0]
        self.run_id = self.run_dir.name
        self.epochs = [
            load_json(p)["payload"] for p in sorted((self.run_dir / "epochs").glob("epoch-*.json"))
        ]
        self.run = load_json(self.run_dir / "run.json")["payload"]
        config = load_json(self.run_dir / "config.json")["payload"]
        self.training_config = config.get("training", config.get("training_config", {}))
        evaluation_path = root / "evaluation-fold-01-seed-17.json"
        if not evaluation_path.exists():
            candidates = sorted(root.glob("evaluation-*.json"))
            if not candidates:
                raise FileNotFoundError("evaluation artifact not found")
            evaluation_path = candidates[0]
        self.evaluation = load_json(evaluation_path)
        backtest_dirs = sorted(root.glob("backtest-*"))
        if not backtest_dirs:
            raise FileNotFoundError("backtest artifact not found")
        backtest_dir = sorted(backtest_dirs)[-1]
        self.backtest = load_json(backtest_dir / "results.json")
        self.predictions_path = backtest_dir / "predictions.jsonl.gz"
        diagnostics_dirs = sorted(root.glob("failure-analysis-*"))
        if not diagnostics_dirs:
            raise FileNotFoundError("failure-analysis artifact not found")
        self.diagnostics = load_json(
            (sorted(diagnostics_dirs)[-1] / "diagnostics.json")
        )

    # -- training telemetry -------------------------------------------------

    @property
    def windows(self) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for payload in self.epochs:
            collected.extend(payload["windows"])
        return collected

    def window_series(self, key: str) -> np.ndarray:
        return np.asarray([w[key] for w in self.windows], dtype=np.float64)

    def epoch_summary(self, key: str) -> list[tuple[int, Any]]:
        return [
            (int(payload["epoch"]), payload.get(key))
            for payload in self.epochs
            if payload.get(key) is not None
        ]

    def validation_series(self, key: str) -> np.ndarray:
        return np.asarray(
            [payload["validation"][key] for payload in self.epochs], dtype=np.float64
        )

    def validation_metrics(self, name: str) -> np.ndarray:
        grids = [
            _numbers(payload["validation"]["metrics"][name]) for payload in self.epochs
        ]
        return np.stack(grids)

    def physical_metric_grids(self, name: str) -> np.ndarray:
        grids = [
            _numbers(payload["validation"]["physical_metrics"][name])
            for payload in self.epochs
        ]
        return np.stack(grids)

    def gate_means(self) -> np.ndarray:
        rows = []
        for payload in self.epochs:
            means = payload["telemetry"]["modality_gate_means"]
            rows.append([float(x) for x in means])
        return np.asarray(rows, dtype=np.float64)

    def peak_memory_mb(self) -> np.ndarray:
        return np.asarray(
            [payload["telemetry"]["cuda_peak_memory_bytes"] for payload in self.epochs],
            dtype=np.float64,
        ) / (1024.0 * 1024.0)

    def elapsed_minutes(self) -> np.ndarray:
        return np.asarray(
            [payload["elapsed_seconds"] for payload in self.epochs], dtype=np.float64
        ) / 60.0

    # -- backtest ------------------------------------------------------------

    def partition_forecast(self, partition: str, model: str, space: str) -> dict[str, Any]:
        return self.backtest["forecast"][partition][model][space]

    def partition_economics(self, partition: str) -> dict[str, Any]:
        return self.backtest["economics"][partition]


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#d9dee4", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _figure(width: float, height: float) -> tuple[plt.Figure, Any]:
    fig = plt.figure(figsize=(width, height), facecolor="white")
    return fig, fig.add_gridspec


def _save(fig: plt.Figure, output: Path, name: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    target = output / name
    fig.savefig(target, dpi=160, facecolor="white")
    plt.close(fig)
    return target


def render_training_overview(study: Study, output: Path) -> Path:
    """Page 1: optimizer telemetry across all 18 epochs (3,618 windows)."""

    windows = study.windows
    steps = study.window_series("optimizer_step")
    loss = study.window_series("loss")
    pinball = study.window_series("pinball")
    huber = study.window_series("median_huber")
    presence = study.window_series("presence")
    grad_norm = np.clip(study.window_series("gradient_norm"), 1e-8, None)
    update_norm = np.clip(study.window_series("update_norm"), 1e-8, None)
    lr = study.window_series("learning_rate")
    aux = study.window_series("auxiliary_scale")

    fig = plt.figure(figsize=(13.5, 9.5), facecolor="white")
    gs = fig.add_gridspec(3, 3, hspace=0.42, wspace=0.3, left=0.06, right=0.97,
                          top=0.9, bottom=0.06)
    fig.suptitle(
        "CHMT-Q Training — Frozen Study June 2026  |  "
        f"run {study.run_id[:12]}…  |  fold-01, seed 17, "
        f"{len(windows)} optimizer steps over {len(study.epochs)} epochs",
        fontsize=13,
        fontweight="bold",
        color=GRID_DARK,
    )

    ax = fig.add_subplot(gs[0, 0])
    _style_axes(ax)
    ax.plot(steps, loss, color=GRID_BLUE, linewidth=1.2)
    ax.set_yscale("log")
    ax.set_title("Total training loss (log)", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("optimizer step", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    _style_axes(ax)
    ax.plot(steps, pinball, color=GRID_TEAL, linewidth=1.2, label="pinball")
    ax.plot(steps, huber, color=GRID_AMBER, linewidth=1.2, label="median Huber")
    ax.plot(steps, presence, color=GRID_RED, linewidth=1.2, label="presence BCE")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Loss components (log)", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("optimizer step", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    _style_axes(ax)
    ax.plot(steps, lr, color=GRID_DARK, linewidth=1.4)
    peak = float(lr.max())
    warmup_end = int(steps[int(np.argmax(lr == peak))]) if np.any(lr == peak) else 0
    ax.axvspan(0, warmup_end, color=GRID_AMBER, alpha=0.15)
    ax.annotate(
        f"warm-up → peak {peak:.1e}",
        xy=(warmup_end, peak),
        xytext=(warmup_end + 260, peak * 0.92),
        fontsize=8,
        color=GRID_DARK,
        arrowprops={"arrowstyle": "->", "color": GRID_GRAY, "lw": 0.8},
    )
    ax.set_title("Learning-rate schedule (warm-up + cosine)", fontsize=10.5, color=GRID_DARK)
    ax.set_xlabel("optimizer step", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    _style_axes(ax)
    ax.plot(steps, grad_norm, color=GRID_BLUE, linewidth=1.0, label="grad ‖·‖ (clipped @1.0)")
    ax.axhline(1.0, color=GRID_RED, linewidth=1.0, linestyle="--", label="clip threshold")
    ax.set_yscale("log")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Gradient norm vs clip", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("optimizer step", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    _style_axes(ax)
    clip_fraction = float(np.mean(grad_norm > 1.0) * 100.0)
    ax.plot(steps, update_norm, color=GRID_TEAL, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_title(
        f"Optimizer update norm (log)  |  {clip_fraction:.1f}% windows clipped",
        fontsize=10.5,
        color=GRID_DARK,
    )
    ax.set_xlabel("optimizer step", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    _style_axes(ax)
    ax.plot(steps, aux, color=GRID_AMBER, linewidth=1.4)
    ax.set_title("Auxiliary deep-supervision scale (0.25 → 0.05)", fontsize=10.5,
                 color=GRID_DARK)
    ax.set_xlabel("optimizer step", fontsize=9)

    ax = fig.add_subplot(gs[2, 0])
    _style_axes(ax)
    elapsed = study.elapsed_minutes()
    epochs_axis = np.arange(1, len(elapsed) + 1)
    ax.bar(epochs_axis, elapsed, color=GRID_BLUE, width=0.65)
    ax.set_title(f"Wall-clock per epoch  |  total {elapsed.sum():.0f} min", fontsize=11,
                 color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("minutes", fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    _style_axes(ax)
    memory = study.peak_memory_mb()
    ax.plot(epochs_axis, memory, marker="o", markersize=3.5, color=GRID_TEAL,
            linewidth=1.2)
    ax.set_title("CUDA peak memory", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("MiB", fontsize=9)

    ax = fig.add_subplot(gs[2, 2])
    _style_axes(ax)
    gates = study.gate_means()
    for i, (label, color) in enumerate(
        (("bars", GRID_BLUE), ("L1", GRID_TEAL), ("L2", GRID_AMBER))
    ):
        ax.plot(epochs_axis, gates[:, i], marker="o", markersize=3.2,
                color=color, linewidth=1.2, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.set_title("Quality-gate availability means per epoch", fontsize=10.5, color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)

    return _save(fig, output, "01_training_overview.png")


def render_validation_curves(study: Study, output: Path) -> Path:
    """Page 2: epoch-end validation primary metric + component losses + selection."""

    epochs_axis = np.arange(1, len(study.epochs) + 1)
    primary = study.validation_series("primary_metric")
    pinball = study.validation_series("pinball")
    huber = study.validation_series("median_huber")
    presence = study.validation_series("presence")
    tie = study.validation_series("tie_breaker_median_mae")
    best_epochs = [
        int(payload["selection"]["best_epoch"]) for payload in study.epochs
    ]
    best_metrics = [
        float(payload["selection"]["best_metric"]) for payload in study.epochs
    ]
    baseline = float(study.evaluation["baseline_primary_metric"])
    eval_primary = float(study.evaluation["primary_metric"])
    best_epoch = best_epochs[-1]

    fig = plt.figure(figsize=(13.5, 8.6), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3, left=0.06, right=0.97,
                          top=0.88, bottom=0.08)
    fig.suptitle(
        "CHMT-Q Validation — Selection Metric per Epoch  |  selected checkpoint: "
        f"epoch {best_epoch} @ step 2,010 (best weighted normalized pinball)",
        fontsize=13,
        fontweight="bold",
        color=GRID_DARK,
    )

    ax = fig.add_subplot(gs[0, 0])
    _style_axes(ax)
    ax.plot(epochs_axis, primary, marker="o", markersize=4, color=GRID_BLUE,
            linewidth=1.6, label="validation weighted pinball")
    ax.axhline(baseline, color=GRID_RED, linestyle="--", linewidth=1.2,
               label=f"empirical baseline {baseline:.4f}")
    ax.scatter([best_epoch], [eval_primary], s=90, facecolor="none",
               edgecolor=GRID_TEAL, linewidth=2.0, zorder=5,
               label=f"selected {eval_primary:.4f}")
    ax.set_title("Selection metric vs baseline", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = fig.add_subplot(gs[0, 1])
    _style_axes(ax)
    ax.plot(epochs_axis, pinball, marker="o", markersize=3.5, color=GRID_TEAL,
            linewidth=1.4, label="pinball")
    ax.plot(epochs_axis, huber, marker="s", markersize=3.5, color=GRID_AMBER,
            linewidth=1.4, label="median Huber")
    ax.plot(epochs_axis, presence, marker="^", markersize=3.5, color=GRID_RED,
            linewidth=1.4, label="presence BCE")
    ax.set_title("Validation loss components", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[0, 2])
    _style_axes(ax)
    ax.plot(epochs_axis, tie, marker="o", markersize=3.5, color=GRID_BLUE,
            linewidth=1.4)
    ax.set_title("Tie-breaker: H5 total-POC median MAE (normalized)", fontsize=10.5,
                 color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    _style_axes(ax)
    improvements = [
        (baseline - value) / baseline * 100.0 for value in primary
    ]
    bars = ax.bar(epochs_axis, improvements, color=[
        GRID_TEAL if value > 0 else GRID_RED for value in improvements
    ], width=0.65)
    ax.axhline(0.0, color=GRID_DARK, linewidth=0.8)
    ax.set_title("Relative improvement vs empirical baseline (%)", fontsize=11,
                 color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    _style_axes(ax)
    running_best = np.minimum.accumulate(best_metrics)
    ax.plot(epochs_axis, best_metrics, marker="o", markersize=3.5, color=GRID_GRAY,
            linewidth=1.2, label="epoch-end best")
    ax.plot(epochs_axis, running_best, color=GRID_TEAL, linewidth=1.8,
            label="running best")
    ax.scatter([best_epoch], [min(best_metrics)], s=90, facecolor="none",
               edgecolor=GRID_TEAL, linewidth=2.0, zorder=5)
    ax.set_title("Checkpoint selection trajectory", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 2])
    _style_axes(ax)
    labels = ("model", "baseline")
    values = (eval_primary, baseline)
    bars = ax.bar(labels, values, color=[GRID_BLUE, GRID_GRAY], width=0.55)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0005, f"{value:.5f}",
                ha="center", fontsize=9, color=GRID_DARK)
    delta = (baseline - eval_primary) / baseline * 100.0
    ax.set_title(f"Selected checkpoint: +{delta:.2f}% vs baseline", fontsize=11,
                 color=GRID_DARK)
    ax.set_ylabel("weighted normalized pinball", fontsize=9)
    ax.set_ylim(0.0, max(values) * 1.18)

    return _save(fig, output, "02_validation_curves.png")


def render_horizon_target_performance(study: Study, output: Path) -> Path:
    """Page 3: per-horizon / per-target grids from the frozen evaluation."""

    evaluation = study.evaluation
    pinball_grid = _numbers(evaluation["metrics"]["pinball"])
    physical_mae = _numbers(evaluation["physical_metrics"]["median_mae"])
    coverage = _numbers(evaluation["metrics"]["interval_coverage"])
    width = _numbers(evaluation["metrics"]["interval_width"])
    brier = _numbers(evaluation["metrics"]["presence_brier"])

    fig = plt.figure(figsize=(13.5, 9.8), facecolor="white")
    gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28, left=0.07, right=0.95,
                          top=0.9, bottom=0.06)
    fig.suptitle(
        "CHMT-Q Testing — Selected Checkpoint (epoch 10, step 2,010)  |  "
        "2,862 validation examples, per-horizon × per-target normalized metrics",
        fontsize=13,
        fontweight="bold",
        color=GRID_DARK,
    )

    panels = (
        (pinball_grid, "Normalized pinball (lower is better)", "RdYlGn_r"),
        (physical_mae, "Physical median MAE (ticks / units)", "RdYlGn_r"),
        (coverage, "q10–q90 interval coverage (target 0.80)", "RdYlGn"),
        (width, "q10–q90 interval width (normalized)", "RdYlGn_r"),
        (brier, "Presence Brier score (buy/sell)", "RdYlGn_r"),
    )
    positions = ((0, 0), (0, 1), (1, 0), (1, 1), (2, 0))
    for (grid, title, cmap), (row, col) in zip(panels, positions, strict=True):
        ax = fig.add_subplot(gs[row, col])
        _style_axes(ax)
        image = ax.imshow(grid, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(TARGET_NAMES)), TARGET_NAMES, fontsize=8,
                      rotation=18)
        ax.set_yticks(range(len(HORIZON_LABELS)), HORIZON_LABELS, fontsize=9)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=7.6, color=GRID_DARK)
        ax.set_title(title, fontsize=10.5, color=GRID_DARK)
        fig.colorbar(image, ax=ax, shrink=0.82)

    ax = fig.add_subplot(gs[2, 1])
    _style_axes(ax)
    h5_pinball = pinball_grid[4, :]
    ax.bar(range(5), h5_pinball, color=GRID_BLUE, width=0.6)
    ax.set_xticks(range(5), TARGET_NAMES, fontsize=8.5, rotation=18)
    ax.set_title("H5 (5-minute) normalized pinball by target", fontsize=10.5,
                 color=GRID_DARK)
    annotation = (
        "Development-forward H5 (the study's primary endpoint):\n"
        "model pinball 0.2544 vs 0.2613 empirical (+2.63% on this cell),\n"
        "but physical median MAE 532.841 vs 532.905 ticks — only 0.0119%\n"
        "better. Aggregate edge does not translate into useful location\n"
        "forecasts. Interval coverage ≈ 0.81–0.87 on POC targets."
    )
    ax.text(
        0.02,
        0.98,
        annotation,
        transform=ax.transAxes,
        fontsize=8,
        va="top",
        color=GRID_DARK,
        bbox={"facecolor": GRID_BG, "edgecolor": GRID_GRAY, "boxstyle": "round,pad=0.4"},
    )

    return _save(fig, output, "03_horizon_target_performance.png")


def render_partition_comparison(study: Study, output: Path) -> Path:
    """Page 4: model vs empirical baseline across the three study partitions."""

    partitions = (
        "in_sample_train_replay",
        "selected_validation",
        "untouched_development_forward",
    )
    decision = study.backtest["decision"]
    fig = plt.figure(figsize=(13.5, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.3, left=0.07, right=0.97,
                          top=0.89, bottom=0.07)
    fig.suptitle(
        "CHMT-Q Backtest — Model vs Empirical-Marginal Baseline  |  "
        f"primary endpoint: {decision['primary_partition']} "
        f"({PARTITION_LABELS[decision['primary_partition']]})",
        fontsize=13,
        fontweight="bold",
        color=GRID_DARK,
    )

    for index, partition in enumerate(partitions):
        model = study.partition_forecast(partition, "model", "normalized")
        empirical = study.partition_forecast(partition, "empirical_marginal", "normalized")
        model_primary = float(model["weighted_primary_metric"])
        empirical_primary = float(empirical["weighted_primary_metric"])
        improvement = (
            (empirical_primary - model_primary) / empirical_primary * 100.0
        )
        block = study.backtest["forecast"][partition]
        bootstrap = block["paired_daily_block_bootstrap_empirical_minus_model"]
        improvement_record = block["relative_model_improvement_percent"]

        ax = fig.add_subplot(gs[0, index])
        _style_axes(ax)
        labels = ("model", "empirical")
        values = (model_primary, empirical_primary)
        bars = ax.bar(labels, values, color=[GRID_BLUE, GRID_GRAY], width=0.55)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.5f}",
                    ha="center", fontsize=9, color=GRID_DARK)
        ax.set_title(
            f"{PARTITION_LABELS[partition]}\nweighted pinball {values[0]:.5f} vs "
            f"{values[1]:.5f}  ({improvement:+.2f}%)",
            fontsize=9.5,
            color=GRID_DARK,
        )
        ax.set_ylim(0.0, max(values) * 1.22)
        if index == 2:
            lower = float(bootstrap["lower"])
            upper = float(bootstrap["upper"])
            ax.text(
                0.5,
                -0.30,
                f"paired day-block bootstrap 95% CI of (empirical − model): "
                f"[{lower:.5f}, {upper:.5f}]\n"
                f"relative improvement {float(improvement_record):.4f}%  |  "
                f"{bootstrap['day_count']} days, {bootstrap['resamples']:,} resamples, "
                f"seed {bootstrap['seed']}",
                transform=ax.transAxes,
                ha="center",
                fontsize=7.6,
                color=GRID_DARK,
                bbox={"facecolor": GRID_BG, "edgecolor": GRID_GRAY,
                      "boxstyle": "round,pad=0.35"},
            )

    econ = study.partition_economics("untouched_development_forward")
    policies = econ["policies"]

    ax = fig.add_subplot(gs[1, 0])
    _style_axes(ax)
    names: list[str] = []
    net_returns: list[float] = []
    colors: list[str] = []
    for policy_key in (
        "median_long_flat",
        "median_long_short",
        "interval_long_flat",
        "interval_long_short",
        "base_cost_hurdle_long_flat",
    ):
        record = policies[policy_key]["base"]
        net_return = float(record["metrics"]["net_compounded_return"])
        names.append(POLICY_LABELS[policy_key])
        net_returns.append(net_return * 100.0)
        colors.append(GRID_BLUE if net_returns[-1] > 0 else GRID_RED)
    bars = ax.barh(names, net_returns, color=colors, height=0.55)
    ax.axvline(0.0, color=GRID_DARK, linewidth=0.8)
    ax.set_title("Policy net returns @ base 12 bps/side\n(development-forward)",
                 fontsize=10, color=GRID_DARK)
    ax.set_xlabel("net return (%)", fontsize=9)
    for bar, value in zip(bars, net_returns, strict=True):
        ax.text(
            value + (2.0 if value >= 0 else -2.0),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=GRID_DARK,
        )
    ax.set_xlim(min(net_returns) * 1.45, max(net_returns) * 2.2)

    ax = fig.add_subplot(gs[1, 1])
    _style_axes(ax)
    long_flat = policies["median_long_flat"]
    scenario_names = list(COST_SCENARIOS)
    scenario_returns = [
        float(long_flat[scenario]["metrics"]["net_compounded_return"]) * 100.0
        for scenario in scenario_names
    ]
    colors = [
        GRID_TEAL if value >= 0 else GRID_RED for value in scenario_returns
    ]
    bars = ax.bar(scenario_names, scenario_returns, color=colors, width=0.6)
    for bar, value in zip(bars, scenario_returns, strict=True):
        y = value + (1.4 if value >= 0 else -3.2)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.2f}%", ha="center",
                fontsize=8.5, color=GRID_DARK)
    ax.axhline(0.0, color=GRID_DARK, linewidth=0.8)
    ax.set_title("Median long/flat net return across cost scenarios",
                 fontsize=10, color=GRID_DARK)
    ax.set_ylabel("net return (%)", fontsize=9)
    ax.set_ylim(-105, 30)

    ax = fig.add_subplot(gs[1, 2])
    _style_axes(ax)
    break_even = float(long_flat["base"]["break_even_one_way_bps"])
    round_trip = 2.0 * break_even
    bars = ax.bar(
        ("break-even\n(round trip)", "optimistic", "base", "stress"),
        (round_trip, 18.0, 24.0, 40.0),
        color=[GRID_TEAL, GRID_AMBER, GRID_RED, "#7b4b94"],
        width=0.6,
    )
    for bar, value in zip(
        bars, (round_trip, 18.0, 24.0, 40.0), strict=True
    ):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.9, f"{value:.2f}",
                ha="center", fontsize=8.5, color=GRID_DARK)
    multiple = 24.0 / round_trip
    ax.set_title(
        f"Gross edge {round_trip:.4f} bps vs base round-trip 24 bps\n"
        f"→ base cost is {multiple:.1f}× break-even (no executable edge)",
        fontsize=10,
        color=GRID_DARK,
    )
    ax.set_ylabel("bps (one-way ×2)", fontsize=9)
    ax.set_ylim(0, 48)

    return _save(fig, output, "04_partition_comparison.png")


def render_forecast_quality(study: Study, output: Path) -> Path:
    """Page 5: per-horizon forecast diagnostics from the frozen backtest tape."""

    import gzip

    rows_by_partition: dict[str, list[dict[str, Any]]] = {}
    with study.predictions_path.open("rb") as handle:
        with gzip.open(handle, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                rows_by_partition.setdefault(record["evidence_class"], []).append(record)

    forward = rows_by_partition.get("untouched_development_forward", [])
    if not forward:
        forward = rows_by_partition[next(iter(rows_by_partition))]

    horizon_index = 4  # H5 total POC — the study's diagnostic focus
    predicted: list[float] = []
    actual: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for record in forward:
        if not record["actual_target_mask"][horizon_index][0]:
            continue
        predicted.append(float(record["predicted_quantiles_physical"][horizon_index][0][1]))
        actual.append(float(record["actual_targets_physical"][horizon_index][0]))
        lower.append(float(record["predicted_quantiles_physical"][horizon_index][0][0]))
        upper.append(float(record["predicted_quantiles_physical"][horizon_index][0][2]))
    predicted_arr = np.asarray(predicted)
    actual_arr = np.asarray(actual)
    lower_arr = np.asarray(lower)
    upper_arr = np.asarray(upper)

    correlation = float(np.corrcoef(predicted_arr, actual_arr)[0, 1])
    std_ratio = float(predicted_arr.std() / actual_arr.std()) if actual_arr.std() > 0 else math.nan

    fig = plt.figure(figsize=(13.5, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.26, left=0.07, right=0.97,
                          top=0.9, bottom=0.08)
    fig.suptitle(
        "CHMT-Q Forecast Quality — H5 Total-POC on Development-Forward  |  "
        f"{len(predicted_arr)} forecasts  |  Pearson r = {correlation:.4f}, "
        f"pred/realized σ ratio = {std_ratio:.4f}",
        fontsize=13,
        fontweight="bold",
        color=GRID_DARK,
    )

    ax = fig.add_subplot(gs[0, 0])
    _style_axes(ax)
    axis_min = min(predicted_arr.min(), actual_arr.min())
    axis_max = max(predicted_arr.max(), actual_arr.max())
    ax.plot((axis_min, axis_max), (axis_min, axis_max), color=GRID_GRAY,
            linestyle="--", linewidth=1.0, label="perfect")
    ax.scatter(actual_arr, predicted_arr, s=4, alpha=0.18, color=GRID_BLUE,
               edgecolors="none")
    ax.set_xlabel("actual H5 total-POC offset (ticks)", fontsize=9)
    ax.set_ylabel("predicted q50 (ticks)", fontsize=9)
    ax.set_title("Predicted median vs actual (H5)", fontsize=11, color=GRID_DARK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    ax = fig.add_subplot(gs[0, 1])
    _style_axes(ax)
    within = np.mean((actual_arr >= lower_arr) & (actual_arr <= upper_arr))
    ax.scatter(actual_arr, upper_arr - lower_arr, s=4, alpha=0.18, color=GRID_AMBER,
               edgecolors="none")
    ax.set_xlabel("actual H5 total-POC offset (ticks)", fontsize=9)
    ax.set_ylabel("q10–q90 width (ticks)", fontsize=9)
    ax.set_title(
        f"Interval width vs actual  |  empirical coverage {within:.3f} (target 0.80)",
        fontsize=10.5,
        color=GRID_DARK,
    )

    ax = fig.add_subplot(gs[1, 0])
    _style_axes(ax)
    time_axis = np.arange(len(predicted_arr))
    sample = min(700, len(predicted_arr))
    ax.fill_between(
        time_axis[:sample],
        lower_arr[:sample],
        upper_arr[:sample],
        color=GRID_TEAL,
        alpha=0.25,
        label="q10–q90",
    )
    ax.plot(time_axis[:sample], predicted_arr[:sample], color=GRID_TEAL, linewidth=0.9,
            label="predicted q50")
    ax.plot(time_axis[:sample], actual_arr[:sample], color=GRID_DARK, linewidth=0.7,
            alpha=0.75, label="actual")
    ax.set_title(f"Forecast vs actual timeline (first {sample} development-forward rows)",
                 fontsize=10.5, color=GRID_DARK)
    ax.set_xlabel("forecast index (chronological)", fontsize=9)
    ax.set_ylabel("ticks", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 1])
    _style_axes(ax)
    ax.hist(
        actual_arr,
        bins=60,
        color=GRID_GRAY,
        alpha=0.55,
        label="actual",
        density=True,
    )
    ax.hist(
        predicted_arr,
        bins=60,
        color=GRID_TEAL,
        alpha=0.55,
        label="predicted q50",
        density=True,
    )
    ax.set_title("Distribution: actual vs predicted (H5)", fontsize=11, color=GRID_DARK)
    ax.set_xlabel("H5 total-POC offset (ticks)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    return _save(fig, output, "05_forecast_quality.png")


def render_attention_gates(study: Study, output: Path) -> Path:
    """Page 6: attention-logit telemetry and quality gates across training."""

    modules: dict[str, list[float]] = {}
    for payload in study.epochs:
        for module, record in payload["telemetry"]["attention_logits"].items():
            raw = record["max_abs_by_head"]
            if isinstance(raw, str):
                values = [float(x) for x in raw.replace("[", " ").replace("]", " ").split(",") if x.strip()]
            else:
                values = [float(x) for x in raw]
            modules.setdefault(module, [0.0] * len(values))
            for head, value in enumerate(values):
                modules[module][head] = max(modules[module][head], value)

    fig = plt.figure(figsize=(13.5, 8.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3, left=0.09, right=0.96,
                          top=0.9, bottom=0.08)
    fig.suptitle(
        "CHMT-Q Model Internals — Attention Logit Telemetry & Modality Gating",
        fontsize=13,
        fontweight="bold",
        color=GRID_DARK,
    )

    ax = fig.add_subplot(gs[0, 0])
    _style_axes(ax)
    names = list(modules)
    per_head = np.asarray([modules[name] for name in names], dtype=np.float64)
    im = ax.imshow(per_head, cmap="YlGnBu", aspect="auto", vmin=0, vmax=10)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_xticks(range(per_head.shape[1]), [f"head {i + 1}" for i in range(per_head.shape[1])],
                  fontsize=8)
    for i in range(per_head.shape[0]):
        for j in range(per_head.shape[1]):
            ax.text(j, i, f"{per_head[i, j]:.1f}", ha="center", va="center", fontsize=6.5,
                    color=GRID_DARK)
    ax.set_title("Max |attention logit| per head (runaway threshold = 80)",
                 fontsize=10.5, color=GRID_DARK)
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = fig.add_subplot(gs[0, 1])
    _style_axes(ax)
    epochs_axis = np.arange(1, len(study.epochs) + 1)
    gates = study.gate_means()
    for i, (label, color) in enumerate(
        (("bars", GRID_BLUE), ("L1 trades", GRID_TEAL), ("L2 depth", GRID_AMBER))
    ):
        ax.plot(epochs_axis, gates[:, i], marker="o", markersize=3.5, color=color,
                linewidth=1.4, label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Modality availability-gate means per epoch", fontsize=10.5,
                 color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[1, 0])
    _style_axes(ax)
    coverage = study.validation_metrics("interval_coverage")
    for target in range(5):
        h5_by_epoch = coverage[:, 4, target]
        ax.plot(epochs_axis, h5_by_epoch, marker="o", markersize=3.2,
                color=(GRID_BLUE, GRID_TEAL, GRID_AMBER, GRID_RED, GRID_GRAY)[target],
                linewidth=1.2, label=TARGET_NAMES[target])
    ax.axhline(0.80, color=GRID_DARK, linestyle="--", linewidth=1.1, label="target 0.80")
    ax.set_title("H5 q10–q90 coverage during training (per target)", fontsize=10.5,
                 color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")

    ax = fig.add_subplot(gs[1, 1])
    _style_axes(ax)
    widths = study.validation_metrics("interval_width")
    for target in range(5):
        ax.plot(epochs_axis, widths[:, 4, target], marker="o", markersize=3.2,
                color=(GRID_BLUE, GRID_TEAL, GRID_AMBER, GRID_RED, GRID_GRAY)[target],
                linewidth=1.2, label=TARGET_NAMES[target])
    ax.set_title("H5 q10–q90 width during training (per target)", fontsize=10.5,
                 color=GRID_DARK)
    ax.set_xlabel("epoch", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)

    return _save(fig, output, "06_attention_gates.png")


def _write_index(output: Path, figures: list[tuple[str, str]]) -> Path:
    rows = "\n".join(
        f'<figure><img src="{name}" alt="{title}" loading="lazy">'
        f"<figcaption>{title}</figcaption></figure>"
        for name, title in figures
    )
    html = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>CHMT-Q Study Visualizer</title><style>"
        "body{font-family:Segoe UI,Arial,sans-serif;background:#f4f5f7;"
        "margin:0;padding:24px;color:#22303c}"
        "h1{font-size:20px}p{font-size:13px;color:#5a6470}"
        "figure{margin:0 0 36px}img{width:100%;max-width:1500px;"
        "border:1px solid #d9dee4;border-radius:8px;background:#fff}"
        "figcaption{font-size:13px;margin-top:8px;color:#5a6470}"
        "</style></head><body>"
        "<h1>CHMT-Q Training &amp; Testing Visualizer</h1>"
        "<p>Read-only dashboard rendered from the frozen June 2026 study artifacts "
        "(fold-01, seed 17, selected epoch 10 / step 2,010). "
        "Research-only, non-promotable.</p>"
        f"{rows}"
        "</body></html>"
    )
    target = output / "index.html"
    target.write_text(html, encoding="utf-8")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        default=Path(r"C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1"),
        help="Frozen study directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("visualizations"),
        help="Output directory for rendered figures",
    )
    args = parser.parse_args()
    study = Study(args.study)
    figures: list[tuple[str, str]] = []
    renderers = (
        ("01_training_overview.png", "Optimizer telemetry across all 18 epochs",
         render_training_overview),
        ("02_validation_curves.png", "Validation selection curves and baseline comparison",
         render_validation_curves),
        ("03_horizon_target_performance.png",
         "Per-horizon × per-target testing performance grids", render_horizon_target_performance),
        ("04_partition_comparison.png", "Backtest partitions, policies, and cost scenarios",
         render_partition_comparison),
        ("05_forecast_quality.png", "H5 forecast quality on development-forward data",
         render_forecast_quality),
        ("06_attention_gates.png", "Attention telemetry, gating, and coverage drift",
         render_attention_gates),
    )
    for name, title, renderer in renderers:
        path = renderer(study, args.output)  # type: ignore[call-arg]
        figures.append((name, title))
        print(f"rendered {path}")
    index = _write_index(args.output, figures)
    print(f"index {index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
