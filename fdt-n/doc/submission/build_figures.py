"""Generate deterministic vector figures for the FDT-N submission manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

BLACK = "#202020"
GRAY = "#6f6f6f"
LIGHT = "#ececec"
BLUE = "#3465a4"
RED = "#a62f2f"
GREEN = "#397a4a"


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, format="pdf")
    plt.close(fig)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 3.35))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(
        x: float, y: float, w: float, h: float, text: str,
        fill: str = "white", fontsize: float = 7.4,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=0.9, edgecolor=BLACK, facecolor=fill
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
                                     linewidth=0.9, color=BLACK))

    ax.set_xlim(0, 15)
    ax.text(1.15, 5.28, "INPUTS", ha="center", fontsize=9, fontweight="bold")
    ax.text(7.55, 5.28, "TRANSFORMER PROCESSING", ha="center", fontsize=9, fontweight="bold")
    ax.text(14.05, 5.28, "OUTPUT", ha="center", fontsize=9, fontweight="bold")
    ax.plot([0.08, 2.25], [5.08, 5.08], color=GRAY, lw=0.8)
    ax.plot([2.48, 12.88], [5.08, 5.08], color=GRAY, lw=0.8)
    ax.plot([13.08, 14.92], [5.08, 5.08], color=GRAY, lw=0.8)

    box(0.15, 3.45, 2.00, 0.94, "row_features\n$[B,64,N,11]$", LIGHT)
    box(0.15, 1.45, 2.00, 0.94, "bar_features\n$[B,64,23]$", LIGHT)
    box(2.48, 3.45, 1.34, 0.94, "row encoder\n$11\\to64$", fontsize=6.4)
    box(4.10, 3.45, 1.68, 0.94, "4-query\nrow pooling", fontsize=6.4)
    box(2.48, 1.45, 1.34, 0.94, "bar encoder\n$23\\to64$", fontsize=6.4)
    box(4.70, 2.38, 1.48, 0.94, "gated\nfusion", fontsize=6.5)
    box(6.48, 2.38, 1.54, 0.94, "3 causal\nblocks", fontsize=6.5)
    box(8.34, 2.38, 1.46, 0.94, "2 decoder\nblocks", fontsize=6.5)
    box(10.12, 2.38, 1.22, 0.94, "$64\\to21$\nreshape", fontsize=6.4)
    box(11.65, 2.38, 1.28, 0.94, "order +\nconstraints", fontsize=6.3)
    box(13.22, 2.38, 1.62, 0.94, "quantiles\n$[B,8,7,3]$", LIGHT)

    arrow(2.15, 3.92, 2.48, 3.92)
    arrow(3.82, 3.92, 4.10, 3.92)
    arrow(5.78, 3.80, 5.62, 3.32)
    arrow(2.15, 1.92, 2.48, 1.92)
    arrow(3.82, 1.92, 5.15, 2.38)
    arrow(6.18, 2.85, 6.48, 2.85)
    arrow(8.02, 2.85, 8.34, 2.85)
    arrow(9.80, 2.85, 10.12, 2.85)
    arrow(11.34, 2.85, 11.65, 2.85)
    arrow(12.93, 2.85, 13.22, 2.85)

    ax.text(4.94, 4.58, "row_mask $[B,64,N]$ masks padding", ha="center",
            fontsize=6.6, color=GRAY)
    ax.text(7.25, 1.72, "causal mask + bar_mask $[B,64]$", ha="center",
            fontsize=6.6, color=GRAY)
    ax.text(9.07, 1.30, "8 learned horizon queries", ha="center",
            fontsize=6.6, color=GRAY)
    ax.text(1.15, 0.52, "float32 values; bool masks\noldest $\\rightarrow$ newest",
            ha="center", fontsize=6.6, color=GRAY)
    ax.text(7.70, 0.52, "371,861 parameters; preprocessing outside ONNX",
            ha="center", fontsize=6.8, color=GRAY)
    ax.text(14.03, 0.52, "horizon $\\times$ target $\\times$\n(q10, q50, q90)",
            ha="center", fontsize=6.6, color=GRAY)
    save(fig, "architecture.pdf")


def split_timeline() -> None:
    fig, ax = plt.subplots(figsize=(7.15, 1.5))
    counts = np.array([13990, 144, 2869, 144, 2942], dtype=float)
    labels = ["train\n13,990", "purge\n72+72", "validation\n2,869", "purge\n72+72", "test\n2,942"]
    colors = ["#d9d9d9", "#ffffff", "#bdbdbd", "#ffffff", "#8c8c8c"]
    left = 0.0
    for value, label, color in zip(counts, labels, colors):
        width = value / counts.sum()
        ax.barh([0], [width], left=[left], height=0.42, color=color, edgecolor=BLACK, linewidth=0.8)
        if value == 144:
            ax.annotate(label, xy=(left + width / 2, 0.2), xytext=(left + width / 2, 0.62),
                        ha="center", va="bottom", fontsize=7,
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=BLACK))
        else:
            ax.text(left + width / 2, 0, label, ha="center", va="center", fontsize=8,
                    color="white" if color == "#8c8c8c" else BLACK)
        left += width
    ax.text(0, -0.42, "2024-01-01 00:00 UTC", ha="left", fontsize=8)
    ax.text(1, -0.42, "2024-01-14 23:59 UTC", ha="right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.55, 0.85)
    ax.axis("off")
    save(fig, "split-timeline.pdf")


def target_mae() -> None:
    labels = ["POC", "Low", "High", "Buy POC", "Sell POC", "Log volume", "Delta"]
    model = np.array([0.2414, 0.2552, 0.2272, 0.2431, 0.2415, 0.1219, 0.3207])
    persistence = np.array([0.2432, 0.2423, 0.2221, 0.2442, 0.2426, 0.1536, 0.4321])
    ridge = np.array([0.2357, 0.2371, 0.2225, 0.2374, 0.2359, 0.1198, 0.3197])
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    ax.bar(x - width, model, width, label="FDT-N", color=BLUE)
    ax.bar(x, persistence, width, label="persistence", color="#bdbdbd", edgecolor=BLACK, linewidth=0.4)
    ax.bar(x + width, ridge, width, label="direct ridge", color=RED)
    ax.set_ylabel("normalized q50-forecast MAE")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylim(0, 0.48)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    save(fig, "target-mae.pdf")


def horizon_mae() -> None:
    h = np.arange(1, 9)
    model = np.array([0.1388, 0.1843, 0.2146, 0.2383, 0.2559, 0.2732, 0.2846, 0.2973])
    persistence = np.array([0.1577, 0.2041, 0.2344, 0.2560, 0.2731, 0.2906, 0.3038, 0.3147])
    ridge = np.array([0.1315, 0.1782, 0.2084, 0.2310, 0.2496, 0.2663, 0.2802, 0.2926])
    fig, ax = plt.subplots(figsize=(7.15, 3.0))
    ax.plot(h, model, marker="o", color=BLUE, label="FDT-N")
    ax.plot(h, persistence, marker="s", color=GRAY, label="persistence")
    ax.plot(h, ridge, marker="^", color=RED, label="direct ridge")
    ax.set_xlabel("forecast horizon (minutes)")
    ax.set_ylabel("mean normalized q50-forecast MAE")
    ax.set_xticks(h)
    ax.set_ylim(0.11, 0.33)
    ax.grid(color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "horizon-mae.pdf")


def coverage() -> None:
    labels = ["POC", "Low", "High", "Buy POC", "Sell POC", "Log volume", "Delta"]
    values = np.array([78.59, 77.00, 78.87, 78.31, 80.15, 80.70, 79.23])
    fig, ax = plt.subplots(figsize=(7.15, 2.8))
    x = np.arange(len(labels))
    ax.vlines(x, np.minimum(values, 80.0), np.maximum(values, 80.0), color="#9eb4cf", linewidth=2.0)
    ax.scatter(x, values, color=BLUE, s=42, zorder=3, label="empirical coverage")
    ax.axhline(80.0, color=RED, linestyle="--", linewidth=1.0, label="nominal 80%")
    ax.set_ylabel("empirical central-80% coverage (%)")
    ax.set_xticks(x, labels, rotation=22, ha="right")
    ax.set_ylim(72, 84)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    for xi, value in zip(x, values):
        ax.text(xi, value + 0.25, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7)
    save(fig, "coverage.pdf")


def resampling() -> None:
    seeds = ["17", "29", "43"]
    persistence = np.array([[21, 21, 14], [16, 19, 21], [18, 21, 17]])
    ridge = np.array([[0, 20, 36], [0, 25, 31], [0, 25, 31]])
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.8), sharey=True)
    colors = [GREEN, "#bdbdbd", RED]
    names = ["range below zero", "range spans zero", "range above zero"]
    for ax, data, title in zip(axes, [persistence, ridge], ["persistence", "direct ridge"]):
        bottom = np.zeros(3)
        for column, color, name in zip(data.T, colors, names):
            ax.bar(seeds, column, bottom=bottom, color=color, label=name,
                   edgecolor="white", linewidth=0.5)
            bottom += column
        ax.set_title(title)
        ax.set_xlabel("seed")
        ax.set_ylim(0, 56)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("number of horizon-target cells")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(top=0.76, wspace=0.25)
    save(fig, "resampling.pdf")


if __name__ == "__main__":
    architecture()
    split_timeline()
    target_mae()
    horizon_mae()
    coverage()
    resampling()
