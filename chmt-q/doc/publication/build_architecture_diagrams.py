"""Build the three minimalist diagrams used by the CHMT-Q architecture paper."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

INK = "#17212B"
MUTED = "#526071"
LINE = "#CBD5E1"
ACCENT = "#0F766E"
ACCENT_SOFT = "#ECFDF5"
SURFACE = "#F8FAFC"
WHITE = "#FFFFFF"


def setup(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height), dpi=220)
    fig.patch.set_facecolor(WHITE)
    ax.set_facecolor(WHITE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


def node(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle="",
    *,
    accent=False,
    muted=False,
    title_size=10.5,
    subtitle_size=9.0,
):
    edge = ACCENT if accent else LINE
    face = ACCENT_SOFT if accent else (SURFACE if muted else WHITE)
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.15,
        edgecolor=edge,
        facecolor=face,
        zorder=2,
    )
    ax.add_patch(patch)

    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.62,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight=600,
            color=INK if not muted else MUTED,
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.31,
            subtitle,
            ha="center",
            va="center",
            fontsize=subtitle_size,
            color=MUTED,
            zorder=3,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight=600,
            color=INK if not muted else MUTED,
            zorder=3,
        )
    return patch


def arrow(ax, start, end, *, dashed=False, color=ACCENT, width=1.25):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        zorder=2.5,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "linewidth": width,
            "linestyle": "--" if dashed else "-",
            "mutation_scale": 10,
            "shrinkA": 1,
            "shrinkB": 4,
        },
    )


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        ASSETS / name,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor=WHITE,
    )
    plt.close(fig)


def build_shape_trace():
    fig, ax = setup(7.8, 2.25)

    lanes = [
        (0.72, "Bars", r"$[B,64,16]\ \rightarrow\ [B,64,128]$"),
        (0.51, "L1", r"$[B,240,16]\ \rightarrow\ [B,60,128]$"),
        (0.30, "L2", r"$[B,240,2,10,12]\ \rightarrow\ [B,60,128]$"),
    ]
    for y, title, subtitle in lanes:
        node(ax, 0.01, y, 0.25, 0.14, title, subtitle, subtitle_size=8.6)

    node(
        ax,
        0.01,
        0.07,
        0.25,
        0.12,
        "Quality",
        r"$[B,16]$",
        muted=True,
        title_size=9.8,
        subtitle_size=8.2,
    )

    node(
        ax,
        0.33,
        0.56,
        0.18,
        0.22,
        "Memory",
        r"$[B,184,128]$",
        accent=True,
        title_size=11.0,
        subtitle_size=9.0,
    )
    node(
        ax,
        0.33,
        0.12,
        0.18,
        0.20,
        "Context + gates",
        r"$z_{\rm ctx},\ \alpha_m$",
    )
    node(ax, 0.57, 0.22, 0.13, 0.19, "Queries", r"$[B,5,128]$")
    node(ax, 0.76, 0.44, 0.12, 0.22, "Decoder", "2 blocks")
    node(
        ax,
        0.92,
        0.44,
        0.06,
        0.22,
        "Raw",
        r"$[B,5,17]$",
        accent=True,
        title_size=9.3,
        subtitle_size=8.6,
    )

    junction = (0.285, 0.67)
    for y, _, _ in lanes:
        ax.plot(
            [0.26, junction[0]],
            [y + 0.07, junction[1]],
            color=ACCENT,
            linewidth=1.25,
            zorder=1,
        )
    ax.scatter(*junction, s=9, color=ACCENT, zorder=2.4)
    arrow(ax, junction, (0.33, 0.67))
    arrow(ax, (0.26, 0.13), (0.33, 0.22), color=MUTED, width=1.0)
    arrow(ax, (0.51, 0.22), (0.57, 0.315))
    arrow(ax, (0.51, 0.67), (0.76, 0.58))
    arrow(ax, (0.70, 0.315), (0.76, 0.48))

    # Reliability contributes only a decoder key bias; it never rewrites memory.
    ax.plot(
        [0.51, 0.72],
        [0.15, 0.15],
        color=MUTED,
        linewidth=1.0,
        linestyle="--",
        zorder=1,
    )
    arrow(
        ax,
        (0.72, 0.15),
        (0.76, 0.44),
        dashed=True,
        color=MUTED,
        width=1.0,
    )
    arrow(ax, (0.88, 0.55), (0.92, 0.55))

    ax.text(
        0.33,
        0.86,
        "64 + 60 + 60 tokens",
        ha="left",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    ax.text(
        0.595,
        0.11,
        "cross-attn bias",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    ax.text(
        0.33,
        0.035,
        "Hard masks remain authoritative.",
        ha="left",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    save(fig, "architecture-shape-trace.png")


def build_l2_hierarchy():
    fig, ax = setup(7.8, 1.66)

    ax.text(
        0.025,
        0.91,
        "WITHIN ONE SNAPSHOT",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight=600,
        color=MUTED,
    )
    ax.plot([0.025, 0.55], [0.855, 0.855], color=LINE, linewidth=0.9)
    ax.text(
        0.595,
        0.91,
        "ACROSS TIME",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight=600,
        color=MUTED,
    )
    ax.plot([0.595, 0.975], [0.855, 0.855], color=LINE, linewidth=0.9)

    boxes = [
        (0.025, 0.34, 0.115, "Snapshot", r"$2\times10\times12$"),
        (0.17, 0.34, 0.115, "Level tokens", r"$20\times32$"),
        (0.315, 0.34, 0.125, "Level attention", "same timestamp"),
        (0.47, 0.34, 0.085, "Book pool", ""),
        (0.595, 0.34, 0.105, "Frames", r"$240\times128$"),
        (0.73, 0.34, 0.11, "4-frame pool", ""),
        (0.87, 0.34, 0.105, "Temporal ×3", r"$60\times128$"),
    ]
    for x, y, w, title, subtitle in boxes:
        node(
            ax,
            x,
            y,
            w,
            0.28,
            title,
            subtitle,
            accent=title.startswith("Temporal"),
            title_size=9.2,
            subtitle_size=8.2,
        )

    for left, right in zip(boxes[:-1], boxes[1:]):
        arrow(
            ax,
            (left[0] + left[2], left[1] + 0.14),
            (right[0], right[1] + 0.14),
            width=1.15,
        )

    ax.text(
        0.315,
        0.17,
        "masked, no causal triangle",
        ha="left",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )
    ax.text(
        0.9225,
        0.17,
        "causal",
        ha="center",
        va="center",
        fontsize=9.0,
        color=ACCENT,
        fontweight=600,
    )
    save(fig, "architecture-l2-hierarchy.png")


def build_quality_gates():
    fig, ax = setup(7.8, 1.88)

    node(ax, 0.025, 0.52, 0.17, 0.23, "Summaries", r"$s_b,\ s_1,\ s_2$")
    node(
        ax,
        0.025,
        0.14,
        0.17,
        0.23,
        "Quality",
        r"$q\in\mathbb{R}^{16}\ \rightarrow\ c_q$",
        muted=True,
    )
    node(
        ax,
        0.255,
        0.14,
        0.16,
        0.23,
        "Availability",
        r"$a_m\in\{0,1\}$",
        muted=True,
    )
    node(
        ax,
        0.255,
        0.52,
        0.16,
        0.23,
        "Gate MLPs",
        r"$\tilde g_m=a_m\,\sigma(s_m,c_q)$",
    )
    node(
        ax,
        0.49,
        0.52,
        0.17,
        0.23,
        "Normalize",
        r"$\alpha_m\propto\tilde g_m$",
        accent=True,
    )
    node(
        ax,
        0.735,
        0.58,
        0.24,
        0.25,
        "Global context",
        r"${\rm RMSNorm}(c_q+\sum_m\alpha_m s_m)$",
        title_size=9.6,
        subtitle_size=8.8,
    )
    node(
        ax,
        0.735,
        0.22,
        0.24,
        0.25,
        "Decoder cross-attn bias",
        r"$\log(\alpha_m+\epsilon)$",
        title_size=8.8,
        subtitle_size=8.8,
    )

    arrow(ax, (0.195, 0.635), (0.255, 0.635))
    arrow(ax, (0.195, 0.26), (0.255, 0.56), dashed=True, color=MUTED, width=1.0)
    arrow(ax, (0.335, 0.35), (0.335, 0.52), dashed=True, color=MUTED, width=1.0)
    arrow(ax, (0.415, 0.635), (0.49, 0.635))
    arrow(ax, (0.66, 0.635), (0.735, 0.705))
    arrow(ax, (0.66, 0.635), (0.735, 0.345))

    ax.text(
        0.49,
        0.22,
        r"$a_m=0\ \Rightarrow\ \alpha_m=0$",
        ha="left",
        va="center",
        fontsize=9.2,
        color=MUTED,
    )
    save(fig, "architecture-quality-gates.png")


def main():
    build_shape_trace()
    build_l2_hierarchy()
    build_quality_gates()
    for path in sorted(ASSETS.glob("architecture-*.png")):
        print(f"{path.name}: {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
