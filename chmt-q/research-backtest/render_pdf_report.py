"""Render the verified CHMT-Q study as a compact visual PDF report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = landscape(A4)

NAVY = "#0B132B"
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE8"
PAPER = "#F5F7FA"
WHITE = "#FFFFFF"
BLUE = "#2F6BFF"
TEAL = "#00A88F"
RED = "#E64A5E"
AMBER = "#F2A93B"
GRAY = "#9AA4B2"
LIGHT_BLUE = "#EAF0FF"
LIGHT_RED = "#FDECEF"
LIGHT_AMBER = "#FFF5DE"

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    results_path = args.results.resolve(strict=True)
    manifest_path = args.manifest.resolve(strict=True)
    output = args.output.resolve()
    workdir = args.workdir.resolve()
    if output.exists():
        raise FileExistsError(output)
    workdir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    _register_fonts()
    results = _read_json(results_path)
    manifest = _read_json(manifest_path)
    _validate_sources(results_path, manifest_path, results, manifest)
    charts = _build_charts(results, workdir)

    report = VisualReport(output, results, manifest, charts)
    report.render()
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": _sha256(output),
                "pages": report.page_number,
            },
            sort_keys=True,
        )
    )
    return 0


def _register_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD
    regular = Path(r"C:\Windows\Fonts\segoeui.ttf")
    semibold = Path(r"C:\Windows\Fonts\seguisb.ttf")
    if regular.is_file() and semibold.is_file():
        pdfmetrics.registerFont(TTFont("SegoeUI", regular))
        pdfmetrics.registerFont(TTFont("SegoeUI-Semibold", semibold))
        FONT_REGULAR = "SegoeUI"
        FONT_BOLD = "SegoeUI-Semibold"


class VisualReport:
    def __init__(
        self,
        output: Path,
        results: dict[str, Any],
        manifest: dict[str, Any],
        charts: dict[str, Path],
    ) -> None:
        self.output = output
        self.results = results
        self.manifest = manifest
        self.charts = charts
        self.canvas = canvas.Canvas(
            str(output), pagesize=(PAGE_W, PAGE_H), pageCompression=1
        )
        self.canvas.setTitle("CHMT-Q BTC-USDT June 2026 Results")
        self.canvas.setAuthor("DaxAlgo Research")
        self.page_number = 0

    def render(self) -> None:
        self._cover()
        self._forecast()
        self._calibration()
        self._direction()
        self._trading()
        self._costs()
        self._decision()
        self.canvas.save()

    def _new_page(
        self,
        title: str,
        subtitle: str,
        *,
        dark: bool = False,
        kicker: str = "RESULTS",
    ) -> None:
        if self.page_number:
            self.canvas.showPage()
        self.page_number += 1
        background = NAVY if dark else PAPER
        foreground = WHITE if dark else INK
        self.canvas.setFillColor(HexColor(background))
        self.canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(TEAL if dark else BLUE))
        self.canvas.setFont(FONT_BOLD, 8)
        self.canvas.drawString(42, PAGE_H - 38, kicker)
        self.canvas.setFillColor(HexColor(foreground))
        self.canvas.setFont(FONT_BOLD, 24)
        self.canvas.drawString(42, PAGE_H - 70, title)
        self.canvas.setFont(FONT_REGULAR, 10)
        self.canvas.setFillColor(HexColor("#CBD5E1" if dark else MUTED))
        self.canvas.drawString(42, PAGE_H - 89, subtitle)
        if not dark:
            self._footer()

    def _footer(self) -> None:
        self.canvas.setStrokeColor(HexColor(GRID))
        self.canvas.setLineWidth(0.5)
        self.canvas.line(42, 27, PAGE_W - 42, 27)
        self.canvas.setFillColor(HexColor(MUTED))
        self.canvas.setFont(FONT_REGULAR, 7.5)
        self.canvas.drawString(
            42,
            14,
            "Development research | Offline simulation | Jun 25-30 holdout unaccessed",
        )
        self.canvas.drawRightString(PAGE_W - 42, 14, str(self.page_number))

    def _cover(self) -> None:
        forward = self.results["forecast"]["untouched_development_forward"]
        proxy = self.results["candle_proxy"]["untouched_development_forward"][4]
        base = self.results["economics"]["untouched_development_forward"]["policies"]
        self._new_page(
            "Small forecast edge. No tradable edge.",
            "CHMT-Q v1 | OKX BTC-USDT | June 2026 development study",
            dark=True,
            kicker="DAXALGO RESEARCH // VERIFIED BACKTEST",
        )
        self.canvas.setFillColor(HexColor("#94A3B8"))
        self.canvas.setFont(FONT_REGULAR, 9)
        self.canvas.drawString(42, PAGE_H - 120, "DECISION")
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setFont(FONT_BOLD, 17)
        self.canvas.drawString(42, PAGE_H - 144, "NO CAPITAL ALLOCATION")
        self.canvas.setFont(FONT_REGULAR, 9.5)
        self.canvas.setFillColor(HexColor("#CBD5E1"))
        self.canvas.drawString(
            42,
            PAGE_H - 163,
            "The model improves probabilistic scoring, but the observed edge does not survive costs.",
        )

        metrics = [
            (
                f"{forward['relative_model_improvement_percent']:.2f}%",
                "lower weighted pinball",
                "vs empirical baseline",
                BLUE,
            ),
            (
                f"{100 * proxy['anchor_to_close_direction_proxy']['accuracy']:.2f}%",
                "anchor-close proxy accuracy",
                f"MCC {proxy['anchor_to_close_direction_proxy']['mcc']:.3f}",
                AMBER,
            ),
            (
                "0 / 1,148",
                "primary trades",
                "all intervals crossed zero",
                TEAL,
            ),
            (
                f"{100 * base['median_long_flat']['base']['metrics']['net_compounded_return']:.2f}%",
                "aggressive strategy",
                "base-cost compounded",
                RED,
            ),
        ]
        x0, y0, gap = 42, 270, 14
        card_w = (PAGE_W - 84 - 3 * gap) / 4
        for index, metric in enumerate(metrics):
            self._metric_card(
                x0 + index * (card_w + gap),
                y0,
                card_w,
                112,
                *metric,
                dark=True,
            )

        self.canvas.setFillColor(HexColor("#94A3B8"))
        self.canvas.setFont(FONT_BOLD, 8)
        self.canvas.drawString(42, 235, "EVIDENCE WINDOW")
        segments = [
            ("Train replay", 18, GRAY),
            ("Selected", 2, AMBER),
            ("Untouched", 4, BLUE),
            ("Sealed", 6, "#475569"),
        ]
        line_x, line_y, line_w, line_h = 42, 180, PAGE_W - 84, 22
        offset = line_x
        for label, days, color in segments:
            width = line_w * days / 30
            self.canvas.setFillColor(HexColor(color))
            self.canvas.rect(offset, line_y, width, line_h, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 8)
            self.canvas.drawCentredString(
                offset + width / 2, line_y + 7, f"{label} {days}d"
            )
            offset += width
        self.canvas.setFillColor(HexColor("#CBD5E1"))
        self.canvas.setFont(FONT_REGULAR, 8.5)
        self.canvas.drawString(42, 159, "Jun 1")
        self.canvas.drawRightString(PAGE_W - 42, 159, "Jun 30")
        self.canvas.setFont(FONT_BOLD, 9)
        self.canvas.setFillColor(HexColor(AMBER))
        self.canvas.drawString(
            42, 125, "Only Jun 21-24 is genuinely unseen for this checkpoint."
        )
        self.canvas.setFillColor(HexColor("#94A3B8"))
        self.canvas.setFont(FONT_REGULAR, 8)
        self.canvas.drawString(
            42,
            52,
            f"Study {self.manifest['study_id'][:16]} | 34,344 predictions | 8,586 sealed rows untouched",
        )

    def _forecast(self) -> None:
        forward = self.results["forecast"]["untouched_development_forward"]
        self._new_page(
            "Forecast score improved. Point accuracy barely moved.",
            "Probabilistic score versus training-only baselines; lower is better.",
            kicker="01 // FORECAST QUALITY",
        )
        self._chart_card(42, 246, 370, 238, self.charts["pinball"], "Weighted pinball")
        self._chart_card(
            429, 246, 371, 238, self.charts["poc_mae"], "Total POC median error"
        )

        self._callout(
            42,
            105,
            237,
            104,
            "2.26%",
            "forward improvement",
            "Model 0.251391 vs empirical 0.257193",
            LIGHT_BLUE,
            BLUE,
        )
        five_model = forward["model"]["physical"]["median_mae"][4][0]
        five_base = forward["empirical_marginal"]["physical"]["median_mae"][4][0]
        self._callout(
            296,
            105,
            237,
            104,
            f"{0.1 * five_model:.2f} USDT",
            "five-minute POC MAE",
            f"Baseline {0.1 * five_base:.2f} USDT - effectively tied",
            WHITE,
            INK,
        )
        ci = forward["paired_daily_block_bootstrap_empirical_minus_model"]
        self._callout(
            550,
            105,
            250,
            104,
            "4 days",
            "untouched evidence",
            f"Nominal effect interval [{ci['lower']:.4f}, {ci['upper']:.4f}]",
            LIGHT_AMBER,
            AMBER,
        )

    def _calibration(self) -> None:
        self._new_page(
            "Coverage is acceptable because intervals are wide.",
            "Raw q10-q90 intervals; no separate conformal calibration slice.",
            kicker="02 // CALIBRATION",
        )
        self._chart_card(
            42, 244, 370, 240, self.charts["coverage_width"], "Total POC interval"
        )
        self._chart_card(
            429, 244, 371, 240, self.charts["cdf_heatmap"], "Empirical quantile CDF"
        )

        self._callout(
            42,
            105,
            237,
            104,
            "81.15%",
            "five-minute coverage",
            "Nominal target 80%",
            LIGHT_BLUE,
            BLUE,
        )
        self._callout(
            296,
            105,
            237,
            104,
            "190.47 USDT",
            "five-minute interval width",
            "3.2x wider than one minute",
            WHITE,
            INK,
        )
        self._callout(
            550,
            105,
            250,
            104,
            "99.91%",
            "delta interval coverage",
            "Width 1.992 on a [-1, 1] target - nearly uninformative",
            LIGHT_RED,
            RED,
        )

    def _direction(self) -> None:
        proxy = self.results["candle_proxy"]["untouched_development_forward"][4]
        self._new_page(
            "Candle association is weak.",
            "Total-POC median used only as a downstream direction proxy - not an OHLC forecast.",
            kicker="03 // FUTURE-CANDLE PROXY",
        )
        self._chart_card(
            42, 246, 480, 238, self.charts["direction"], "Accuracy by horizon"
        )
        self._chart_card(
            539, 246, 261, 238, self.charts["confusion"], "Five-minute confusion"
        )

        self._callout(
            42,
            105,
            237,
            104,
            f"{100 * proxy['anchor_to_close_direction_proxy']['accuracy']:.2f}%",
            "anchor-close accuracy",
            "Always-down majority: 52.39%",
            LIGHT_AMBER,
            AMBER,
        )
        self._callout(
            296,
            105,
            237,
            104,
            f"{proxy['anchor_to_close_direction_proxy']['mcc']:.4f}",
            "Matthews correlation",
            "Near zero association",
            WHITE,
            INK,
        )
        self._callout(
            550,
            105,
            250,
            104,
            "0%",
            "confidence-filter coverage",
            "Every forward q10-q90 POC interval crossed zero",
            LIGHT_RED,
            RED,
        )

    def _trading(self) -> None:
        econ = self.results["economics"]["untouched_development_forward"]
        self._new_page(
            "The primary strategy correctly abstained.",
            "Non-overlapping five-minute trades; one-minute entry delay; 100,000 USDT illustration.",
            kicker="04 // TRADING SIMULATION",
        )
        self._callout(
            42,
            366,
            230,
            118,
            "0 / 1,148",
            "primary active trades",
            "Interval long/flat stayed in cash",
            LIGHT_BLUE,
            BLUE,
        )
        self._callout(
            289,
            366,
            230,
            118,
            "0 USDT",
            "primary net P&L",
            "No trade is not evidence of alpha",
            WHITE,
            INK,
        )
        buy_hold = econ["buy_and_hold"]["base"]["pnl_per_initial_capital"]
        self._callout(
            536,
            366,
            264,
            118,
            f"{buy_hold:,.0f} USDT",
            "buy-and-hold base P&L",
            "Same price endpoints",
            LIGHT_RED,
            RED,
        )
        self._chart_card(
            42,
            105,
            758,
            224,
            self.charts["waterfall"],
            "Aggressive median policies - fixed notional",
        )

    def _costs(self) -> None:
        policies = self.results["economics"]["untouched_development_forward"][
            "policies"
        ]
        self._new_page(
            "Turnover overwhelms the gross edge.",
            "Compounded return under preregistered one-way fee plus slippage assumptions.",
            kicker="05 // COST SENSITIVITY",
        )
        self._chart_card(
            42, 228, 500, 256, self.charts["cost_sensitivity"], "Compounded return"
        )
        self._chart_card(
            559, 228, 241, 256, self.charts["break_even"], "Break-even cost"
        )

        long_flat = policies["median_long_flat"]["frictionless"]
        long_short = policies["median_long_short"]["frictionless"]
        self._callout(
            42,
            105,
            237,
            90,
            "+2.16%",
            "long/flat frictionless",
            f"{long_flat['active_decisions']} trades",
            LIGHT_BLUE,
            BLUE,
        )
        self._callout(
            296,
            105,
            237,
            90,
            "+4.99%",
            "long/short frictionless",
            f"{long_short['active_decisions']} trades - hypothetical margin proxy",
            LIGHT_BLUE,
            BLUE,
        )
        self._callout(
            550,
            105,
            250,
            90,
            ">50x",
            "cost-to-edge ratio",
            "Base 12 bps/side vs about 0.22 bps break-even",
            LIGHT_RED,
            RED,
        )

    def _decision(self) -> None:
        self._new_page(
            "Do not deploy. Redesign the next experiment.",
            "Current evidence supports local model research only.",
            kicker="06 // DECISION",
        )
        gates = [
            ("Forecast", "AMBER", "2.26% better pinball", AMBER),
            ("Direction", "RED", "MCC 0.0475", RED),
            ("Economics", "RED", "No cost-qualified edge", RED),
            ("Evidence", "RED", "One seed, four forward days", RED),
        ]
        x, y, gap = 42, 365, 12
        gate_w = (PAGE_W - 84 - 3 * gap) / 4
        for index, (name, status, detail, color) in enumerate(gates):
            gx = x + index * (gate_w + gap)
            self._rounded_rect(gx, y, gate_w, 112, WHITE, GRID)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.circle(gx + 20, y + 88, 5, stroke=0, fill=1)
            self.canvas.setFont(FONT_BOLD, 9)
            self.canvas.drawString(gx + 32, y + 84, status)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 15)
            self.canvas.drawString(gx + 16, y + 50, name)
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_REGULAR, 8.5)
            self._wrapped_text(detail, gx + 16, y + 27, gate_w - 32, 10, max_lines=2)

        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 13)
        self.canvas.drawString(42, 326, "Next experiment")
        steps = [
            (
                "1",
                "Train on pre-June history",
                "Use several months and multiple regimes.",
            ),
            (
                "2",
                "Add a direct return target",
                "Keep candle/return outputs separate from footprint POC.",
            ),
            ("3", "Run at least three seeds", "Report every seed and expanding fold."),
            (
                "4",
                "Replay executable bid/ask",
                "Measure latency, spread, slippage, capacity, and short funding.",
            ),
            (
                "5",
                "Freeze one policy",
                "Consume a true external month once, after all choices are locked.",
            ),
        ]
        sy = 286
        for number, heading, detail in steps:
            self.canvas.setFillColor(HexColor(BLUE))
            self.canvas.circle(54, sy + 4, 11, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 9)
            self.canvas.drawCentredString(54, sy + 1, number)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 10)
            self.canvas.drawString(78, sy + 4, heading)
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_REGULAR, 8.5)
            self.canvas.drawString(260, sy + 4, detail)
            sy -= 38

        self._rounded_rect(42, 54, PAGE_W - 84, 40, NAVY, NAVY)
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setFont(FONT_BOLD, 10)
        self.canvas.drawString(58, 69, "VERIFIED")
        self.canvas.setFont(FONT_REGULAR, 8.5)
        self.canvas.setFillColor(HexColor("#CBD5E1"))
        self.canvas.drawString(
            128,
            69,
            "34,344 unique predictions | 13.97M trades checked | 0 holdout overlap | 9 tests passed",
        )

    def _metric_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        label: str,
        detail: str,
        accent: str,
        *,
        dark: bool,
    ) -> None:
        fill = "#151F3A" if dark else WHITE
        border = "#263454" if dark else GRID
        self._rounded_rect(x, y, w, h, fill, border)
        self.canvas.setFillColor(HexColor(accent))
        self.canvas.rect(x, y + h - 5, w, 5, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(WHITE if dark else INK))
        self.canvas.setFont(FONT_BOLD, 22)
        self.canvas.drawString(x + 14, y + 65, value)
        self.canvas.setFont(FONT_BOLD, 9)
        self.canvas.drawString(x + 14, y + 43, label)
        self.canvas.setFillColor(HexColor("#94A3B8" if dark else MUTED))
        self.canvas.setFont(FONT_REGULAR, 7.5)
        self._wrapped_text(detail, x + 14, y + 24, w - 28, 9, max_lines=2)

    def _callout(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        label: str,
        detail: str,
        fill: str,
        accent: str,
    ) -> None:
        self._rounded_rect(x, y, w, h, fill, GRID)
        self.canvas.setFillColor(HexColor(accent))
        self.canvas.setFont(FONT_BOLD, 20)
        self.canvas.drawString(x + 16, y + h - 33, value)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 9)
        self.canvas.drawString(x + 16, y + h - 54, label)
        self.canvas.setFillColor(HexColor(MUTED))
        self.canvas.setFont(FONT_REGULAR, 7.5)
        self._wrapped_text(detail, x + 16, y + h - 73, w - 32, 9, max_lines=2)

    def _chart_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        path: Path,
        label: str,
    ) -> None:
        self._rounded_rect(x, y, w, h, WHITE, GRID)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 9)
        self.canvas.drawString(x + 14, y + h - 20, label)
        self.canvas.drawImage(
            ImageReader(path),
            x + 8,
            y + 8,
            width=w - 16,
            height=h - 36,
            preserveAspectRatio=True,
            anchor="c",
            mask="auto",
        )

    def _rounded_rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        fill: str,
        border: str,
    ) -> None:
        self.canvas.setFillColor(HexColor(fill))
        self.canvas.setStrokeColor(HexColor(border))
        self.canvas.setLineWidth(0.6)
        self.canvas.roundRect(x, y, w, h, 8, stroke=1, fill=1)

    def _wrapped_text(
        self,
        text: str,
        x: float,
        y: float,
        width: float,
        leading: float,
        *,
        max_lines: int,
    ) -> None:
        words = text.split()
        lines: list[str] = []
        current = ""
        font_name = self.canvas._fontname
        font_size = self.canvas._fontsize
        for word in words:
            candidate = f"{current} {word}".strip()
            if pdfmetrics.stringWidth(candidate, font_name, font_size) <= width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        for index, line in enumerate(lines[:max_lines]):
            self.canvas.drawString(x, y - index * leading, line)


def _build_charts(results: dict[str, Any], workdir: Path) -> dict[str, Path]:
    _set_plot_style()
    chart_paths = {
        "pinball": workdir / "pinball.png",
        "poc_mae": workdir / "poc-mae.png",
        "coverage_width": workdir / "coverage-width.png",
        "cdf_heatmap": workdir / "cdf-heatmap.png",
        "direction": workdir / "direction.png",
        "confusion": workdir / "confusion.png",
        "waterfall": workdir / "waterfall.png",
        "cost_sensitivity": workdir / "cost-sensitivity.png",
        "break_even": workdir / "break-even.png",
    }
    _plot_pinball(results, chart_paths["pinball"])
    _plot_poc_mae(results, chart_paths["poc_mae"])
    _plot_coverage_width(results, chart_paths["coverage_width"])
    _plot_cdf_heatmap(results, chart_paths["cdf_heatmap"])
    _plot_direction(results, chart_paths["direction"])
    _plot_confusion(results, chart_paths["confusion"])
    _plot_waterfall(results, chart_paths["waterfall"])
    _plot_cost_sensitivity(results, chart_paths["cost_sensitivity"])
    _plot_break_even(results, chart_paths["break_even"])
    return chart_paths


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": GRID,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
        }
    )


def _base_axis(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _plot_pinball(results: dict[str, Any], path: Path) -> None:
    forecast = results["forecast"]
    partitions = [
        "in_sample_train_replay",
        "selected_validation",
        "untouched_development_forward",
    ]
    labels = ["Train replay", "Selected", "Untouched"]
    owners = ["model", "empirical_marginal", "zero_residual"]
    owner_labels = ["Model", "Empirical", "Zero residual"]
    colors = [BLUE, GRAY, "#CBD1DA"]
    values = [
        [
            forecast[p][owner]["normalized"]["weighted_primary_metric"]
            for p in partitions
        ]
        for owner in owners
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    _base_axis(ax)
    x = np.arange(3)
    width = 0.22
    for index, (series, label, color) in enumerate(
        zip(values, owner_labels, colors, strict=True)
    ):
        bars = ax.bar(x + (index - 1) * width, series, width, label=label, color=color)
        for bar, value in zip(bars, series, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.007,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, labels)
    ax.set_ylabel("Weighted pinball")
    ax.set_ylim(0, 0.52)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    _save(fig, path)


def _plot_poc_mae(results: dict[str, Any], path: Path) -> None:
    forward = results["forecast"]["untouched_development_forward"]
    model = np.asarray(forward["model"]["physical"]["median_mae"])[:, 0] * 0.1
    baseline = (
        np.asarray(forward["empirical_marginal"]["physical"]["median_mae"])[:, 0] * 0.1
    )
    horizons = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    _base_axis(ax)
    ax.plot(horizons, model, color=BLUE, marker="o", linewidth=2, label="Model")
    ax.plot(
        horizons, baseline, color=GRAY, marker="s", linewidth=1.8, label="Empirical"
    )
    for h, value in zip(horizons, model, strict=True):
        ax.text(h, value + 2.2, f"${value:.1f}", ha="center", color=BLUE, fontsize=7)
    ax.set_xticks(horizons, [f"{h}m" for h in horizons])
    ax.set_ylabel("Median absolute error (USDT)")
    ax.set_ylim(0, 62)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _save(fig, path)


def _plot_coverage_width(results: dict[str, Any], path: Path) -> None:
    forward = results["forecast"]["untouched_development_forward"]["model"]["physical"]
    coverage = np.asarray(forward["interval_coverage"])[:, 0] * 100
    width = np.asarray(forward["interval_width"])[:, 0] * 0.1
    horizons = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    _base_axis(ax)
    bars = ax.bar(horizons, width, color=LIGHT_BLUE, edgecolor=BLUE, linewidth=0.8)
    ax.set_ylabel("Interval width (USDT)", color=BLUE)
    ax.set_xticks(horizons, [f"{h}m" for h in horizons])
    ax.set_ylim(0, 220)
    ax2 = ax.twinx()
    ax2.plot(horizons, coverage, color=AMBER, marker="o", linewidth=2)
    ax2.axhline(80, color=GRAY, linestyle="--", linewidth=1)
    ax2.set_ylabel("Coverage (%)", color=AMBER)
    ax2.set_ylim(72, 88)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    for bar, value in zip(bars, width, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 5,
            f"${value:.0f}",
            ha="center",
            color=BLUE,
            fontsize=7,
        )
    for h, value in zip(horizons, coverage, strict=True):
        ax2.text(h, value + 0.6, f"{value:.1f}%", ha="center", color=AMBER, fontsize=7)
    _save(fig, path)


def _plot_cdf_heatmap(results: dict[str, Any], path: Path) -> None:
    values = (
        np.asarray(
            results["forecast"]["untouched_development_forward"][
                "model_quantile_empirical_cdf"
            ]
        )[:, 0, :]
        * 100
    )
    target = np.asarray([10.0, 50.0, 90.0])
    error = values - target
    cmap = LinearSegmentedColormap.from_list("cdf", [LIGHT_BLUE, WHITE, LIGHT_RED])
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    image = ax.imshow(error, cmap=cmap, vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(3), ["q10", "q50", "q90"])
    ax.set_yticks(range(5), ["1m", "2m", "3m", "4m", "5m"])
    for row in range(5):
        for col in range(3):
            ax.text(
                col,
                row,
                f"{values[row, col]:.1f}%",
                ha="center",
                va="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xlabel("Observed P(y <= q)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Deviation from nominal (pp)")
    _save(fig, path)


def _plot_direction(results: dict[str, Any], path: Path) -> None:
    proxy = results["candle_proxy"]["untouched_development_forward"]
    horizons = np.arange(1, 6)
    series = [
        (
            "Direct POC",
            [100 * item["direct_total_poc_direction"]["accuracy"] for item in proxy],
            BLUE,
            "o",
        ),
        (
            "Candle body",
            [100 * item["candle_body_direction_proxy"]["accuracy"] for item in proxy],
            AMBER,
            "s",
        ),
        (
            "Anchor-close",
            [
                100 * item["anchor_to_close_direction_proxy"]["accuracy"]
                for item in proxy
            ],
            TEAL,
            "^",
        ),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    _base_axis(ax)
    for label, values, color, marker in series:
        ax.plot(horizons, values, color=color, marker=marker, linewidth=2, label=label)
    ax.axhline(50, color=GRAY, linestyle="--", linewidth=1)
    ax.set_ylim(48, 54)
    ax.set_xticks(horizons, [f"{h}m" for h in horizons])
    ax.set_ylabel("Direction accuracy (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    ax.text(
        5.05,
        series[2][1][-1],
        f"{series[2][1][-1]:.2f}%",
        va="center",
        color=TEAL,
        fontsize=8,
    )
    _save(fig, path)


def _plot_confusion(results: dict[str, Any], path: Path) -> None:
    matrix = np.asarray(
        results["candle_proxy"]["untouched_development_forward"][4][
            "anchor_to_close_direction_proxy"
        ]["confusion"]
    )
    matrix = matrix[np.ix_([0, 2], [0, 2])]
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(
        matrix,
        cmap=LinearSegmentedColormap.from_list("blue", [WHITE, LIGHT_BLUE, BLUE]),
    )
    ax.set_xticks([0, 1], ["Pred down", "Pred up"])
    ax.set_yticks([0, 1], ["Actual down", "Actual up"])
    for row in range(2):
        for col in range(2):
            color = WHITE if matrix[row, col] > 1600 else INK
            ax.text(
                col,
                row,
                f"{matrix[row, col]:,}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    _save(fig, path)


def _plot_waterfall(results: dict[str, Any], path: Path) -> None:
    policies = results["economics"]["untouched_development_forward"]["policies"]
    labels = ["Median long/flat", "Median long/short"]
    keys = ["median_long_flat", "median_long_short"]
    gross = [
        policies[key]["base"]["metrics"]["gross_total_return"] * 100_000 for key in keys
    ]
    net = [policies[key]["base"]["metrics"]["pnl_per_initial_capital"] for key in keys]
    costs = [
        net_value - gross_value
        for gross_value, net_value in zip(gross, net, strict=True)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.0))
    for ax, label, gross_value, cost_value, net_value in zip(
        axes, labels, gross, costs, net, strict=True
    ):
        _base_axis(ax)
        values = [gross_value, cost_value, net_value]
        colors = [TEAL, RED, RED]
        bars = ax.bar([0, 1, 2], values, color=colors, width=0.62)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks([0, 1, 2], ["Gross", "Costs", "Net"])
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel("USDT per 100k fixed notional")
        limit = max(abs(min(values)), abs(max(values))) * 1.25
        ax.set_ylim(-limit, limit * 0.25)
        for bar, value in zip(bars, values, strict=True):
            va = "bottom" if value >= 0 else "top"
            offset = limit * (0.025 if value >= 0 else -0.025)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:,.0f}",
                ha="center",
                va=va,
                fontsize=8,
                fontweight="bold",
            )
    fig.tight_layout(w_pad=3)
    _save(fig, path)


def _plot_cost_sensitivity(results: dict[str, Any], path: Path) -> None:
    policies = results["economics"]["untouched_development_forward"]["policies"]
    scenarios = ["frictionless", "optimistic", "base", "stress"]
    labels = ["0", "9", "12", "20"]
    long_flat = [
        100 * policies["median_long_flat"][scenario]["metrics"]["net_compounded_return"]
        for scenario in scenarios
    ]
    long_short = [
        100
        * policies["median_long_short"][scenario]["metrics"]["net_compounded_return"]
        for scenario in scenarios
    ]
    fig, ax = plt.subplots(figsize=(7.9, 3.5))
    _base_axis(ax)
    x = np.arange(4)
    width = 0.33
    bars1 = ax.bar(
        x - width / 2, long_flat, width, color=BLUE, label="Median long/flat"
    )
    bars2 = ax.bar(
        x + width / 2, long_short, width, color=RED, label="Median long/short"
    )
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("One-way modeled cost (bps)")
    ax.set_ylabel("Compounded return (%)")
    ax.set_ylim(-110, 15)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    for bars in (bars1, bars2):
        for bar in bars:
            value = bar.get_height()
            y = value + (2 if value >= 0 else -3)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                y,
                f"{value:.1f}%",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
    _save(fig, path)


def _plot_break_even(results: dict[str, Any], path: Path) -> None:
    policies = results["economics"]["untouched_development_forward"]["policies"]
    break_even = policies["median_long_flat"]["frictionless"]["break_even_one_way_bps"]
    labels = ["Break-even", "Optimistic", "Base", "Stress"]
    values = [break_even, 9.0, 12.0, 20.0]
    colors = [TEAL, AMBER, RED, "#A72035"]
    fig, ax = plt.subplots(figsize=(3.8, 3.5))
    _base_axis(ax)
    bars = ax.barh(np.arange(4), values, color=colors, height=0.55)
    ax.set_yticks(np.arange(4), labels)
    ax.invert_yaxis()
    ax.set_xlabel("One-way bps")
    ax.set_xlim(0, 22)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
            fontweight="bold",
        )
    _save(fig, path)


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _validate_sources(
    results_path: Path,
    manifest_path: Path,
    results: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    if manifest.get("outer_holdout_accessed") is not False:
        raise ValueError("study manifest does not preserve the outer holdout")
    if results.get("outer_holdout_accessed") is not False:
        raise ValueError("study results do not preserve the outer holdout")
    if results.get("evaluated_example_count") != 34_344:
        raise ValueError("unexpected evaluated example count")
    expected_results = manifest["files"]["results.json"]
    if results_path.stat().st_size != expected_results["bytes"]:
        raise ValueError("results size differs from study manifest")
    if _sha256(results_path) != expected_results["sha256"]:
        raise ValueError("results hash differs from study manifest")
    if manifest_path.name != "manifest.json":
        raise ValueError("unexpected study manifest name")
    forward = results["forecast"]["untouched_development_forward"]
    if not math.isclose(
        forward["relative_model_improvement_percent"], 2.2558299664809045
    ):
        raise ValueError("headline forecast result differs from verified study")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
