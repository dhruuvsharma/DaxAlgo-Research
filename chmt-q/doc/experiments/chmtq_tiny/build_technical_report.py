"""Build the finance-facing CHMT-Q Tiny technical evidence PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


PAGE_W, PAGE_H = landscape(A4)
NAVY = HexColor("#0B1630")
PAPER = HexColor("#F6F8FC")
WHITE = HexColor("#FFFFFF")
INK = HexColor("#15223A")
BLUE = HexColor("#2F6DF6")
BLUE_LIGHT = HexColor("#E9F0FF")
TEAL = HexColor("#00A991")
TEAL_LIGHT = HexColor("#E4F7F3")
AMBER = HexColor("#F2AD35")
AMBER_LIGHT = HexColor("#FFF4DE")
RED = HexColor("#E94B62")
RED_LIGHT = HexColor("#FDE9ED")
SLATE = HexColor("#7F8A9B")
SLATE_DARK = HexColor("#4F5D73")
LINE = HexColor("#DDE3ED")
SOFT = HexColor("#EEF2F8")

FONT_REGULAR = "SegoeUI"
FONT_BOLD = "SegoeUI-Bold"
FONT_SEMIBOLD = "SegoeUI-Semibold"
FONT_MONO = "Consolas"


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont(FONT_REGULAR, r"C:\Windows\Fonts\segoeui.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_BOLD, r"C:\Windows\Fonts\segoeuib.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_SEMIBOLD, r"C:\Windows\Fonts\seguisb.ttf"))
    pdfmetrics.registerFont(TTFont(FONT_MONO, r"C:\Windows\Fonts\consola.ttf"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_lines(text: str, font: str, size: float, width: float) -> list[str]:
    output: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            output.append("")
            continue
        words = paragraph.split()
        line = words[0]
        for word in words[1:]:
            candidate = f"{line} {word}"
            if pdfmetrics.stringWidth(candidate, font, size) <= width:
                line = candidate
            else:
                output.append(line)
                line = word
        output.append(line)
    return output


def draw_text(
    c: canvas.Canvas,
    text: str,
    x: float,
    top: float,
    width: float,
    *,
    font: str = FONT_REGULAR,
    size: float = 9,
    leading: float | None = None,
    color: colors.Color = INK,
    max_lines: int | None = None,
) -> float:
    leading = leading or size * 1.28
    lines = text_lines(text, font, size, width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        while pdfmetrics.stringWidth(lines[-1] + "...", font, size) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    c.setFillColor(color)
    c.setFont(font, size)
    y = top - size
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def round_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    fill: colors.Color = WHITE,
    stroke: colors.Color = LINE,
    radius: float = 10,
    line_width: float = 0.8,
) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(line_width)
    c.roundRect(x, y, width, height, radius, fill=1, stroke=1)


def pill(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    *,
    fill: colors.Color,
    text_color: colors.Color = WHITE,
    width: float | None = None,
) -> float:
    size = 7.4
    width = width or pdfmetrics.stringWidth(text, FONT_SEMIBOLD, size) + 18
    c.setFillColor(fill)
    c.roundRect(x, y, width, 19, 9.5, fill=1, stroke=0)
    c.setFillColor(text_color)
    c.setFont(FONT_SEMIBOLD, size)
    c.drawCentredString(x + width / 2, y + 6.1, text)
    return width


def metric_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    width: float,
    height: float,
    value: str,
    label: str,
    note: str,
    *,
    accent: colors.Color = BLUE,
    fill: colors.Color = WHITE,
) -> None:
    round_card(c, x, y, width, height, fill=fill)
    c.setFillColor(accent)
    c.roundRect(x, y, 5, height, 2.5, fill=1, stroke=0)
    c.setFillColor(INK)
    c.setFont(FONT_BOLD, 20)
    c.drawString(x + 18, y + height - 30, value)
    c.setFillColor(SLATE_DARK)
    c.setFont(FONT_SEMIBOLD, 8.2)
    c.drawString(x + 18, y + height - 48, label)
    draw_text(c, note, x + 18, y + height - 58, width - 32, size=7.2, color=SLATE)


def page_frame(c: canvas.Canvas, page: int, section: str, title: str, subtitle: str = "") -> None:
    c.setFillColor(PAPER)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont(FONT_SEMIBOLD, 7.5)
    c.drawString(38, PAGE_H - 31, f"CHMT-Q TINY  /  {section.upper()}")
    pill(c, "RESEARCH-ONLY", PAGE_W - 144, PAGE_H - 39, fill=RED, width=106)
    c.setFillColor(INK)
    c.setFont(FONT_BOLD, 23)
    c.drawString(38, PAGE_H - 68, title)
    if subtitle:
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_REGULAR, 8.5)
        c.drawString(39, PAGE_H - 84, subtitle)
    c.setStrokeColor(LINE)
    c.line(38, 35, PAGE_W - 38, 35)
    c.setFillColor(SLATE)
    c.setFont(FONT_REGULAR, 7)
    c.drawString(38, 21, "Retrospective June 2026 BTC-USDT development evidence")
    c.drawRightString(PAGE_W - 38, 21, f"Dhruv Sharma  |  {page:02d} / 12")


def paragraph_style(size: float = 7.5, color: colors.Color = INK, bold: bool = False) -> ParagraphStyle:
    return ParagraphStyle(
        name=f"p-{size}-{bold}",
        fontName=FONT_SEMIBOLD if bold else FONT_REGULAR,
        fontSize=size,
        leading=size * 1.25,
        textColor=color,
        alignment=TA_LEFT,
        spaceAfter=0,
        spaceBefore=0,
    )


def draw_table(
    c: canvas.Canvas,
    rows: list[list[Any]],
    x: float,
    top: float,
    widths: list[float],
    *,
    font_size: float = 7.3,
    header_fill: colors.Color = NAVY,
    highlight_rows: dict[int, colors.Color] | None = None,
    row_padding: float = 5,
) -> float:
    styled: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        color = WHITE if row_index == 0 else INK
        style = paragraph_style(font_size, color, bold=row_index == 0)
        styled.append([Paragraph(str(value), style) for value in row])
    table = Table(styled, colWidths=widths, repeatRows=1)
    commands: list[tuple] = [
        ("BACKGROUND", (0, 0), (-1, 0), header_fill),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), row_padding),
        ("RIGHTPADDING", (0, 0), (-1, -1), row_padding),
        ("TOPPADDING", (0, 0), (-1, -1), row_padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), row_padding),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, HexColor("#F9FAFD")]),
    ]
    if highlight_rows:
        for row, fill in highlight_rows.items():
            commands.append(("BACKGROUND", (0, row), (-1, row), fill))
    table.setStyle(TableStyle(commands))
    width, height = table.wrap(sum(widths), PAGE_H)
    table.drawOn(c, x, top - height)
    return top - height


def heat_color(value: float, minimum: float, maximum: float, base: colors.Color = BLUE) -> colors.Color:
    span = max(maximum - minimum, 1e-9)
    strength = 0.18 + 0.74 * (value - minimum) / span
    return colors.Color(
        1 - strength * (1 - base.red),
        1 - strength * (1 - base.green),
        1 - strength * (1 - base.blue),
    )


def draw_heatmap(
    c: canvas.Canvas,
    matrix: np.ndarray,
    x: float,
    top: float,
    cell_w: float,
    cell_h: float,
    row_labels: list[str],
    col_labels: list[str],
    *,
    fmt: str = "{:.1f}",
    color_mode: str = "positive",
) -> None:
    label_w = 35
    minimum = float(np.nanmin(matrix))
    maximum = float(np.nanmax(matrix))
    c.setFont(FONT_SEMIBOLD, 7)
    c.setFillColor(SLATE_DARK)
    for col, label in enumerate(col_labels):
        c.drawCentredString(x + label_w + col * cell_w + cell_w / 2, top + 7, label)
    for row, label in enumerate(row_labels):
        y = top - (row + 1) * cell_h
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_SEMIBOLD, 7)
        c.drawRightString(x + label_w - 7, y + cell_h / 2 - 2.5, label)
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if color_mode == "coverage":
                error = abs(value - 0.8)
                base = TEAL if error <= 0.04 else AMBER if error <= 0.08 else RED
                fill = heat_color(error, 0, max(0.12, float(np.max(abs(matrix - 0.8)))), base)
            else:
                fill = heat_color(value, minimum, maximum, BLUE)
            cx = x + label_w + col * cell_w
            c.setFillColor(fill)
            c.setStrokeColor(WHITE)
            c.rect(cx, y, cell_w, cell_h, fill=1, stroke=1)
            brightness = 0.299 * fill.red + 0.587 * fill.green + 0.114 * fill.blue
            c.setFillColor(INK if brightness > 0.66 else WHITE)
            c.setFont(FONT_SEMIBOLD, 7.3)
            c.drawCentredString(cx + cell_w / 2, y + cell_h / 2 - 2.5, fmt.format(value))


def draw_dot_chart(
    c: canvas.Canvas,
    items: list[tuple[str, float, colors.Color]],
    x: float,
    top: float,
    width: float,
    *,
    minimum: float,
    maximum: float,
) -> None:
    label_w = 155
    chart_x = x + label_w
    chart_w = width - label_w - 48
    row_h = 32
    c.setStrokeColor(LINE)
    for tick in np.linspace(minimum, maximum, 5):
        tx = chart_x + chart_w * (tick - minimum) / (maximum - minimum)
        c.line(tx, top + 8, tx, top - len(items) * row_h + 9)
        c.setFillColor(SLATE)
        c.setFont(FONT_REGULAR, 6.8)
        c.drawCentredString(tx, top + 13, f"{tick:.2f}")
    for index, (label, value, color) in enumerate(items):
        y = top - (index + 1) * row_h + 13
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD if index == 0 else FONT_REGULAR, 7.6)
        c.drawString(x, y - 2.5, label)
        vx = chart_x + chart_w * (value - minimum) / (maximum - minimum)
        c.setStrokeColor(color)
        c.setLineWidth(2)
        c.line(chart_x, y, vx, y)
        c.setFillColor(color)
        c.circle(vx, y, 4.3, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD, 7.5)
        c.drawRightString(x + width, y - 2.5, f"{value:.6f}")


def draw_line_chart(
    c: canvas.Canvas,
    values: list[tuple[int, float]],
    x: float,
    y: float,
    width: float,
    height: float,
    baseline: float,
) -> None:
    ymin = min(value for _, value in values) - 0.001
    ymax = max(max(value for _, value in values), baseline) + 0.001
    c.setStrokeColor(LINE)
    c.rect(x, y, width, height, fill=0, stroke=1)
    for step in np.linspace(ymin, ymax, 4):
        py = y + height * (step - ymin) / (ymax - ymin)
        c.setStrokeColor(LINE)
        c.line(x, py, x + width, py)
        c.setFillColor(SLATE)
        c.setFont(FONT_REGULAR, 6.5)
        c.drawRightString(x - 5, py - 2.4, f"{step:.3f}")
    baseline_y = y + height * (baseline - ymin) / (ymax - ymin)
    c.setStrokeColor(SLATE)
    c.setDash(3, 2)
    c.line(x, baseline_y, x + width, baseline_y)
    c.setDash()
    points: list[tuple[float, float]] = []
    for epoch, value in values:
        px = x + width * (epoch - 1) / max(len(values) - 1, 1)
        py = y + height * (value - ymin) / (ymax - ymin)
        points.append((px, py))
    c.setStrokeColor(BLUE)
    c.setLineWidth(2)
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    c.drawPath(path, fill=0, stroke=1)
    best_epoch = min(values, key=lambda item: item[1])[0]
    for epoch, point in zip((item[0] for item in values), points):
        c.setFillColor(TEAL if epoch == best_epoch else BLUE)
        c.circle(point[0], point[1], 4 if epoch == best_epoch else 2.7, fill=1, stroke=0)
        c.setFillColor(SLATE)
        c.setFont(FONT_REGULAR, 6.4)
        c.drawCentredString(point[0], y - 12, str(epoch))


def cover_page(c: canvas.Canvas, d: dict[str, Any]) -> None:
    c.setFillColor(NAVY)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.setFillColor(BLUE)
    c.circle(PAGE_W - 70, PAGE_H - 55, 115, fill=1, stroke=0)
    c.setFillColor(HexColor("#173365"))
    c.circle(PAGE_W - 15, 10, 180, fill=1, stroke=0)
    pill(c, "TECHNICAL EVIDENCE MEMO", 48, PAGE_H - 67, fill=BLUE, width=164)
    c.setFillColor(WHITE)
    c.setFont(FONT_BOLD, 34)
    c.drawString(48, PAGE_H - 135, "CHMT-Q Tiny")
    c.setFont(FONT_SEMIBOLD, 21)
    c.drawString(48, PAGE_H - 170, "Retrospective BTC-USDT Forecast Evaluation")
    c.setFillColor(HexColor("#B7C8EB"))
    c.setFont(FONT_REGULAR, 11)
    c.drawString(49, PAGE_H - 197, "Better proper score. No trading conclusion.")

    tiny = d["benchmarks"]["chmt_q_tiny"]
    baseline = d["benchmarks"]["empirical_marginal_train_only"]
    metrics = [
        (
            f"{d['registered_result_reconciliation']['development_improvement_percent']:.2f}%",
            "LOWER FORWARD PINBALL",
            f"{tiny['weighted_normalized_pinball']:.6f} vs {baseline['weighted_normalized_pinball']:.6f}",
            BLUE,
        ),
        (
            f"{d['model']['parameters']:,}",
            "TRAINABLE PARAMETERS",
            f"{d['model']['parameter_reduction_percent']:.2f}% below reference size",
            TEAL,
        ),
        (
            f"{100*tiny['coverage_overall']:.2f}%",
            "Q10-Q90 COVERAGE",
            "Nominal central-band target: 80%",
            AMBER,
        ),
    ]
    card_w = 220
    for index, (value, label, note, accent) in enumerate(metrics):
        x = 48 + index * (card_w + 18)
        c.setFillColor(HexColor("#14264A"))
        c.setStrokeColor(HexColor("#315183"))
        c.roundRect(x, 145, card_w, 108, 10, fill=1, stroke=1)
        c.setFillColor(accent)
        c.rect(x, 145, 5, 108, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_BOLD, 22)
        c.drawString(x + 18, 210, value)
        c.setFont(FONT_SEMIBOLD, 7.7)
        c.drawString(x + 18, 188, label)
        draw_text(c, note, x + 18, 176, card_w - 32, size=7.3, color=HexColor("#B7C8EB"))

    c.setFillColor(AMBER)
    c.roundRect(48, 78, PAGE_W - 96, 38, 8, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont(FONT_SEMIBOLD, 10)
    c.drawString(63, 92, "DECISION: CONTINUE RESEARCH. DO NOT ALLOCATE CAPITAL OR CLAIM PROFITABILITY.")
    c.setFillColor(HexColor("#B7C8EB"))
    c.setFont(FONT_REGULAR, 7.4)
    c.drawString(48, 38, "Dhruv Sharma  |  OKX BTC-USDT spot  |  Previously studied 21-24 June 2026 partition")


def page_2(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 2, "Executive decision", "What the result establishes - and what it does not")
    tiny = d["benchmarks"]["chmt_q_tiny"]
    best_context_name = min(
        [
            "empirical_marginal_train_only",
            "utc_hour_train_only",
            "trailing_three_complete_days",
            "causal_residual_persistence",
        ],
        key=lambda key: d["benchmarks"][key]["weighted_normalized_pinball"],
    )
    best_context = d["benchmarks"][best_context_name]
    skill = 100 * (
        best_context["weighted_normalized_pinball"] - tiny["weighted_normalized_pinball"]
    ) / best_context["weighted_normalized_pinball"]
    round_card(c, 38, 328, 390, 165, fill=WHITE)
    c.setFillColor(BLUE)
    c.setFont(FONT_SEMIBOLD, 8)
    c.drawString(55, 469, "AUDITED HEADLINE")
    claim = (
        "On 5,724 overlapping development examples, seed-17 CHMT-Q Tiny produced "
        "weighted normalized pinball loss of 0.242370 versus 0.257193 for the "
        "registered train-only empirical baseline. The descriptive reduction was "
        "5.7633%, and every one of the 25 horizon-target cells was lower."
    )
    draw_text(c, claim, 55, 450, 352, font=FONT_SEMIBOLD, size=10.4, leading=14, color=INK)
    draw_text(
        c,
        f"Against the best-performing added simple context baseline ({best_context_name.replace('_', ' ')}), Tiny was {skill:.2f}% lower. That comparison is post hoc, not registered.",
        55,
        362,
        352,
        size=7.7,
        color=SLATE_DARK,
    )

    stages = [
        ("1", "Marginal quantile score", "EVIDENCED", TEAL, "Proper score improved across all 25 cells."),
        ("2", "Conditional signal", "WEAK", AMBER, "H5 total-POC correlation and dispersion remain small."),
        ("3", "Executable economics", "NOT MEASURED", RED, "No Tiny cost, fill, PnL, Sharpe, or capacity study."),
    ]
    for index, (number, label, status, color, note) in enumerate(stages):
        y = 426 - index * 104
        round_card(c, 453, y - 45, 350, 82, fill=WHITE)
        c.setFillColor(color)
        c.circle(478, y - 4, 15, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_BOLD, 10)
        c.drawCentredString(478, y - 7.5, number)
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD, 9.5)
        c.drawString(505, y + 10, label)
        pill(c, status, 681, y + 2, fill=color, width=105)
        draw_text(c, note, 505, y - 7, 270, size=7.4, color=SLATE_DARK)

    round_card(c, 38, 64, 765, 90, fill=AMBER_LIGHT, stroke=AMBER)
    c.setFillColor(AMBER)
    c.setFont(FONT_BOLD, 9)
    c.drawString(55, 131, "FUND-FACING INTERPRETATION")
    draw_text(
        c,
        "A compact hierarchical Transformer achieved lower aggregate weighted normalized pinball loss than several simple retrospective baselines on one accessed period. This is not evidence of alpha, calibration under regime change, architectural superiority, or a profitable strategy.",
        55,
        117,
        730,
        size=8.6,
        leading=12,
        color=INK,
    )


def page_3(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 3, "Evidence boundary", "Chronology, dependence, and governance")
    timeline_y = 388
    start_x = 60
    segments = [
        (18, "TRAIN", "01-18 JUN", BLUE, "25,758"),
        (2, "SELECT", "19-20 JUN", TEAL, "2,862"),
        (4, "DEVELOP", "21-24 JUN", AMBER, "5,724"),
        (6, "RETIRED", "25-30 JUN", RED, "not scored"),
    ]
    total = sum(days for days, *_ in segments)
    available_w = 715
    x = start_x
    for days, label, dates, color, count in segments:
        width = available_w * days / total
        c.setFillColor(color)
        c.roundRect(x, timeline_y, width - 4, 62, 7, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_BOLD, 8.5)
        c.drawCentredString(x + (width - 4) / 2, timeline_y + 39, label)
        c.setFont(FONT_REGULAR, 7)
        c.drawCentredString(x + (width - 4) / 2, timeline_y + 25, dates)
        c.drawCentredString(x + (width - 4) / 2, timeline_y + 11, count)
        x += width
    c.setFillColor(SLATE_DARK)
    c.setFont(FONT_REGULAR, 7.2)
    c.drawString(60, timeline_y - 15, "UTC sessions. June 25-30 is process-accessed and retired; it is not a sealed outer holdout.")

    cards = [
        (
            "RETROSPECTIVE STATUS",
            "The development partition was already studied. All added metrics and context baselines in this memo are post hoc and hypothesis-generating.",
            RED,
        ),
        (
            "ROW DEPENDENCE",
            "Adjacent rows share 63 of 64 bars and four of five longest-horizon labels. Row-wise p-values and confidence intervals would be anti-conservative.",
            AMBER,
        ),
        (
            "TARGET CONTRACT",
            "Five footprint descriptors are forecast at five one-minute horizons. They are not returns, trade actions, or a joint predictive distribution.",
            BLUE,
        ),
    ]
    for index, (title, body, color) in enumerate(cards):
        x = 38 + index * 258
        round_card(c, x, 205, 240, 135, fill=WHITE)
        c.setFillColor(color)
        c.rect(x, 205, 240, 5, fill=1, stroke=0)
        c.setFillColor(color)
        c.setFont(FONT_SEMIBOLD, 7.8)
        c.drawString(x + 16, 310, title)
        draw_text(c, body, x + 16, 291, 208, size=8, leading=11, color=INK)

    round_card(c, 38, 60, 765, 115, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 149, "Artifact governance")
    governance = [
        ("168", "Arrow shards hash/schema verified"),
        ("42,930", "content-addressed corpus rows"),
        ("1.725 GB", "verified Arrow bytes"),
        ("0", "accepted nonfinite outputs"),
    ]
    for index, (value, label) in enumerate(governance):
        x = 55 + index * 184
        c.setFillColor(TEAL if index != 3 else BLUE)
        c.setFont(FONT_BOLD, 16)
        c.drawString(x, 112, value)
        draw_text(c, label, x, 99, 150, size=7.2, color=SLATE_DARK)


def page_4(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 4, "Model contract", "Causal hierarchy, hard masks, quality gates, direct quantiles")
    stages = [
        (45, 382, 145, 92, "MULTI-SCALE INPUT", "Bars  64 x 16\nL1  240 x 16\nL2  240 x 2 x 10 x 12", BLUE),
        (220, 382, 145, 92, "HIERARCHY", "Bars: causal tokens\nL1: 4-frame patches\nL2: side/rank then time", BLUE),
        (395, 382, 145, 92, "GATED MEMORY", "64 + 60 + 60\n184 tokens max\nhard-mask eligibility", TEAL),
        (570, 382, 195, 92, "DIRECT QUANTILES", "5 learned horizon queries\nraw output 5 x 17\nordered q10/q50/q90", AMBER),
    ]
    for x, ny, width, height, title, body, color in stages:
        round_card(c, x, ny, width, height, fill=WHITE, stroke=color)
        c.setFillColor(color)
        c.setFont(FONT_SEMIBOLD, 7.5)
        c.drawString(x + 12, ny + height - 19, title)
        draw_text(c, body, x + 12, ny + height - 31, width - 24, size=7.4, leading=10, color=INK)
    for start, end in [((190, 427), (220, 427)), ((365, 427), (395, 427)), ((540, 427), (570, 427))]:
        c.setStrokeColor(SLATE)
        c.setLineWidth(1.5)
        c.line(start[0], start[1], end[0], end[1])
        c.setFillColor(SLATE)
        c.wedge(end[0] - 7, end[1] - 4, end[0] + 1, end[1] + 4, 270, 180, fill=1, stroke=0)

    rows = [
        ["Subsystem", "Tiny setting", "Reference setting", "Interpretation"],
        ["Shared width", "64", "128", "Capacity halved"],
        ["L2 level width", "16", "32", "Side/rank hierarchy retained"],
        ["Encoder depth", "1 per branch", "3 per branch", "Shorter temporal stacks"],
        ["Decoder depth", "2", "2", "Five direct horizons retained"],
        ["Parameters", f"{d['model']['parameters']:,}", f"{d['model']['reference_parameters']:,}", f"{d['model']['parameter_reduction_percent']:.2f}% fewer"],
        ["Presence logits", "2 inactive slots", "2 trained slots", "Tiny loss weight fixed to zero"],
    ]
    draw_table(c, rows, 38, 338, [150, 120, 120, 365], font_size=7.5, highlight_rows={5: BLUE_LIGHT, 6: AMBER_LIGHT})
    round_card(c, 38, 58, 765, 60, fill=AMBER_LIGHT, stroke=AMBER)
    draw_text(
        c,
        "Causal means temporal availability and masking, not causal inference. Hard masks determine eligibility; soft reliability gates cannot restore unavailable evidence.",
        55,
        102,
        730,
        size=8.2,
        color=INK,
    )


def page_5(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 5, "Training and operations", "Selection curve, objective, and reproducibility ledger")
    history = d["operational"]["training_history"]
    values = [(int(row["epoch"]), float(row["validation_pinball"])) for row in history]
    baseline = float(history[0]["validation_baseline_pinball"])
    round_card(c, 38, 250, 470, 245, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 470, "Validation selection curve")
    draw_line_chart(c, values, 92, 290, 385, 145, baseline)
    c.setFillColor(SLATE_DARK)
    c.setFont(FONT_REGULAR, 7)
    c.drawString(92, 265, "Epoch (teal marks selected epoch 5; dashed line is empirical baseline)")

    round_card(c, 528, 250, 275, 245, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(545, 470, "Objective and optimizer")
    settings = [
        ("Primary", "weighted normalized pinball"),
        ("Auxiliary", "median Huber + deep supervision"),
        ("Presence", "weight 0; labels 100% positive"),
        ("Optimizer", "AdamW, matrix decay only"),
        ("Learning rate", "3e-4 cosine to 3e-5"),
        ("Effective batch", "32 x 4 = 128"),
        ("Gradient clip", "global norm 1.0"),
        ("Selection", "epoch 5 of 8; seed 17"),
    ]
    for index, (key, value) in enumerate(settings):
        yy = 446 - index * 23
        c.setFillColor(SLATE)
        c.setFont(FONT_REGULAR, 7.2)
        c.drawString(545, yy, key)
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD, 7.5)
        c.drawRightString(784, yy, value)
        c.setStrokeColor(SOFT)
        c.line(545, yy - 7, 784, yy - 7)

    ops = [
        (f"{d['operational']['training_wall_seconds']/60:.1f} min", "TRAIN WALL TIME", "Local shared GPU environment", BLUE),
        (f"{d['operational']['batch_one_latency_ms_shared_gtx1650']:.2f} ms", "BATCH-ONE LATENCY", "Shared GTX 1650; not an SLA", TEAL),
        ("0.0", "REPEAT MAX ABS DIFF", "Two identical inference calls", TEAL),
        (f"{d['operational']['optimizer_updates']:,}", "OPTIMIZER UPDATES", "Eight completed epochs", AMBER),
    ]
    for index, (value, label, note, color) in enumerate(ops):
        metric_card(c, 38 + index * 195, 82, 180, 135, value, label, note, accent=color)


def page_6(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 6, "Benchmark suite", "Registered result plus post hoc context baselines")
    order = [
        ("CHMT-Q Tiny", "chmt_q_tiny", BLUE),
        ("Trailing 3 complete days", "trailing_three_complete_days", TEAL),
        ("UTC-hour train-only", "utc_hour_train_only", TEAL),
        ("Empirical marginal", "empirical_marginal_train_only", SLATE_DARK),
        ("Zero degenerate", "zero_degenerate", AMBER),
        ("Residual persistence", "causal_residual_persistence", RED),
    ]
    items = [(label, d["benchmarks"][key]["weighted_normalized_pinball"], color) for label, key, color in order]
    round_card(c, 38, 285, 765, 210, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 470, "Weighted normalized pinball - lower is better")
    draw_dot_chart(c, items, 55, 445, 720, minimum=0.23, maximum=0.39)

    empirical = d["benchmarks"]["empirical_marginal_train_only"]["weighted_normalized_pinball"]
    rows = [["Benchmark", "Pinball", "Skill vs empirical", "Coverage", "Status"]]
    status = {
        "chmt_q_tiny": "selected on validation",
        "empirical_marginal_train_only": "registered baseline",
        "utc_hour_train_only": "post hoc context",
        "trailing_three_complete_days": "post hoc adaptive",
        "zero_degenerate": "post hoc sanity check",
        "causal_residual_persistence": "post hoc causal anchor",
    }
    for label, key, _ in order:
        value = d["benchmarks"][key]["weighted_normalized_pinball"]
        skill = 100 * (empirical - value) / empirical
        coverage = 100 * d["benchmarks"][key]["coverage_overall"]
        rows.append([label, f"{value:.6f}", f"{skill:+.2f}%", f"{coverage:.2f}%", status[key]])
    draw_table(c, rows, 38, 260, [205, 100, 120, 95, 245], font_size=7.2, highlight_rows={1: BLUE_LIGHT, 4: SOFT})
    draw_text(
        c,
        "Score composition: total/buy/sell POC receive 75% of target weight; log-volume and delta receive 12.5% each; horizon weights decay as 1/sqrt(h). Only the empirical marginal comparator was frozen before the original evaluation. The four added baselines are post hoc context, not new evidence.",
        45,
        82,
        750,
        size=7.2,
        leading=9.5,
        color=SLATE_DARK,
    )


def page_7(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 7, "Primary result breadth", "Aggregate gains and all 25 horizon-target cells")
    validation_model = d["registered_result_reconciliation"]["validation_model"]
    validation_base = d["registered_result_reconciliation"]["validation_baseline"]
    development_model = d["registered_result_reconciliation"]["development_model"]
    development_base = d["registered_result_reconciliation"]["development_baseline"]
    pairs = [
        ("Validation", validation_model, validation_base, 100 * (validation_base - validation_model) / validation_base),
        ("Development-forward", development_model, development_base, 100 * (development_base - development_model) / development_base),
    ]
    round_card(c, 38, 350, 355, 145, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 470, "Tiny versus registered empirical baseline")
    for index, (label, model, base, skill) in enumerate(pairs):
        yy = 430 - index * 58
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_SEMIBOLD, 7.5)
        c.drawString(55, yy + 10, label)
        chart_x, chart_w = 155, 165
        minimum, maximum = 0.19, 0.27
        mx = chart_x + chart_w * (model - minimum) / (maximum - minimum)
        bx = chart_x + chart_w * (base - minimum) / (maximum - minimum)
        c.setStrokeColor(LINE)
        c.setLineWidth(4)
        c.line(mx, yy, bx, yy)
        c.setFillColor(BLUE)
        c.circle(mx, yy, 5, fill=1, stroke=0)
        c.setFillColor(SLATE_DARK)
        c.circle(bx, yy, 5, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.setFont(FONT_BOLD, 9)
        c.drawRightString(373, yy - 3, f"{skill:.2f}%")
        c.setFillColor(SLATE)
        c.setFont(FONT_REGULAR, 6.5)
        c.drawString(chart_x, yy - 13, f"Tiny {model:.6f}")
        c.drawRightString(chart_x + chart_w, yy - 13, f"Base {base:.6f}")

    round_card(c, 413, 350, 390, 145, fill=WHITE)
    metric_card(c, 430, 370, 165, 100, "25 / 25", "CELL WINS", "Tiny below empirical baseline", accent=TEAL, fill=WHITE)
    metric_card(c, 614, 370, 170, 100, "0.90%-7.19%", "CELL SKILL RANGE", "No target/horizon cell reversed", accent=BLUE, fill=WHITE)

    skill = np.array(d["cell_skill_vs_empirical_percent"])
    round_card(c, 38, 95, 455, 220, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 292, "Cell-level pinball skill versus empirical (%)")
    draw_heatmap(
        c,
        skill,
        62,
        260,
        68,
        31,
        ["H1", "H2", "H3", "H4", "H5"],
        ["Total", "Buy", "Sell", "Log-vol", "Delta"],
        fmt="{:.1f}",
    )

    round_card(c, 513, 95, 290, 220, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(530, 292, "Mean skill decomposition")
    target_names = ["Total POC", "Buy POC", "Sell POC", "Log-volume", "Delta"]
    target_skill = d["target_mean_skill_percent"]
    for index, (label, value) in enumerate(zip(target_names, target_skill)):
        yy = 258 - index * 31
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_REGULAR, 7.2)
        c.drawString(530, yy, label)
        c.setFillColor(BLUE_LIGHT)
        c.roundRect(610, yy - 2, 130, 9, 4.5, fill=1, stroke=0)
        c.setFillColor(BLUE)
        c.roundRect(610, yy - 2, 130 * value / 7.0, 9, 4.5, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD, 7.2)
        c.drawRightString(778, yy, f"{value:.2f}%")
    draw_text(c, "Gain narrows with horizon: 5.29% at H1 to 4.14% at H5 on an equal-cell average.", 530, 114, 245, size=7.2, color=SLATE_DARK)


def page_8(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 8, "Distribution diagnostics", "Coverage, sharpness, quantile hits, and output-path checks")
    tiny = d["benchmarks"]["chmt_q_tiny"]
    empirical = d["benchmarks"]["empirical_marginal_train_only"]
    metric_card(c, 38, 374, 180, 115, f"{100*tiny['coverage_overall']:.2f}%", "TINY COVERAGE", "Q10-Q90 central band", accent=TEAL)
    metric_card(c, 233, 374, 180, 115, f"{100*empirical['coverage_overall']:.2f}%", "EMPIRICAL COVERAGE", "Wider is not automatically better", accent=SLATE_DARK)
    metric_card(c, 428, 374, 180, 115, f"{100*tiny['coverage_error_overall']:+.2f} pp", "NOMINAL GAP", "Descriptive only; target is 80%", accent=BLUE)
    metric_card(c, 623, 374, 180, 115, "0 / 0", "CROSSINGS / SATURATION", "No quantile crossing; delta tails unsaturated", accent=TEAL)

    coverage = np.array(tiny["coverage_by_cell"])
    round_card(c, 38, 108, 455, 225, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 310, "Tiny q10-q90 coverage by cell")
    draw_heatmap(
        c,
        coverage,
        62,
        278,
        68,
        31,
        ["H1", "H2", "H3", "H4", "H5"],
        ["Total", "Buy", "Sell", "Log-vol", "Delta"],
        fmt="{:.0%}",
        color_mode="coverage",
    )

    round_card(c, 513, 108, 290, 225, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(530, 310, "Target-mean diagnostics")
    hit = np.array(tiny["quantile_hit_rate_by_cell"])
    width = np.array(tiny["mean_width_normalized_by_cell"])
    interval = np.array(tiny["interval_score_normalized_by_cell"])
    rows = [["Target", "q10 hit", "q50 hit", "q90 hit", "Norm width", "Interval score"]]
    labels = ["Total", "Buy", "Sell", "Log-vol", "Delta"]
    for target, label in enumerate(labels):
        rates = hit[:, target, :].mean(axis=0)
        rows.append(
            [
                label,
                f"{rates[0]:.1%}",
                f"{rates[1]:.1%}",
                f"{rates[2]:.1%}",
                f"{width[:, target].mean():.2f}",
                f"{interval[:, target].mean():.2f}",
            ]
        )
    draw_table(c, rows, 526, 288, [66, 39, 39, 39, 49, 58], font_size=6.2, row_padding=3.5)
    draw_text(
        c,
        "Coverage near 80% does not prove calibration. The 25 cells differ materially, and interval width must be reported in target-specific units or normalized scale - never averaged across ticks, log-volume, and fractions.",
        530,
        146,
        255,
        size=7.2,
        leading=9.5,
        color=SLATE_DARK,
    )


def page_9(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 9, "Conditional median utility", "Proper-score gains coexist with weak H5 location signal")
    names = [
        ("Tiny", "chmt_q_tiny"),
        ("Empirical", "empirical_marginal_train_only"),
        ("UTC hour", "utc_hour_train_only"),
        ("Trailing 3d", "trailing_three_complete_days"),
        ("Persistence", "causal_residual_persistence"),
    ]
    rows = [["H5 total-POC", "MAE ticks", "RMSE", "Pearson", "Spearman", "Dispersion", "Sign MCC"]]
    for label, key in names:
        h5 = d["benchmarks"][key]["h5_total_poc"]
        rows.append(
            [
                label,
                f"{h5['mae_ticks']:.1f}",
                f"{h5['rmse_ticks']:.1f}",
                "n/a" if h5["pearson"] is None else f"{h5['pearson']:.3f}",
                "n/a" if h5["spearman"] is None else f"{h5['spearman']:.3f}",
                f"{h5['dispersion_ratio']:.3f}",
                "n/a" if h5["sign"]["mcc"] is None else f"{h5['sign']['mcc']:.3f}",
            ]
        )
    draw_table(c, rows, 38, 487, [130, 90, 85, 85, 85, 85, 90], font_size=7.3, highlight_rows={1: BLUE_LIGHT})

    h5 = d["benchmarks"]["chmt_q_tiny"]["h5_total_poc"]
    cards = [
        (f"{h5['pearson']:.3f}", "PEARSON", "Weak linear association", AMBER),
        (f"{h5['spearman']:.3f}", "SPEARMAN", "Weak rank association", AMBER),
        (f"{h5['dispersion_ratio']:.3f}", "DISPERSION RATIO", "Medians strongly compressed", RED),
        (f"{h5['sign']['mcc']:.3f}", "SIGN MCC", "Not an executable direction test", AMBER),
    ]
    for index, (value, label, note, color) in enumerate(cards):
        metric_card(c, 38 + index * 195, 218, 180, 115, value, label, note, accent=color)

    round_card(c, 38, 75, 765, 110, fill=RED_LIGHT, stroke=RED)
    c.setFillColor(RED)
    c.setFont(FONT_BOLD, 9)
    c.drawString(55, 157, "DECISION-UTILITY GAP")
    draw_text(
        c,
        "Tiny improves the marginal quantile score and H5 total-POC MAE by about 0.20% versus the empirical median, but its H5 medians show weak association with realized variation. Predicted-median standard deviation is only 7.1% of realized standard deviation. Better proper scoring is not equivalent to useful direction, timing, or edge.",
        55,
        142,
        730,
        size=8.5,
        leading=12,
        color=INK,
    )


def page_10(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 10, "Temporal stability", "Four forward days and a 64-phase overlap sensitivity")
    daily = d["daily_stability"]
    round_card(c, 38, 245, 470, 250, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 470, "Daily relative improvement versus empirical")
    max_value = max(item["relative_improvement_percent"] for item in daily)
    for index, item in enumerate(daily):
        yy = 425 - index * 45
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_SEMIBOLD, 7.5)
        c.drawString(55, yy + 4, item["date"])
        c.setFillColor(BLUE_LIGHT)
        c.roundRect(155, yy, 260, 13, 6.5, fill=1, stroke=0)
        width = 260 * item["relative_improvement_percent"] / max_value
        c.setFillColor(TEAL)
        c.roundRect(155, yy, width, 13, 6.5, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD, 8)
        c.drawRightString(475, yy + 2.5, f"{item['relative_improvement_percent']:.2f}%")
    draw_text(c, "All four days favored Tiny, but four days are not an inferential sample.", 55, 277, 420, size=7.3, color=SLATE_DARK)

    phase = d["overlap_sensitivity"]
    round_card(c, 528, 245, 275, 250, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(545, 470, "64-minute phase sensitivity")
    metric_card(c, 545, 353, 112, 88, f"{phase['favorable_phases']} / {phase['phase_count']}", "FAVORABLE PHASES", "One row every 64 minutes", accent=TEAL)
    metric_card(c, 674, 353, 112, 88, f"{phase['relative_improvement_median_percent']:.2f}%", "MEDIAN SKILL", "Across clock phases", accent=BLUE)
    c.setFillColor(SLATE_DARK)
    c.setFont(FONT_REGULAR, 7.2)
    c.drawString(545, 320, "Relative improvement range")
    c.setStrokeColor(LINE)
    c.setLineWidth(6)
    c.line(555, 292, 776, 292)
    min_value = phase["relative_improvement_min_percent"]
    max_value = phase["relative_improvement_max_percent"]
    med_value = phase["relative_improvement_median_percent"]
    for value, color, radius in [(min_value, AMBER, 4), (med_value, BLUE, 5), (max_value, TEAL, 4)]:
        px = 555 + 221 * value / max_value
        c.setFillColor(color)
        c.circle(px, 292, radius, fill=1, stroke=0)
    c.setFillColor(SLATE)
    c.setFont(FONT_REGULAR, 6.8)
    c.drawString(545, 268, f"min {min_value:.2f}%")
    c.drawCentredString(665, 268, f"median {med_value:.2f}%")
    c.drawRightString(786, 268, f"max {max_value:.2f}%")

    round_card(c, 38, 72, 765, 135, fill=AMBER_LIGHT, stroke=AMBER)
    c.setFillColor(AMBER)
    c.setFont(FONT_BOLD, 9)
    c.drawString(55, 181, "WHY THIS IS A SENSITIVITY, NOT A CONFIDENCE INTERVAL")
    draw_text(
        c,
        "The phase check samples one row every 64 minutes for each possible clock phase, reducing shared 64-minute bar context. Every phase remains favorable, which is useful robustness context. It does not solve market nonstationarity, target overlap, model-selection reuse, or the fact that only four UTC days are observed. No row-wise p-value, Sharpe inference, or bootstrap claim is reported.",
        55,
        164,
        730,
        size=8,
        leading=11,
        color=INK,
    )


def page_11(c: canvas.Canvas, d: dict[str, Any]) -> None:
    page_frame(c, 11, "Concrete execution trace", "One input/output path - illustrative, not representative")
    trace = d["example_trace"]
    round_card(c, 38, 365, 765, 130, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(55, 469, trace["example_id"])
    shape_text = "  |  ".join(
        [
            f"bars {trace['input_shapes']['bars']}",
            f"L1 {trace['input_shapes']['l1']}",
            f"L2 {trace['input_shapes']['l2']}",
            f"quality {trace['input_shapes']['quality']}",
        ]
    )
    c.setFillColor(SLATE_DARK)
    c.setFont(FONT_MONO, 7.3)
    c.drawString(55, 449, shape_text)
    values = [
        (str(trace["valid_input_counts"]["bars"]), "valid bars"),
        (str(trace["valid_input_counts"]["l1"]), "valid L1 frames"),
        (str(trace["valid_input_counts"]["l2_levels"]), "valid L2 cells"),
        (str(trace["eligible_memory_tokens"]), "eligible memory tokens"),
    ]
    for index, (value, label) in enumerate(values):
        x = 55 + index * 180
        c.setFillColor(BLUE if index < 3 else TEAL)
        c.setFont(FONT_BOLD, 15)
        c.drawString(x, 414, value)
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_REGULAR, 7.1)
        c.drawString(x, 399, label)

    target_labels = ["Total POC ticks", "Buy POC ticks", "Sell POC ticks", "Log-volume residual", "Delta fraction"]
    forecast = trace["horizon_1_forecast_q10_q50_q90"]
    realized = trace["horizon_1_realized"]
    rows = [["Horizon-1 target", "q10", "q50", "q90", "Realized", "Inside band"]]
    for label, values, actual in zip(target_labels, forecast, realized):
        inside = values[0] <= actual <= values[2]
        rows.append(
            [
                label,
                f"{values[0]:.3f}",
                f"{values[1]:.3f}",
                f"{values[2]:.3f}",
                f"{actual:.3f}",
                "yes" if inside else "no",
            ]
        )
    draw_table(c, rows, 38, 337, [190, 100, 100, 100, 100, 95], font_size=7.5, highlight_rows={1: BLUE_LIGHT})

    gates = trace["modality_gates"]
    round_card(c, 38, 62, 365, 92, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 8.5)
    c.drawString(55, 132, "Quality-bounded modality gates")
    for index, (label, value, color) in enumerate(zip(["Bars", "L1", "L2"], gates, [BLUE, TEAL, AMBER])):
        x = 55 + index * 110
        c.setFillColor(color)
        c.setFont(FONT_BOLD, 13)
        c.drawString(x, 103, f"{value:.3f}")
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_REGULAR, 7)
        c.drawString(x, 88, label)
    round_card(c, 423, 62, 380, 92, fill=RED_LIGHT, stroke=RED)
    c.setFillColor(RED)
    c.setFont(FONT_SEMIBOLD, 8)
    c.drawString(440, 132, "DO NOT READ THIS ROW AS A TRADE")
    draw_text(c, "The five outputs are footprint descriptors, not returns or actions. Presence logits were inactive. A single trace demonstrates tensor semantics only.", 440, 118, 345, size=7.7, leading=10, color=INK)


def page_12(c: canvas.Canvas, d: dict[str, Any], metrics_hash: str, protocol_hash: str) -> None:
    page_frame(c, 12, "Economic boundary and next gate", "No Tiny profitability result; explicit requirements before fund use")
    round_card(c, 38, 355, 355, 140, fill=RED_LIGHT, stroke=RED)
    c.setFillColor(RED)
    c.setFont(FONT_BOLD, 22)
    c.drawString(55, 452, "TINY PnL: NOT MEASURED")
    draw_text(
        c,
        "No Tiny-specific return target, execution policy, fill simulation, fee/slippage model, turnover, Sharpe, drawdown, or capacity estimate exists. Forecast skill cannot be translated into return percent.",
        55,
        428,
        320,
        size=8.4,
        leading=11.5,
        color=INK,
    )

    context = d["historical_full_model_economic_context_not_tiny"]
    round_card(c, 413, 355, 390, 140, fill=WHITE)
    c.setFillColor(INK)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawString(430, 470, "Historical full-model diagnostic - not Tiny")
    rows = [
        ["Registered / post hoc trades", f"{context['registered_interval_policy_trades']} / {context['post_hoc_trades']}"],
        ["Post hoc gross / net", f"{context['post_hoc_median_sign_gross_return_percent']:+.2f}% / {context['post_hoc_median_sign_net_return_percent_at_12bp_one_way']:+.2f}%"],
        ["Gross edge / modeled round-trip cost", f"{context['gross_edge_bp_per_round_trip']:.3f} bp / {context['modeled_cost_bp_per_round_trip']:.1f} bp"],
    ]
    for index, (label, value) in enumerate(rows):
        yy = 439 - index * 27
        c.setFillColor(SLATE_DARK)
        c.setFont(FONT_REGULAR, 7.2)
        c.drawString(430, yy, label)
        c.setFillColor(RED if index in (1, 2) else INK)
        c.setFont(FONT_SEMIBOLD, 7.6)
        c.drawRightString(786, yy, value)
    c.setFillColor(SLATE)
    c.setFont(FONT_REGULAR, 6.5)
    c.drawString(430, 369, "Contextual warning only. It cannot be assigned to the Tiny checkpoint.")

    gates = [
        ("1", "DIRECT UTILITY", "Freeze executable return and spread-aware targets."),
        ("2", "STRONG BASELINES", "Freeze trees, quantile regression, and matched TCN."),
        ("3", "ROBUSTNESS", "Run 5 seeds and at least 5 purged rolling folds."),
        ("4", "EXECUTION", "Freeze bid/ask timing, fees, slippage, and impact."),
        ("5", "NEW EVIDENCE", "Freeze code, thresholds, and primary endpoints; then access 60+ new sessions."),
    ]
    for index, (number, title, body) in enumerate(gates):
        x = 38 + index * 153
        round_card(c, x, 190, 140, 125, fill=WHITE)
        c.setFillColor(BLUE if index < 3 else AMBER)
        c.circle(x + 23, 286, 12, fill=1, stroke=0)
        c.setFillColor(WHITE)
        c.setFont(FONT_BOLD, 8)
        c.drawCentredString(x + 23, 283, number)
        c.setFillColor(INK)
        c.setFont(FONT_SEMIBOLD, 7.3)
        c.drawString(x + 13, 261, title)
        draw_text(c, body, x + 13, 246, 114, size=6.9, leading=9.2, color=SLATE_DARK)

    round_card(c, 38, 62, 765, 90, fill=NAVY, stroke=NAVY)
    c.setFillColor(WHITE)
    c.setFont(FONT_SEMIBOLD, 8)
    c.drawString(55, 127, "PROVENANCE AND DECISION")
    c.setFillColor(HexColor("#B7C8EB"))
    c.setFont(FONT_MONO, 6.4)
    c.drawString(55, 108, f"metrics {metrics_hash[:20]}...  |  protocol {protocol_hash[:20]}...")
    c.drawString(55, 93, f"checkpoint {d['source_artifacts']['checkpoint_sha256'][:20]}...  |  result {d['source_artifacts']['result_json_sha256'][:20]}...")
    c.setFillColor(AMBER)
    c.setFont(FONT_SEMIBOLD, 9)
    c.drawRightString(786, 91, "RESEARCH CONTINUATION - NOT DEPLOYMENT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol-hash",
        type=Path,
        default=Path(__file__).with_name("protocol-july-august-2026.sha256"),
    )
    return parser.parse_args()


def build_report(metrics_path: Path, output_path: Path, protocol_hash_path: Path) -> None:
    register_fonts()
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics_hash = file_sha256(metrics_path)
    protocol_hash = protocol_hash_path.read_text(encoding="utf-8").split()[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(output_path), pagesize=(PAGE_W, PAGE_H), pageCompression=1)
    c.setTitle("CHMT-Q Tiny: Retrospective BTC-USDT Forecast Evaluation")
    c.setAuthor("Dhruv Sharma")
    c.setCreator("Dhruv Sharma")
    c.setSubject("Research-only technical evidence memo")
    pages = [
        lambda: cover_page(c, data),
        lambda: page_2(c, data),
        lambda: page_3(c, data),
        lambda: page_4(c, data),
        lambda: page_5(c, data),
        lambda: page_6(c, data),
        lambda: page_7(c, data),
        lambda: page_8(c, data),
        lambda: page_9(c, data),
        lambda: page_10(c, data),
        lambda: page_11(c, data),
        lambda: page_12(c, data, metrics_hash, protocol_hash),
    ]
    for draw in pages:
        draw()
        c.showPage()
    c.save()
    digest = file_sha256(output_path)
    output_path.with_suffix(".sha256").write_text(
        f"{digest}  {output_path.name}\n", encoding="ascii"
    )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": digest,
                "pages": len(pages),
                "metrics_sha256": metrics_hash,
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    build_report(args.metrics, args.output, args.protocol_hash)


if __name__ == "__main__":
    main()
