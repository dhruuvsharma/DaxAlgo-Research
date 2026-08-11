"""Render the artifact-bound FDT-N academic publication manuscript."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
from reportlab import rl_config
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .config import TARGET_NAMES
from .report import ReportBundle, load_report_bundle

plt.switch_backend("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False

INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#596579")
BLUE = colors.HexColor("#2457D6")
TEAL = colors.HexColor("#008F7A")
AMBER = colors.HexColor("#C77B00")
RED = colors.HexColor("#C53B53")
GRID = colors.HexColor("#D6DCE6")
PALE = colors.HexColor("#F4F7FB")
WHITE = colors.white

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"
FONT_ITALIC = "Helvetica-Oblique"
FONT_MONO = "Courier"

TARGET_LABELS = {
    "poc_offset": "POC",
    "low_offset": "Low",
    "high_offset": "High",
    "buy_poc_offset": "Buy POC",
    "sell_poc_offset": "Sell POC",
    "log_volume_relative": "Log volume",
    "delta_fraction": "Delta",
}

REQUIRED_FIGURES = {
    "data_split",
    "architecture",
    "target_mae",
    "horizon_mae",
    "coverage",
    "bootstrap",
}
FIGURE_PATTERN = re.compile(r"^\{\{FIGURE:([a-z0-9_]+)\|(.+)}}$")
HEX_PATTERN = re.compile(r"\b[0-9a-f]{48,64}\b", re.IGNORECASE)
LINK_PATTERN = re.compile(r"\[([^]]+)]\((https?://[^)]+)\)")


def generate_publication_pdf(
    results_paths: Sequence[Path],
    dataset_manifest_path: Path,
    dataset_path: Path,
    manuscript_path: Path,
    evidence_ledger_path: Path,
    model_metadata_path: Path,
    output_path: Path,
    workdir: Path,
    *,
    enforce_repo_paths: bool = True,
) -> dict[str, Any]:
    """Validate evidence, render the academic paper, and publish atomically."""

    bundle = load_report_bundle(
        results_paths,
        dataset_manifest_path,
        dataset_path,
        manuscript_path.parent / "research-paper.md",
        evidence_ledger_path,
        model_metadata_path,
    )
    metadata, manuscript = parse_front_matter(manuscript_path.read_text(encoding="utf-8"))
    _validate_manuscript(metadata, manuscript)

    output = output_path.resolve()
    figure_dir = workdir.resolve()
    repo_root = _find_repo_root(manuscript_path)
    if enforce_repo_paths:
        _require_descendant(output, repo_root / "output" / "pdf", "PDF output")
        _require_descendant(figure_dir, repo_root / "tmp" / "pdfs", "PDF work directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing publication paper: {output}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    _register_fonts()
    figures = _build_figures(bundle, figure_dir)
    staged = figure_dir / "fdt-n-publication.staged.pdf"
    if staged.exists():
        staged.unlink()
    styles = _styles()
    story = _title_flowables(metadata, styles)
    story.extend(_render_manuscript(manuscript, figures, figure_dir, styles, metadata["title"]))

    doc = PublicationDocTemplate(
        str(staged),
        title=metadata["title"],
        author=metadata["author"],
        short_title=metadata["short_title"],
    )
    try:
        doc.build(story)
        reader = PdfReader(str(staged))
        if not 10 <= len(reader.pages) <= 40:
            raise RuntimeError(f"unexpected publication page count: {len(reader.pages)}")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if any(token in text for token in ("{{FIGURE:", "[PENDING]", "PLACEHOLDER")):
            raise RuntimeError("publication PDF contains an unresolved token")
        normalized_text = " ".join(text.split())
        required_text = (
            "Do Ragged Volume-Footprint Transformers Beat Aggregate Linear Models?",
            "2.68%",
            "direct ridge",
            "No result in this paper supports deployment",
            "Anonymous author(s)",
        )
        missing = [value for value in required_text if value not in normalized_text]
        if missing:
            raise RuntimeError(f"publication PDF is missing required text: {missing}")
        shutil.copyfile(staged, output)
    finally:
        if staged.exists():
            staged.unlink()

    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "pages": len(reader.pages),
        "primary_seed": bundle.primary.seed,
        "manuscript_sha256": _sha256(manuscript_path),
        "model_metadata_sha256": bundle.deployment.metadata.sha256,
        "onnx_sha256": bundle.deployment.onnx.sha256,
        "figures": {name: str(path) for name, path in figures.items()},
    }


class PublicationDocTemplate(BaseDocTemplate):
    """A4 academic template with stable headers, footers, and metadata."""

    def __init__(self, filename: str, *, title: str, author: str, short_title: str) -> None:
        self.paper_title = title
        self.paper_author = author
        self.short_title = short_title
        margin_x = 19 * mm
        margin_top = 18 * mm
        margin_bottom = 18 * mm
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=margin_top,
            bottomMargin=margin_bottom,
            title=title,
            author=author,
            subject="Volume-footprint quantile forecasting research manuscript",
            creator="FDT-N artifact-bound publication renderer",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="body",
        )
        self.addPageTemplates(
            PageTemplate(id="academic", frames=[frame], onPage=self._decorate_page)
        )

    def _decorate_page(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setTitle(self.paper_title)
        canvas.setAuthor(self.paper_author)
        width, height = A4
        if doc.page > 1:
            canvas.setStrokeColor(GRID)
            canvas.setLineWidth(0.45)
            canvas.line(
                self.leftMargin, height - 12.5 * mm, width - self.rightMargin, height - 12.5 * mm
            )
            canvas.setFont(FONT_REGULAR, 7.2)
            canvas.setFillColor(MUTED)
            canvas.drawString(self.leftMargin, height - 10 * mm, self.short_title)
            canvas.drawRightString(
                width - self.rightMargin, height - 10 * mm, "Anonymous review manuscript"
            )
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.45)
        canvas.line(self.leftMargin, 12.5 * mm, width - self.rightMargin, 12.5 * mm)
        canvas.setFont(FONT_REGULAR, 7.2)
        canvas.setFillColor(MUTED)
        canvas.drawString(self.leftMargin, 9 * mm, "FDT-N | artifact-bound development study")
        canvas.drawRightString(width - self.rightMargin, 9 * mm, str(doc.page))
        canvas.restoreState()


def parse_front_matter(source: str) -> tuple[dict[str, str], str]:
    """Parse the small YAML-like metadata block used by the manuscript."""

    normalized = source.replace("\r\n", "\n")
    if not normalized.startswith("---\n"):
        raise ValueError("publication manuscript must start with front matter")
    end = normalized.find("\n---\n", 4)
    if end < 0:
        raise ValueError("publication manuscript front matter is not closed")
    metadata: dict[str, str] = {}
    for line in normalized[4:end].splitlines():
        if not line.strip():
            continue
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid front-matter line: {line}")
        metadata[key.strip()] = value.strip()
    return metadata, normalized[end + 5 :].strip() + "\n"


def _validate_manuscript(metadata: dict[str, str], manuscript: str) -> None:
    required = {"title", "short_title", "author", "affiliation", "date", "keywords"}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"publication front matter is missing: {missing}")
    if metadata["author"] != "Anonymous author(s)":
        raise ValueError(
            "author identity must not be fabricated; use the anonymous review identity"
        )
    forbidden = ("[PENDING]", "PLACEHOLDER", "TODO", "TBD")
    hits = [token for token in forbidden if token in manuscript]
    if hits:
        raise ValueError(f"publication manuscript contains unfinished tokens: {hits}")
    directives = set(re.findall(r"\{\{FIGURE:([a-z0-9_]+)\|", manuscript))
    if directives != REQUIRED_FIGURES:
        raise ValueError(
            f"publication figure directives differ: expected {sorted(REQUIRED_FIGURES)}, "
            f"found {sorted(directives)}"
        )
    required_phrases = (
        "2.68% above a train-only direct ridge model",
        "does not outperform the train-only direct ridge model",
        "not as calibrated quantiles",
        "does not define a coherent joint candle distribution",
        "No result in this paper supports deployment",
    )
    normalized_claims = " ".join(manuscript.split())
    absent = [phrase for phrase in required_phrases if phrase not in normalized_claims]
    if absent:
        raise ValueError(f"publication manuscript is missing guarded claims: {absent}")


def _styles() -> dict[str, ParagraphStyle]:
    return {
        "title": ParagraphStyle(
            "PublicationTitle",
            fontName=FONT_BOLD,
            fontSize=20,
            leading=23,
            textColor=INK,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "author": ParagraphStyle(
            "PublicationAuthor",
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=13,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "metadata": ParagraphStyle(
            "PublicationMetadata",
            fontName=FONT_REGULAR,
            fontSize=7.8,
            leading=10.2,
            textColor=MUTED,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "PublicationH1",
            fontName=FONT_BOLD,
            fontSize=14,
            leading=17,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "PublicationH2",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=13,
            textColor=BLUE,
            spaceBefore=9,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "PublicationBody",
            fontName=FONT_REGULAR,
            fontSize=9.05,
            leading=12.2,
            textColor=INK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            splitLongWords=True,
            allowWidows=0,
            allowOrphans=0,
        ),
        "abstract": ParagraphStyle(
            "PublicationAbstract",
            fontName=FONT_REGULAR,
            fontSize=8.7,
            leading=11.7,
            textColor=INK,
            alignment=TA_JUSTIFY,
            leftIndent=7 * mm,
            rightIndent=7 * mm,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "PublicationCaption",
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=9.5,
            textColor=MUTED,
            alignment=TA_LEFT,
            spaceBefore=3,
            spaceAfter=9,
        ),
        "table": ParagraphStyle(
            "PublicationTableCell",
            fontName=FONT_REGULAR,
            fontSize=6.65,
            leading=8.2,
            textColor=INK,
            alignment=TA_LEFT,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "PublicationTableHeader",
            fontName=FONT_BOLD,
            fontSize=6.65,
            leading=8.2,
            textColor=WHITE,
            alignment=TA_LEFT,
            splitLongWords=True,
            wordWrap="CJK",
        ),
        "reference": ParagraphStyle(
            "PublicationReference",
            parent=None,
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=9.6,
            textColor=INK,
            leftIndent=5 * mm,
            firstLineIndent=-5 * mm,
            spaceAfter=4,
            splitLongWords=True,
        ),
        "quote": ParagraphStyle(
            "PublicationQuote",
            fontName=FONT_ITALIC,
            fontSize=8.4,
            leading=11.2,
            textColor=INK,
            leftIndent=6 * mm,
            rightIndent=6 * mm,
            borderColor=BLUE,
            borderWidth=1,
            borderPadding=6,
            backColor=PALE,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "list": ParagraphStyle(
            "PublicationList",
            fontName=FONT_REGULAR,
            fontSize=8.9,
            leading=11.8,
            textColor=INK,
            alignment=TA_JUSTIFY,
        ),
    }


def _title_flowables(metadata: dict[str, str], styles: dict[str, ParagraphStyle]) -> list[Any]:
    return [
        Spacer(1, 5 * mm),
        Paragraph(_inline(metadata["title"]), styles["title"]),
        Paragraph(_inline(metadata["author"]), styles["author"]),
        Paragraph(_inline(metadata["affiliation"]), styles["author"]),
        Spacer(1, 2 * mm),
        HRFlowable(width="100%", thickness=1.2, color=BLUE, spaceBefore=2, spaceAfter=7),
        Paragraph(
            f"<b>Date:</b> {_inline(metadata['date'])}<br/>"
            f"<b>Keywords:</b> {_inline(metadata['keywords'])}",
            styles["metadata"],
        ),
    ]


def _render_manuscript(
    manuscript: str,
    figures: dict[str, Path],
    workdir: Path,
    styles: dict[str, ParagraphStyle],
    title: str,
) -> list[Any]:
    lines = manuscript.splitlines()
    flowables: list[Any] = []
    paragraph: list[str] = []
    index = 0
    first_heading_skipped = False
    abstract_mode = False

    def flush_paragraph() -> None:
        nonlocal paragraph
        if not paragraph:
            return
        text = " ".join(value.strip() for value in paragraph).strip()
        if text:
            style = styles["abstract"] if abstract_mode else styles["body"]
            if text.startswith("[") and re.match(r"^\[\d+]", text):
                style = styles["reference"]
            flowables.append(Paragraph(_inline(text), style))
        paragraph = []

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("# "):
            flush_paragraph()
            heading = stripped[2:].strip()
            if not first_heading_skipped and heading == title:
                first_heading_skipped = True
            else:
                flowables.append(Paragraph(_inline(heading), styles["h1"]))
            index += 1
            continue
        if stripped.startswith("## "):
            flush_paragraph()
            heading = stripped[3:].strip()
            abstract_mode = heading == "Abstract"
            flowables.append(Paragraph(_inline(heading), styles["h1"]))
            index += 1
            continue
        if stripped.startswith("### "):
            flush_paragraph()
            abstract_mode = False
            flowables.append(Paragraph(_inline(stripped[4:].strip()), styles["h2"]))
            index += 1
            continue
        if stripped == "$$":
            flush_paragraph()
            equation_lines: list[str] = []
            index += 1
            while index < len(lines) and lines[index].strip() != "$$":
                equation_lines.append(lines[index].strip())
                index += 1
            if index >= len(lines):
                raise ValueError("unclosed display equation")
            equation = " ".join(equation_lines)
            path = _equation_image(equation, workdir)
            flowables.extend(
                [
                    Image(str(path), width=165 * mm, height=14 * mm, kind="proportional"),
                    Spacer(1, 3),
                ]
            )
            index += 1
            continue
        figure_match = FIGURE_PATTERN.match(stripped)
        if figure_match:
            flush_paragraph()
            name, caption = figure_match.groups()
            image_path = figures.get(name)
            if image_path is None:
                raise ValueError(f"unknown publication figure: {name}")
            image = Image(str(image_path), width=170 * mm, height=80 * mm, kind="proportional")
            flowables.append(
                KeepTogether(
                    [Spacer(1, 2), image, Paragraph(_inline(caption), styles["caption"])]
                )
            )
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            flowables.extend(_markdown_table(table_lines, styles))
            continue
        if stripped.startswith("- ") or re.match(r"^\d+\.\s", stripped):
            flush_paragraph()
            ordered = bool(re.match(r"^\d+\.\s", stripped))
            items: list[str] = []
            while index < len(lines):
                candidate = lines[index].strip()
                pattern = r"^\d+\.\s+(.+)" if ordered else r"^-\s+(.+)"
                match = re.match(pattern, candidate)
                if not match:
                    break
                index += 1
                item_parts = [match.group(1)]
                while index < len(lines):
                    continuation = lines[index]
                    continuation_text = continuation.strip()
                    if not continuation_text or not continuation[:1].isspace():
                        break
                    item_parts.append(continuation_text)
                    index += 1
                items.append(" ".join(item_parts))
            flowables.append(
                ListFlowable(
                    [
                        ListItem(Paragraph(_inline(item), styles["list"]), leftIndent=9)
                        for item in items
                    ],
                    bulletType="1" if ordered else "bullet",
                    start="1",
                    leftIndent=15,
                    bulletFontName=FONT_REGULAR,
                    bulletFontSize=8,
                    spaceAfter=6,
                )
            )
            continue
        if stripped.startswith("> "):
            flush_paragraph()
            quote_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("> "):
                quote_lines.append(lines[index].strip()[2:])
                index += 1
            flowables.append(Paragraph(_inline(" ".join(quote_lines)), styles["quote"]))
            continue
        if stripped == "{{PAGEBREAK}}":
            flush_paragraph()
            flowables.append(PageBreak())
            index += 1
            continue
        if stripped == "---":
            flush_paragraph()
            flowables.append(
                HRFlowable(width="100%", thickness=0.6, color=GRID, spaceBefore=4, spaceAfter=6)
            )
            index += 1
            continue
        paragraph.append(stripped)
        index += 1
    flush_paragraph()
    return flowables


def _markdown_table(lines: list[str], styles: dict[str, ParagraphStyle]) -> list[Any]:
    if len(lines) < 2:
        raise ValueError("publication table requires a header and separator")
    rows = [_split_table_line(line) for line in lines]
    if not all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in rows[1]):
        raise ValueError("publication table separator is invalid")
    data = [rows[0], *rows[2:]]
    column_count = len(data[0])
    if any(len(row) != column_count for row in data):
        raise ValueError("publication table has inconsistent columns")
    weights = []
    for column in range(column_count):
        longest = max(len(row[column]) for row in data)
        weights.append(max(7.0, min(float(longest), 30.0)))
    total = sum(weights)
    widths = [170 * mm * weight / total for weight in weights]
    rendered = []
    for row_index, row in enumerate(data):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        rendered.append([Paragraph(_break_hash(_inline(value)), style) for value in row])
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT", splitByRow=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#172746")),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), FONT_REGULAR),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return [Spacer(1, 3), table, Spacer(1, 8)]


def _split_table_line(line: str) -> list[str]:
    return [value.strip() for value in line.strip().strip("|").split("|")]


def _inline(value: str) -> str:
    escaped = html.escape(value, quote=False)
    escaped = LINK_PATTERN.sub(
        lambda match: f'<link href="{match.group(2)}" color="#2457D6">{match.group(1)}</link>',
        escaped,
    )
    escaped = re.sub(r"`([^`]+)`", rf'<font name="{FONT_MONO}">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _break_hash(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        return "<br/>".join(token[index : index + 16] for index in range(0, len(token), 16))

    return HEX_PATTERN.sub(replace, value)


def _equation_image(equation: str, workdir: Path) -> Path:
    digest = hashlib.sha256(equation.encode("utf-8")).hexdigest()[:16]
    path = workdir / f"equation-{digest}.png"
    if path.exists():
        return path
    fig = plt.figure(figsize=(7.0, 0.62), facecolor="white")
    fig.text(0.5, 0.5, f"${equation}$", ha="center", va="center", fontsize=13, color="#172033")
    fig.savefig(path, dpi=260, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def _build_figures(bundle: ReportBundle, workdir: Path) -> dict[str, Path]:
    paths = {
        name: workdir / f"publication-{name.replace('_', '-')}.png" for name in REQUIRED_FIGURES
    }
    _plot_data_split(bundle, paths["data_split"])
    _plot_architecture(paths["architecture"])
    _plot_target_mae(bundle, paths["target_mae"])
    _plot_horizon_mae(bundle, paths["horizon_mae"])
    _plot_coverage(bundle, paths["coverage"])
    _plot_bootstrap(bundle, paths["bootstrap"])
    return paths


def _plot_data_split(bundle: ReportBundle, path: Path) -> None:
    _plot_style()
    counts = bundle.primary.data["splits"]
    labels = ["Train", "Validation", "Test"]
    values = [counts["train_examples"], counts["validation_examples"], counts["test_examples"]]
    colors_list = ["#2457D6", "#008F7A", "#C77B00"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.0, 1.45]})
    axes[0].bar(labels, values, color=colors_list, width=0.66)
    axes[0].set_ylabel("Valid windows")
    axes[0].set_title("Purged chronological counts", loc="left", fontweight="bold")
    for index, value in enumerate(values):
        axes[0].text(index, value + max(values) * 0.025, f"{value:,}", ha="center", fontsize=9)
    axes[0].set_ylim(0, max(values) * 1.16)
    axes[0].spines[["top", "right"]].set_visible(False)

    axes[1].set_xlim(0, sum(values) + 144)
    axes[1].set_ylim(0, 1)
    cursor = 0
    for label, value, color in zip(labels, values, colors_list, strict=True):
        axes[1].barh(0.55, value, left=cursor, height=0.24, color=color)
        axes[1].text(
            cursor + value / 2,
            0.55,
            label,
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=8,
        )
        cursor += value
        if label != "Test":
            axes[1].barh(0.55, 72, left=cursor, height=0.24, color="#D6DCE6")
            axes[1].text(
                cursor + 36, 0.30, "72-bar purge", ha="center", fontsize=7, color="#596579"
            )
            cursor += 72
    axes[1].set_title("Causal order and boundary purges", loc="left", fontweight="bold")
    axes[1].axis("off")
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_architecture(path: Path) -> None:
    _plot_style()
    fig, axis = plt.subplots(figsize=(11.0, 4.2))
    axis.set_xlim(0, 11)
    axis.set_ylim(0, 4)
    axis.axis("off")
    boxes = [
        (0.2, 2.25, 1.25, 0.72, "11 row\nfeatures", "#E8EEFF", "#2457D6"),
        (1.75, 2.25, 1.25, 0.72, "Row MLP\nper price", "#E8EEFF", "#2457D6"),
        (3.30, 2.25, 1.35, 0.72, "4 learned\nrow queries", "#E4F6F2", "#008F7A"),
        (3.30, 0.85, 1.35, 0.72, "23 causal\nbar features", "#FFF1D8", "#C77B00"),
        (5.00, 1.55, 1.35, 0.72, "Gated row /\nbar fusion", "#F0EBFF", "#6E52D9"),
        (6.75, 1.55, 1.35, 0.72, "3 causal\ntemporal blocks", "#E8EEFF", "#2457D6"),
        (8.45, 1.55, 1.15, 0.72, "8 horizon\nqueries", "#E4F6F2", "#008F7A"),
        (9.90, 1.55, 0.90, 0.72, "7 x 3\noutputs", "#FFE8EC", "#C53B53"),
    ]
    for x, y, width, height, label, face, edge in boxes:
        patch = matplotlib.patches.FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=face,
            edgecolor=edge,
            linewidth=1.4,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#172033",
        )
    arrows = [
        ((1.45, 2.61), (1.75, 2.61)),
        ((3.00, 2.61), (3.30, 2.61)),
        ((4.65, 2.61), (5.00, 2.05)),
        ((4.65, 1.21), (5.00, 1.77)),
        ((6.35, 1.91), (6.75, 1.91)),
        ((8.10, 1.91), (8.45, 1.91)),
        ((9.60, 1.91), (9.90, 1.91)),
    ]
    for start, end in arrows:
        axis.annotate(
            "", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#596579", "lw": 1.5}
        )
    axis.text(
        0.2, 3.55, "Ragged row set per sealed bar", fontsize=10, fontweight="bold", color="#2457D6"
    )
    axis.text(5.0, 3.55, "64-bar causal history", fontsize=10, fontweight="bold", color="#6E52D9")
    axis.text(
        8.45,
        3.55,
        "Direct next-8 marginal quantiles",
        fontsize=10,
        fontweight="bold",
        color="#008F7A",
    )
    axis.text(9.95, 0.75, "q10 / q50 / q90", fontsize=8, color="#596579")
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_target_mae(bundle: ReportBundle, path: Path) -> None:
    _plot_style()
    run = bundle.primary.data
    model = run["test"]["median_mae_normalized"]
    persistence = run["baselines"]["persistence"]["test"]["median_mae_normalized"]
    ridge = run["baselines"]["direct_ridge"]["test"]["median_mae_normalized"]
    labels = [TARGET_LABELS[name] for name in TARGET_NAMES]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(10.2, 4.4))
    axis.bar(
        x - width, [model[name] for name in TARGET_NAMES], width, label="FDT-N", color="#2457D6"
    )
    axis.bar(
        x, [persistence[name] for name in TARGET_NAMES], width, label="Persistence", color="#9AA5B5"
    )
    axis.bar(
        x + width,
        [ridge[name] for name in TARGET_NAMES],
        width,
        label="Direct ridge",
        color="#C77B00",
    )
    axis.set_xticks(x, labels, rotation=22, ha="right")
    axis.set_ylabel("Normalized q50-forecast MAE")
    axis.set_title(
        "Selected seed 17: point-forecast comparison by target", loc="left", fontweight="bold"
    )
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.grid(axis="y", color="#D6DCE6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_horizon_mae(bundle: ReportBundle, path: Path) -> None:
    _plot_style()
    run = bundle.primary.data
    horizons = np.arange(1, 9)

    def horizon_values(section: dict[str, Any]) -> list[float]:
        output = []
        for horizon in horizons:
            cells = [section["by_horizon_target"][str(horizon)][name] for name in TARGET_NAMES]
            weights = np.asarray([cell["examples"] for cell in cells], dtype=np.float64)
            output.append(
                float(
                    np.average([cell["median_mae_normalized"] for cell in cells], weights=weights)
                )
            )
        return output

    model = horizon_values(run["test"])
    persistence = horizon_values(run["baselines"]["persistence"]["test"])
    ridge = horizon_values(run["baselines"]["direct_ridge"]["test"])
    fig, axis = plt.subplots(figsize=(10.2, 4.25))
    axis.plot(horizons, model, marker="o", linewidth=2.2, color="#2457D6", label="FDT-N")
    axis.plot(
        horizons, persistence, marker="o", linewidth=1.8, color="#9AA5B5", label="Persistence"
    )
    axis.plot(horizons, ridge, marker="o", linewidth=1.8, color="#C77B00", label="Direct ridge")
    axis.set_xticks(horizons)
    axis.set_xlabel("Forecast horizon (one-minute bars)")
    axis.set_ylabel("Normalized q50-forecast MAE")
    axis.set_title("Point-forecast error by horizon", loc="left", fontweight="bold")
    axis.legend(frameon=False, ncol=3, loc="upper left")
    axis.grid(color="#D6DCE6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_coverage(bundle: ReportBundle, path: Path) -> None:
    _plot_style()
    section = bundle.primary.data["test"]["by_horizon_target"]
    matrix = np.asarray(
        [
            [section[str(horizon)][name]["central_80_coverage"] for name in TARGET_NAMES]
            for horizon in range(1, 9)
        ],
        dtype=np.float64,
    )
    fig, axis = plt.subplots(figsize=(10.2, 4.7))
    image = axis.imshow(matrix, cmap="RdYlBu", vmin=0.72, vmax=0.88, aspect="auto")
    axis.set_xticks(
        np.arange(len(TARGET_NAMES)),
        [TARGET_LABELS[name] for name in TARGET_NAMES],
        rotation=25,
        ha="right",
    )
    axis.set_yticks(np.arange(8), [f"h{value}" for value in range(1, 9)])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{100 * matrix[row, column]:.1f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="#172033",
            )
    colorbar = fig.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Empirical central-80% coverage")
    axis.set_title(
        "Selected seed 17: marginal central-interval coverage", loc="left", fontweight="bold"
    )
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_bootstrap(bundle: ReportBundle, path: Path) -> None:
    _plot_style()
    rows = []
    for run in sorted(bundle.runs, key=lambda item: item.seed):
        for key, label in (
            ("model_vs_persistence", "Persistence"),
            ("model_vs_direct_ridge", "Direct ridge"),
        ):
            cells = [
                cell
                for horizon in run.data["comparisons"][key]["by_horizon_target"].values()
                for cell in horizon.values()
            ]
            rows.append(
                (
                    f"Seed {run.seed} vs {label}",
                    sum(cell["ci_95_upper"] < 0.0 for cell in cells),
                    sum(cell["ci_95_lower"] <= 0.0 <= cell["ci_95_upper"] for cell in cells),
                    sum(cell["ci_95_lower"] > 0.0 for cell in cells),
                )
            )
    labels = [row[0] for row in rows]
    favorable = np.asarray([row[1] for row in rows])
    crossing = np.asarray([row[2] for row in rows])
    unfavorable = np.asarray([row[3] for row in rows])
    y = np.arange(len(rows))
    fig, axis = plt.subplots(figsize=(10.2, 4.55))
    axis.barh(y, favorable, color="#008F7A", label="Favors FDT-N")
    axis.barh(y, crossing, left=favorable, color="#D6DCE6", label="Spans zero")
    axis.barh(y, unfavorable, left=favorable + crossing, color="#C53B53", label="Favors comparator")
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 56)
    axis.set_xlabel("Horizon-target cells (56 total)")
    axis.set_title(
        "Paired four-hour block-resampling sensitivity summaries", loc="left", fontweight="bold"
    )
    axis.legend(frameon=False, ncol=3, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#D6DCE6", linewidth=0.7)
    fig.tight_layout()
    _save_figure(fig, path)


def _plot_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelcolor": "#172033",
            "axes.titlecolor": "#172033",
            "xtick.color": "#596579",
            "ytick.color": "#596579",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _register_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD, FONT_ITALIC, FONT_MONO
    fonts = {
        "SegoeUI": Path(r"C:\Windows\Fonts\segoeui.ttf"),
        "SegoeUI-Semibold": Path(r"C:\Windows\Fonts\seguisb.ttf"),
        "SegoeUI-Italic": Path(r"C:\Windows\Fonts\segoeuii.ttf"),
        "Consolas": Path(r"C:\Windows\Fonts\consola.ttf"),
    }
    if all(path.is_file() for path in fonts.values()):
        for name, path in fonts.items():
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, path))
        FONT_REGULAR = "SegoeUI"
        FONT_BOLD = "SegoeUI-Semibold"
        FONT_ITALIC = "SegoeUI-Italic"
        FONT_MONO = "Consolas"
        rl_config.canvas_basefontname = FONT_REGULAR


def _find_repo_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / ".git").exists() and (candidate / "public").exists():
            return candidate
    raise ValueError(f"could not identify repository root from {path}")


def _require_descendant(path: Path, directory: Path, label: str) -> None:
    try:
        path.relative_to(directory.resolve())
    except ValueError as exception:
        raise ValueError(f"{label} must be under {directory.resolve()}: {path}") from exception


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
