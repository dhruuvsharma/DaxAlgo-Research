"""Render the CHMT-Q failure analysis and Transformer-DQN revision plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

PAGE_W, PAGE_H = landscape(A4)

NAVY = "#091426"
INK = "#172033"
MUTED = "#667085"
GRID = "#D8DEE8"
PAPER = "#F4F6F9"
WHITE = "#FFFFFF"
BLUE = "#2F6BFF"
TEAL = "#00A88F"
RED = "#D9485F"
AMBER = "#E69A25"
GRAY = "#98A2B3"
LIGHT_BLUE = "#EAF0FF"
LIGHT_TEAL = "#E5F7F3"
LIGHT_RED = "#FCECEF"
LIGHT_AMBER = "#FFF4DD"
LIGHT_GRAY = "#EEF1F5"

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

TARGET_SHORT = ("Total POC", "Buy POC", "Sell POC", "Log volume", "Delta")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    parser.add_argument("--dqn-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "diagnostics": args.diagnostics.resolve(strict=True),
        "results": args.results.resolve(strict=True),
        "protocol": args.protocol.resolve(strict=True),
        "supplement": args.supplement.resolve(strict=True),
        "dqn_plan": args.dqn_plan.resolve(strict=True),
    }
    output = args.output.resolve()
    workdir = args.workdir.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    diagnostics = _read_json(paths["diagnostics"])
    results = _read_json(paths["results"])
    protocol = _read_json(paths["protocol"])
    supplement = _read_json(paths["supplement"])
    dqn_plan = _read_json(paths["dqn_plan"])
    _validate_sources(diagnostics, results, protocol, supplement, dqn_plan, paths)
    _register_fonts()
    charts = _build_charts(diagnostics, results, supplement, workdir)
    report = RevisionReport(
        output=output,
        diagnostics=diagnostics,
        results=results,
        protocol=protocol,
        supplement=supplement,
        dqn_plan=dqn_plan,
        charts=charts,
    )
    report.render()
    print(
        json.dumps(
            {
                "output": str(output),
                "bytes": output.stat().st_size,
                "sha256": _sha256(output),
                "pages": report.page_number,
                "charts": len(charts),
                "plan_id": dqn_plan["plan_id"],
                "plan_sha256": _sha256(paths["dqn_plan"]),
                "holdout_integrity_status": diagnostics[
                    "current_audit_holdout_integrity"
                ]["status"],
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


class RevisionReport:
    def __init__(
        self,
        *,
        output: Path,
        diagnostics: dict[str, Any],
        results: dict[str, Any],
        protocol: dict[str, Any],
        supplement: dict[str, Any],
        dqn_plan: dict[str, Any],
        charts: dict[str, Path],
    ) -> None:
        self.output = output
        self.diagnostics = diagnostics
        self.results = results
        self.protocol = protocol
        self.supplement = supplement
        self.dqn_plan = dqn_plan
        self.charts = charts
        self.canvas = canvas.Canvas(
            str(output), pagesize=(PAGE_W, PAGE_H), pageCompression=1
        )
        self.canvas.setTitle("CHMT-Q Transformer-DQN Architecture Revision")
        self.canvas.setAuthor("DaxAlgo Research")
        self.canvas.setSubject(
            "Evidence-backed diagnosis and gated Transformer-DQN research plan"
        )
        self.page_number = 0

    @property
    def forward(self) -> dict[str, Any]:
        return self.diagnostics["partition_metrics"][
            "untouched_development_forward"
        ]

    @property
    def economics(self) -> dict[str, Any]:
        return self.diagnostics["economic_diagnostics"]

    @property
    def training(self) -> dict[str, Any]:
        return self.diagnostics["training_diagnostics"]

    def render(self) -> None:
        self._cover()
        self._failure_chain()
        self._objective_and_scoring()
        self._forecast_and_calibration()
        self._training_and_optimization()
        self._data_and_power()
        self._economics()
        self._revision_architecture()
        self._dqn_environment()
        self._dqn_training_plan()
        self._experiment_program()
        self._audit_and_decision()
        self.canvas.save()

    def _new_page(
        self,
        title: str,
        subtitle: str,
        *,
        kicker: str,
        dark: bool = False,
    ) -> None:
        if self.page_number:
            self.canvas.showPage()
        self.page_number += 1
        self.canvas.setFillColor(HexColor(NAVY if dark else PAPER))
        self.canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(TEAL if dark else BLUE))
        self.canvas.setFont(FONT_BOLD, 8)
        self.canvas.drawString(38, PAGE_H - 34, kicker)
        self.canvas.setFillColor(HexColor(WHITE if dark else INK))
        self.canvas.setFont(FONT_BOLD, 23)
        self.canvas.drawString(38, PAGE_H - 65, title)
        self.canvas.setFillColor(HexColor("#C7D2E1" if dark else MUTED))
        self.canvas.setFont(FONT_REGULAR, 9.5)
        self.canvas.drawString(38, PAGE_H - 84, subtitle)
        if not dark:
            self._footer()

    def _footer(self) -> None:
        self.canvas.setStrokeColor(HexColor(GRID))
        self.canvas.setLineWidth(0.5)
        self.canvas.line(38, 27, PAGE_W - 38, 27)
        self.canvas.setFillColor(HexColor(MUTED))
        self.canvas.setFont(FONT_REGULAR, 7.2)
        self.canvas.drawString(
            38,
            14,
            "Architecture plan | Not implemented or validated | No deployment or capital | Jun 25-30 excluded",
        )
        self.canvas.drawRightString(PAGE_W - 38, 14, str(self.page_number))

    def _cover(self) -> None:
        econ = self.economics
        scoring = self.diagnostics["scoring_diagnostics"][
            "untouched_development_forward"
        ]
        self._new_page(
            "CHMT-Q revision: signal first, policy second",
            "Failure evidence and gated Transformer-DQN architecture | OKX BTC-USDT | July 2026",
            kicker="DAXALGO RESEARCH // ARCHITECTURE REVISION",
            dark=True,
        )
        self.canvas.setFillColor(HexColor("#93A4BA"))
        self.canvas.setFont(FONT_REGULAR, 8.5)
        self.canvas.drawString(38, PAGE_H - 115, "DECISION")
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setFont(FONT_BOLD, 18)
        self.canvas.drawString(38, PAGE_H - 141, "DO NOT DEPLOY. DO NOT ALLOCATE CAPITAL.")
        self.canvas.setFillColor(HexColor("#C7D2E1"))
        self.canvas.setFont(FONT_REGULAR, 9.5)
        self.canvas.drawString(
            38,
            PAGE_H - 160,
            "The old checkpoint is not tradable. DQN is proposed only after supervised return-signal gates pass.",
        )

        h5_mae = scoring["cell_median_mae_improvement_percent"][4][0]
        metrics = [
            (
                f"+{scoring['weighted_normalized_pinball']['model_improvement_percent']:.2f}%",
                "aggregate pinball",
                "vs published marginal baseline",
                BLUE,
            ),
            (
                f"{h5_mae:+.2f}%",
                "H5 POC median MAE",
                "effectively tied with baseline",
                AMBER,
            ),
            (
                str(econ["interval_policy"]["trade_count"]),
                "primary-policy trades",
                "all intervals crossed zero",
                RED,
            ),
            (
                f"{100 * econ['median_long_flat']['base_cost_compounded_return']:.1f}%",
                "base-cost return",
                "median long/flat proxy",
                RED,
            ),
        ]
        x0 = 38
        gap = 12
        width = (PAGE_W - 76 - 3 * gap) / 4
        for index, (value, label, note, color) in enumerate(metrics):
            self._metric_card(
                x=x0 + index * (width + gap),
                y=226,
                w=width,
                h=102,
                value=value,
                label=label,
                note=note,
                color=color,
                dark=True,
            )

        self._rounded_rect(38, 56, PAGE_W - 76, 138, "#111E33", "#263650", 10)
        self.canvas.setFillColor(HexColor(TEAL))
        self.canvas.setFont(FONT_BOLD, 8)
        self.canvas.drawString(54, 172, "WHAT FAILED")
        failures = [
            ("Objective", "Five footprint distributions, not return, price, action, or net PnL."),
            ("Signal", "H5 close-direction MCC 0.0475; POC Spearman 0.078."),
            ("Economics", "0.45 bps gross round-trip edge versus 24 bps modeled cost."),
            ("Evidence", "18 train days, 2 selection days, 4 forward days, one seed and fold."),
        ]
        col_w = (PAGE_W - 108) / 4
        for index, (label, text) in enumerate(failures):
            x = 54 + index * col_w
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 10)
            self.canvas.drawString(x, 148, label)
            self._wrapped_text(
                text,
                x,
                132,
                col_w - 18,
                FONT_REGULAR,
                8.2,
                11,
                "#C7D2E1",
                max_lines=4,
            )
        self.canvas.setFillColor(HexColor("#8EA0B7"))
        self.canvas.setFont(FONT_REGULAR, 7.3)
        self.canvas.drawString(
            38,
            26,
            "Revision status: diagnosis complete; replacement model is not yet trained or validated.",
        )

    def _failure_chain(self) -> None:
        self._new_page(
            "The failure is a chain, not one bad hyperparameter",
            "Every link is observed in the authorized Jun 1-24 evidence; causal strength is labeled explicitly.",
            kicker="01 // ROOT-CAUSE MAP",
        )
        boxes = [
            ("1  WRONG ENDPOINT", "Footprint pinball", "No direct return target", RED),
            ("2  WEAK MEDIAN", "H5 POC rho 0.075", "Std ratio 0.151", AMBER),
            ("3  WIDE RISK", "$190.47 interval", "100% crosses zero", AMBER),
            ("4  ABSTENTION", "0 primary trades", "Correct under uncertainty", BLUE),
            ("5  COST FAILURE", "0.45 vs 24 bps", "-68.1% base cost", RED),
        ]
        left = 38
        top_y = 332
        box_w = 139
        box_h = 112
        gap = 15
        for index, (head, value, note, color) in enumerate(boxes):
            x = left + index * (box_w + gap)
            self._rounded_rect(x, top_y, box_w, box_h, WHITE, GRID, 8)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.roundRect(x + 10, top_y + box_h - 29, box_w - 20, 19, 5, 0, 1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 7.2)
            self.canvas.drawCentredString(x + box_w / 2, top_y + box_h - 23, head)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 12)
            self.canvas.drawCentredString(x + box_w / 2, top_y + 47, value)
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_REGULAR, 7.8)
            self.canvas.drawCentredString(x + box_w / 2, top_y + 27, note)
            if index < len(boxes) - 1:
                self.canvas.setStrokeColor(HexColor(GRAY))
                self.canvas.setLineWidth(1.2)
                self.canvas.line(x + box_w + 3, top_y + box_h / 2, x + box_w + gap - 3, top_y + box_h / 2)
                self.canvas.setFillColor(HexColor(GRAY))
                self.canvas.setFont(FONT_BOLD, 11)
                self.canvas.drawString(x + box_w + 5, top_y + box_h / 2 - 4, ">")

        self._rounded_rect(38, 73, 488, 224, WHITE, GRID, 9)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 12)
        self.canvas.drawString(54, 273, "Ranked evidence")
        ranked = [
            ("PROVEN", "Objective does not optimize executable return."),
            ("PROVEN", "Gross edge is 53.9x below the one-way base-cost hurdle."),
            ("PROVEN", "Three POC heads are redundant and presence is constant."),
            ("PROVEN", "Every update was gradient-clipped; schedules stopped at 36%."),
            ("PROVEN", "Sparse L1 changes lose magnitude at scaler boundaries."),
            ("LIKELY", "Selection optimism, regime shift, and overfit coexist."),
        ]
        y = 246
        for status, text in ranked:
            color = RED if status == "PROVEN" else AMBER
            self._tag(54, y - 4, status, color, width=48)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_REGULAR, 8.4)
            self.canvas.drawString(111, y, text)
            y -= 29

        self._rounded_rect(542, 73, 261, 224, LIGHT_TEAL, "#B8E7DD", 9)
        self.canvas.setFillColor(HexColor(TEAL))
        self.canvas.setFont(FONT_BOLD, 12)
        self.canvas.drawString(558, 273, "What did not fail")
        passed = [
            "Checkpoint and source hashes match.",
            "Scaler is bound to training IDs.",
            "Horizon alignment reconstructs exactly.",
            "No NaN or attention-logit runaway.",
            "No evidence of a compiler off-by-one.",
        ]
        y = 246
        for text in passed:
            self.canvas.setFillColor(HexColor(TEAL))
            self.canvas.circle(563, y + 2, 2.2, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_REGULAR, 8.4)
            self.canvas.drawString(572, y, text)
            y -= 29
        self._wrapped_text(
            "Conclusion: the artifact is internally consistent, but consistency cannot repair a target, sample, and execution design that does not identify tradable edge.",
            558,
            114,
            227,
            FONT_BOLD,
            8.2,
            11,
            INK,
            max_lines=4,
        )

    def _objective_and_scoring(self) -> None:
        scoring = self.diagnostics["scoring_diagnostics"][
            "untouched_development_forward"
        ]
        contract = self.diagnostics["model_contract"]
        self._new_page(
            "The headline score hides the target that matters",
            "Positive means model improvement over the published empirical marginal; negative means degradation.",
            kicker="02 // OBJECTIVE AND SCORE DECOMPOSITION",
        )
        self._metric_card(
            38,
            365,
            190,
            101,
            "3.46%",
            "objective weight",
            "H5 total-POC quantile cell",
            RED,
        )
        self._metric_card(
            38,
            246,
            190,
            101,
            f"+{scoring['weighted_normalized_pinball']['model_improvement_percent']:.2f}%",
            "aggregate pinball",
            "all horizons and targets",
            BLUE,
        )
        self._metric_card(
            38,
            127,
            190,
            101,
            "75%",
            "forecast weight",
            "three highly correlated POC heads",
            AMBER,
        )
        self._chart_card(self.charts["score_heatmaps"], 246, 126, 557, 340)

        h5_pin = scoring["cell_pinball_improvement_percent"][4]
        h5_mae = scoring["cell_median_mae_improvement_percent"][4]
        self._rounded_rect(38, 55, 765, 51, WHITE, GRID, 8)
        text = (
            f"H5 result: total POC pinball {h5_pin[0]:+.2f}% and median MAE {h5_mae[0]:+.2f}%; "
            f"log-volume pinball {h5_pin[3]:+.2f}% / MAE {h5_mae[3]:+.2f}%; "
            f"delta pinball {h5_pin[4]:+.2f}% / MAE {h5_mae[4]:+.2f}%."
        )
        self._wrapped_text(
            text,
            53,
            87,
            735,
            FONT_BOLD,
            8.8,
            12,
            INK,
            max_lines=2,
        )
        self.canvas.setFillColor(HexColor(MUTED))
        self.canvas.setFont(FONT_REGULAR, 6.8)
        self.canvas.drawRightString(
            795,
            60,
            f"Model parameters: {contract['parameter_count']:,} | target count: 5 x 5 x 3 quantiles",
        )

    def _forecast_and_calibration(self) -> None:
        h5 = self.forward["cells"][4]
        poc = h5[0]
        delta = h5[4]
        self._new_page(
            "Forecasts are smooth, weak, and too uncertain to trade",
            "Forward period: Jun 21-24 only. POC offsets are converted from ticks using the 0.1 USDT tick size.",
            kicker="03 // FORECAST QUALITY AND CALIBRATION",
        )
        self._chart_card(self.charts["forecast"], 38, 251, 765, 219)
        self._chart_card(self.charts["delta"], 38, 70, 360, 159)
        self._rounded_rect(416, 70, 387, 159, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 11)
        self.canvas.drawString(432, 205, "Five-minute diagnostics")
        facts = [
            ("POC Pearson / Spearman", f"{poc['pearson']:.3f} / {poc['spearman']:.3f}"),
            ("Prediction std / actual std", f"{poc['prediction_to_actual_std_ratio']:.3f}"),
            ("Mean q10-q90 width", f"{poc['mean_interval_width'] * self.forward['tick_size']:.2f} USDT"),
            ("Intervals crossing zero", f"{100 * poc['interval_crosses_zero_fraction']:.1f}%"),
            ("Delta median bias / correlation", f"{delta['median_bias']:+.3f} / {delta['pearson']:+.3f}"),
        ]
        y = 179
        for label, value in facts:
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_REGULAR, 8)
            self.canvas.drawString(432, y, label)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 8.4)
            self.canvas.drawRightString(787, y, value)
            y -= 23
        self._rounded_rect(38, 43, 765, 16, LIGHT_RED, "#F4C1CA", 5)
        self.canvas.setFillColor(HexColor(RED))
        self.canvas.setFont(FONT_BOLD, 7.4)
        self.canvas.drawCentredString(
            420,
            48,
            "Calibration coverage alone is not useful when every actionable interval includes zero.",
        )

    def _training_and_optimization(self) -> None:
        train = self.training
        gradients = train["all_recorded_preclip_gradient_norms"]
        self._new_page(
            "Training selected a volatile checkpoint before schedules matured",
            "Mechanics are proven; their causal effect requires controlled ablation.",
            kicker="04 // TRAINING AND OPTIMIZATION",
        )
        self._chart_card(self.charts["training"], 38, 132, 533, 337)
        cards = [
            (
                "100%",
                "updates clipped",
                f"median norm {gradients['median']:.2f} vs clip 1.0",
                RED,
            ),
            (
                "36%",
                "schedule completed",
                "3,618 of 10,050 planned steps",
                AMBER,
            ),
            (
                "0.25",
                "auxiliary weight",
                "never decayed toward 0.05",
                AMBER,
            ),
        ]
        for index, (value, label, note, color) in enumerate(cards):
            self._metric_card(590, 353 - index * 111, 213, 94, value, label, note, color)
        self._rounded_rect(38, 54, 765, 59, WHITE, GRID, 8)
        notes = (
            f"Best epoch 10: {train['best_validation_primary_metric']:.6f}. Final epoch 18: "
            f"{train['epochs'][-1]['primary_metric']:.6f} ({100 * train['final_minus_best_primary_metric'] / train['best_validation_primary_metric']:.1f}% worse). "
            f"Training-window pinball improved {train['epochs'][9]['mean_training_window_pinball']:.6f} -> "
            f"{train['epochs'][-1]['mean_training_window_pinball']:.6f}. Final LR {train['final_learning_rate']:.2e}; intended floor {train['intended_minimum_learning_rate']:.1e}."
        )
        self._wrapped_text(notes, 53, 92, 735, FONT_REGULAR, 8.3, 11.5, INK, max_lines=3)

    def _data_and_power(self) -> None:
        contract = self.diagnostics["model_contract"]
        supplement = self.supplement
        self._new_page(
            "25,758 rows are not 25,758 independent markets",
            "Dependence, scale collapse, and narrow regimes reduce the information available to a 2.09M-parameter model.",
            kicker="05 // DATA, POWER, AND PROVENANCE",
        )
        self._chart_card(self.charts["sample_power"], 38, 230, 431, 240)
        self._rounded_rect(487, 230, 316, 240, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 11)
        self.canvas.drawString(503, 446, "Pipeline defects and validity limits")
        defects = [
            "98.4% adjacent bar-context overlap; 80% H5-label overlap.",
            f"Five future-minute reuses: {supplement['dependence']['horizon_occurrences']:,} occurrences -> {supplement['dependence']['unique_future_minutes']:,} unique minutes.",
            "L1 mid/bid/ask changes clip at about 22.84% each.",
            f"UTC reset leaves {100 * supplement['session_boundary']['partial_64_bar_context_fraction']:.2f}% of rows with partial history.",
            "No independent receive time or native L2 sequence/checksum evidence.",
            "Event time substitutes for availability; decision lag is 0 ms.",
        ]
        y = 421
        for text in defects:
            self.canvas.setFillColor(HexColor(RED if y > 350 else AMBER))
            self.canvas.circle(508, y + 2, 2.1, stroke=0, fill=1)
            y = self._wrapped_text(
                text,
                518,
                y + 5,
                267,
                FONT_REGULAR,
                8.0,
                10.5,
                INK,
                max_lines=3,
            ) - 12

        self._metric_card(
            38,
            99,
            181,
            103,
            "18",
            "training sessions",
            "plus 2 selection / 4 forward",
            RED,
        )
        self._metric_card(
            235,
            99,
            181,
            103,
            f"{contract['parameters_per_train_row']:.1f}",
            "parameters per row",
            f"{contract['parameters_per_train_session']:,.0f} per session",
            AMBER,
        )
        self._metric_card(
            432,
            99,
            181,
            103,
            "1 seed",
            "one fold trained",
            "no variance estimate",
            AMBER,
        )
        self._metric_card(
            629,
            99,
            174,
            103,
            "4 days",
            "forward evidence",
            "insufficient for regime/tail claims",
            RED,
        )
        self._rounded_rect(38, 45, 765, 35, LIGHT_AMBER, "#F2D5A0", 7)
        self._wrapped_text(
            "Holdout integrity disclosure: Jun 25-30 target fields were inadvertently read during an auxiliary schema/count audit, not retained or used. Treat that period as process-accessed and replace it for final confirmation.",
            51,
            68,
            739,
            FONT_BOLD,
            7.6,
            9.5,
            "#7A4A00",
            max_lines=2,
        )

    def _economics(self) -> None:
        econ = self.economics
        self._new_page(
            "Transaction costs are not a sensitivity; they are the result",
            "Five-minute non-overlapping execution proxy. Base cost is 12 bps per side, 24 bps round trip.",
            kicker="06 // ECONOMIC FAILURE",
        )
        self._chart_card(self.charts["economics"], 38, 139, 526, 331)
        self._metric_card(
            582,
            368,
            221,
            102,
            f"{econ['median_long_flat']['base_cost_to_break_even_multiple']:.1f}x",
            "cost / break-even",
            "median long/flat",
            RED,
        )
        self._metric_card(
            582,
            248,
            221,
            102,
            f"{econ['five_minute_direction_proxy']['mcc']:.4f}",
            "direction MCC",
            f"accuracy {100 * econ['five_minute_direction_proxy']['accuracy']:.2f}%",
            AMBER,
        )
        self._metric_card(
            582,
            128,
            221,
            102,
            "0",
            "primary trades",
            "uncertainty-aware interval rule",
            BLUE,
        )
        self._rounded_rect(38, 52, 765, 60, WHITE, GRID, 8)
        text = (
            f"Long/flat: {econ['median_long_flat']['trade_count']} trades, frictionless "
            f"{100 * econ['median_long_flat']['frictionless_compounded_return']:+.2f}%, base "
            f"{100 * econ['median_long_flat']['base_cost_compounded_return']:+.2f}%, break-even "
            f"{econ['median_long_flat']['break_even_one_way_bps']:.3f} bps/side. Long/short: "
            f"{econ['median_long_short']['trade_count']} trades, frictionless "
            f"{100 * econ['median_long_short']['frictionless_compounded_return']:+.2f}%, base "
            f"{100 * econ['median_long_short']['base_cost_compounded_return']:+.2f}%."
        )
        self._wrapped_text(text, 53, 91, 735, FONT_REGULAR, 8.4, 11.5, INK, max_lines=3)

    def _revision_architecture(self) -> None:
        self._new_page(
            "Hybrid architecture: supervised signal first, DQN policy second",
            "The policy stage stays locked until the encoder and a simple cost-aware policy pass every development gate.",
            kicker="07 // TRANSFORMER-DQN SYSTEM DESIGN",
        )
        self._rounded_rect(38, 441, 765, 31, LIGHT_RED, "#F5B8C2", 7)
        self.canvas.setFillColor(HexColor(RED))
        self.canvas.setFont(FONT_BOLD, 8.5)
        self.canvas.drawString(52, 452, "HARD GATE")
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 8.2)
        self.canvas.drawString(
            116,
            452,
            ">=5% proper-score gain + MCC >0.10 + non-negative simple policy after costs - otherwise stop before RL",
        )

        y = 304
        cards = [
            ("QUALIFIED DATA", ">=240 sessions\nreceive time + sequence\nexecutable bid/ask", BLUE),
            ("FEATURES", "tick-aware scaling\nstrict cutoff + latency\ncontinuous warm-up", TEAL),
            ("ENCODER", "fold-local retraining\ncurrent + compact form\nmodality ablations", AMBER),
            ("RETURN HEADS", "long/short q10-q90\nvolatility + downside\nfootprint auxiliary", RED),
            ("STATE ADAPTER", "market forecasts\nportfolio + risk state\ntrain-fold scaling", TEAL),
            ("DQN POLICY", "Double + dueling\ntarget net + n-step\nSHORT / FLAT / LONG", BLUE),
        ]
        card_w = 112
        gap = 17
        for index, (head, body, color) in enumerate(cards):
            x = 38 + index * (card_w + gap)
            self._rounded_rect(x, y, card_w, 112, WHITE, GRID, 8)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.roundRect(x + 8, y + 80, card_w - 16, 22, 5, 0, 1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 6.8)
            self.canvas.drawCentredString(x + card_w / 2, y + 88, head)
            for row, line in enumerate(body.split("\n")):
                self.canvas.setFillColor(HexColor(INK))
                self.canvas.setFont(FONT_REGULAR, 6.9)
                self.canvas.drawCentredString(x + card_w / 2, y + 61 - row * 16, line)
            if index < len(cards) - 1:
                self.canvas.setFillColor(HexColor(GRAY))
                self.canvas.setFont(FONT_BOLD, 13)
                self.canvas.drawString(x + card_w + 4, y + 49, ">")

        self._rounded_rect(38, 54, 371, 222, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(TEAL))
        self.canvas.setFont(FONT_BOLD, 10.5)
        self.canvas.drawString(54, 251, "Encoder protocol")
        encoder_points = [
            ("RETRAIN", "Old weights cannot seed a fold whose validation/test dates they saw."),
            ("COMPARE", "Require simple baselines and a 0.2-0.5M compact encoder ablation."),
            ("FREEZE", "Train the DQN head on fold-local or out-of-fold states first."),
            ("UNFREEZE", "Only the top block, at lower LR, with a supervised-score guardrail."),
        ]
        py = 223
        for tag, body in encoder_points:
            self._tag(54, py - 1, tag, TEAL, width=54)
            self._wrapped_text(body, 119, py + 1, 270, FONT_REGULAR, 7.6, 10, INK, max_lines=2)
            py -= 42

        self._rounded_rect(427, 54, 376, 222, LIGHT_BLUE, "#C8D6FF", 8)
        self.canvas.setFillColor(HexColor(BLUE))
        self.canvas.setFont(FONT_BOLD, 10.5)
        self.canvas.drawString(443, 251, "Two endpoints - never conflate them")
        endpoints = [
            ("FORECAST", "Proper score, MCC, calibration and interval width on rolling folds."),
            ("POLICY", "Raw net PnL, drawdown, tail risk, turnover and paired confidence intervals."),
            ("CONTROL", "DQN must beat the best frozen non-RL policy, not only random or buy-and-hold."),
            ("STATUS", "Proposed architecture only - not implemented, validated, or approved for capital."),
        ]
        py = 223
        for tag, body in endpoints:
            color = BLUE if tag != "STATUS" else RED
            self._tag(443, py - 1, tag, color, width=57)
            self._wrapped_text(body, 511, py + 1, 271, FONT_REGULAR, 7.6, 10, INK, max_lines=2)
            py -= 42

    def _dqn_environment(self) -> None:
        actions = self.dqn_plan["architecture"]["actions"]["values"]
        reward = self.dqn_plan["architecture"]["reward"]
        self._new_page(
            "The environment is an accounting engine, not a reward toy",
            "Use a fixed-small-notional exogenous replay: actions change portfolio state, never the recorded market path.",
            kicker="08 // STATE, ACTION, EXECUTION, AND REWARD",
        )
        action_colors = (RED, GRAY, TEAL)
        action_notes = (
            "Target short; sell only the required position delta.",
            "Target flat; close inventory or remain out of market.",
            "Target long; buy only the required position delta.",
        )
        gap = 12
        width = (765 - 2 * gap) / 3
        for index, (action, note, color) in enumerate(zip(actions, action_notes, action_colors, strict=True)):
            x = 38 + index * (width + gap)
            self._rounded_rect(x, 378, width, 82, WHITE, GRID, 8)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.setFont(FONT_BOLD, 21)
            self.canvas.drawString(x + 14, 424, f"{int(action['id']):+d}")
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 9.5)
            self.canvas.drawString(x + 51, 430, action["name"].replace("_", " / "))
            self._wrapped_text(note, x + 51, 414, width - 65, FONT_REGULAR, 7.2, 9, MUTED, max_lines=2)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.setFont(FONT_BOLD, 6.8)
            self.canvas.drawString(x + 14, 391, "TARGET POSITION - NOT AN ORDER COMMAND")

        self._rounded_rect(38, 285, 765, 68, LIGHT_GRAY, GRID, 8)
        transition = [
            ("OBSERVE", "cutoff-safe state"),
            ("SELECT", "target {-1,0,+1}"),
            ("WAIT", ">=1m latency"),
            ("EXECUTE", "bid/ask + costs"),
            ("MARK", "liquidation equity"),
        ]
        node_w = 127
        node_gap = 25
        for index, (head, body) in enumerate(transition):
            x = 50 + index * (node_w + node_gap)
            self.canvas.setFillColor(HexColor(BLUE if index in (1, 3) else INK))
            self.canvas.setFont(FONT_BOLD, 7.8)
            self.canvas.drawCentredString(x + node_w / 2, 326, head)
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_REGULAR, 7.0)
            self.canvas.drawCentredString(x + node_w / 2, 307, body)
            if index < len(transition) - 1:
                self.canvas.setFillColor(HexColor(GRAY))
                self.canvas.setFont(FONT_BOLD, 12)
                self.canvas.drawString(x + node_w + 6, 314, ">")

        state_cards = [
            (
                38,
                371,
                "MARKET STATE",
                TEAL,
                "Transformer latent + return quantiles; spread, depth, imbalance, volatility, funding, time, liquidity, age and quality flags.",
            ),
            (
                427,
                376,
                "PORTFOLIO STATE",
                AMBER,
                "Current target, entry basis, unrealized PnL, holding time, equity, drawdown, turnover, prior cost, risk budget and episode time.",
            ),
        ]
        for x, w, head, color, body in state_cards:
            self._rounded_rect(x, 169, w, 91, WHITE, GRID, 8)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.setFont(FONT_BOLD, 9.5)
            self.canvas.drawString(x + 16, 237, head)
            self._wrapped_text(body, x + 16, 218, w - 32, FONT_REGULAR, 7.8, 10.5, INK, max_lines=4)

        self._rounded_rect(38, 54, 765, 90, LIGHT_BLUE, "#C8D6FF", 8)
        self.canvas.setFillColor(HexColor(BLUE))
        self.canvas.setFont(FONT_BOLD, 9.8)
        self.canvas.drawString(54, 120, "NET-EQUITY REWARD")
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 9.0)
        self.canvas.drawString(54, 98, reward["formula"])
        self._wrapped_text(
            "Equity charges spread, fees, frozen slippage, funding and terminal close exactly once. Report raw net PnL and risk separately from shaped reward. Selecting the current target is hold/no trade; stale, OOD or risk-breach states force flat.",
            54,
            79,
            729,
            FONT_REGULAR,
            7.2,
            9.3,
            MUTED,
            max_lines=3,
        )

    def _dqn_training_plan(self) -> None:
        self._new_page(
            "Train in stages; never joint-train from scratch on this corpus",
            "Five rolling folds, fold-local states and replay, >=5 encoder seeds, >=10 DQN seeds for the finalist.",
            kicker="09 // TRAINING PROGRAM AND FAILURE CONTROLS",
        )
        phases = [
            (0, "CERTIFY ENVIRONMENT", "Causality, ledger, fills, forced close and deterministic replay.", "Zero causal/accounting defects."),
            (1, "TRAIN ENCODER", "Five rolling folds, fold-local states, >=5 seeds and simple baselines.", "Forecast + simple-policy gates."),
            (2, "TRAIN FROZEN DQN", "Freeze encoder; train adapter and DQN; >=10 finalist seeds.", "Beats best frozen non-RL policy."),
            (3, "FINE-TUNE TOP BLOCK", "Optional lower LR with a supervised forecast-score guardrail.", "No regression or instability."),
            (4, "FREEZE POLICY", "Freeze costs, latency, reward, risk, scaler and champion rule.", "Immutable manifest + checkpoint."),
            (5, "SEALED HOLDOUT", "Open one later >=60-session sealed period exactly once.", "All gates pass; no retuning."),
            (6, "LIVE SHADOW", "Frozen policy, no learning, capital or order route, for >=60 days.", "Zero breaches; shadow gate passes."),
        ]
        gap = 7
        width = (765 - 6 * gap) / 7
        for index, (number, name, work, exit_gate) in enumerate(phases):
            x = 38 + index * (width + gap)
            self._rounded_rect(x, 351, width, 112, WHITE, GRID, 8)
            color = BLUE if index < 4 else (AMBER if index == 4 else RED)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.circle(x + 16, 445, 10, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 7.5)
            self.canvas.drawCentredString(x + 16, 442, str(number))
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 7.1)
            self._wrapped_text(name, x + 31, 446, width - 38, FONT_BOLD, 6.6, 7.8, INK, max_lines=2)
            self._wrapped_text(work, x + 10, 414, width - 20, FONT_REGULAR, 6.2, 7.7, MUTED, max_lines=6)
            self.canvas.setStrokeColor(HexColor(GRID))
            self.canvas.line(x + 10, 375, x + width - 10, 375)
            self._wrapped_text(exit_gate, x + 10, 366, width - 20, FONT_BOLD, 6.0, 7.4, INK, max_lines=2)

        self._rounded_rect(38, 271, 765, 55, LIGHT_BLUE, "#C8D6FF", 8)
        self.canvas.setFillColor(HexColor(BLUE))
        self.canvas.setFont(FONT_BOLD, 9)
        self.canvas.drawString(54, 303, "INITIAL POLICY LEARNER")
        self._wrapped_text(
            "Dueling Double DQN + target network + Huber TD + day-balanced uniform trajectory replay + short n-step returns. QR-DQN or conservative Q regularization is an ablation only after the base learner and non-RL controls pass identical evidence.",
            178,
            305,
            606,
            FONT_REGULAR,
            7.6,
            9.5,
            INK,
            max_lines=3,
        )

        controls = [
            (
                "CHRONOLOGY",
                TEAL,
                "Nested rolling train/select/test blocks with horizon-aware purge and embargo. Scalers, rewards, states and replay are fold-local. No validation or holdout transition enters replay.",
            ),
            (
                "BASELINES + ABLATIONS",
                AMBER,
                "Always-flat, buy-and-hold, random, threshold, tree and DQN-without-transformer. Compare bars, L1, L2; compact vs current topology; frozen vs top-block fine-tune.",
            ),
            (
                "NEGATIVE CONTROLS",
                RED,
                "Shuffle rewards and states within train days, randomize actions, verify regime failures, reconcile zero/base/stress costs and pre-register the trial budget plus one champion rule.",
            ),
        ]
        gap = 12
        width = (765 - 2 * gap) / 3
        for index, (head, color, body) in enumerate(controls):
            x = 38 + index * (width + gap)
            self._rounded_rect(x, 54, width, 191, WHITE, GRID, 8)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.rect(x, 238, width, 7, stroke=0, fill=1)
            self.canvas.setFont(FONT_BOLD, 9.2)
            self.canvas.drawString(x + 14, 216, head)
            self._wrapped_text(body, x + 14, 196, width - 28, FONT_REGULAR, 7.5, 10.2, INK, max_lines=9)
            if index == 0:
                foot = "Old checkpoint weights are disallowed wherever they have seen evaluation dates."
            elif index == 1:
                foot = "DQN must add paired raw net PnL over the strongest frozen non-RL policy."
            else:
                foot = "A shuffled control that appears profitable invalidates the experiment."
            self.canvas.setFillColor(HexColor(color))
            self.canvas.setFont(FONT_BOLD, 6.8)
            self._wrapped_text(foot, x + 14, 82, width - 28, FONT_BOLD, 6.8, 8.5, color, max_lines=3)

    def _experiment_program(self) -> None:
        self._new_page(
            "Promotion requires evidence at every layer",
            "Thresholds are proposed now and must be registered before training; one failure stops the next stage.",
            kicker="10 // ACCEPTANCE GATES AND INDEPENDENT CONFIRMATION",
        )
        stages = [
            ("A", "DATA", "causal + executable", "zero audit defects"),
            ("B", "ENCODER", "5 rolling folds", "forecast gate"),
            ("C", "SIMPLE POLICY", "threshold / tree", "net after costs"),
            ("D", "DQN", "paired incremental", "beats non-RL"),
            ("E", "CONFIRM", ">=60 sealed sessions", "one champion"),
            ("F", "SHADOW", ">=60 live days", "no learning/capital"),
        ]
        start_x = 38
        gap = 9
        width = (PAGE_W - 76 - 5 * gap) / 6
        for index, (letter, head, body, exit_gate) in enumerate(stages):
            x = start_x + index * (width + gap)
            self._rounded_rect(x, 363, width, 103, WHITE, GRID, 8)
            color = BLUE if index < 3 else (AMBER if index == 3 else RED)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.circle(x + 17, 447, 10, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 8)
            self.canvas.drawCentredString(x + 17, 444, letter)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 8.5)
            self.canvas.drawString(x + 33, 443, head)
            self._wrapped_text(body, x + 10, 421, width - 20, FONT_REGULAR, 7.4, 9.3, MUTED, max_lines=2)
            self.canvas.setStrokeColor(HexColor(GRID))
            self.canvas.line(x + 10, 392, x + width - 10, 392)
            self._wrapped_text(exit_gate, x + 10, 383, width - 20, FONT_BOLD, 7.2, 9, INK, max_lines=2)

        gates = [
            ("Data", "zero causal/accounting defects; <1% feature clipping", "purge + embargo; fold-local state/replay"),
            ("Forecast", ">=5% paired proper-score gain; lower 95% CI >0", "positive >=4/5 folds; MCC >0.10"),
            ("Simple policy", "non-negative raw net PnL after frozen base cost", "must pass before DQN is trained"),
            ("DQN value", "paired incremental raw net-PnL lower 95% CI >0", "versus best frozen non-RL policy"),
            ("Economics", "positive under stress; gross edge >=3x all-in cost", "fills + latency + funding + close"),
            ("Stability", ">=80% positive seed-fold runs; >=4/5 fold medians", "no fold/day supplies >30% of profit"),
            ("Controls", "shuffled states/rewards fail; leak sentinel rejected", "all trials retained and registered"),
            ("Confirmation", ">=60 sealed sessions then >=60 shadow days", "one champion; no retuning/learning"),
        ]
        self._rounded_rect(38, 48, 765, 288, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 11)
        self.canvas.drawString(54, 311, "Go / no-go table")
        header_y = 286
        self.canvas.setFillColor(HexColor(LIGHT_GRAY))
        self.canvas.rect(52, header_y - 17, 737, 23, stroke=0, fill=1)
        columns = (54, 156, 505)
        for x, text in zip(columns, ("GATE", "PASS CONDITION", "EVIDENCE"), strict=True):
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_BOLD, 7.2)
            self.canvas.drawString(x, header_y - 9, text)
        y = 251
        for index, (name, condition, evidence) in enumerate(gates):
            if index:
                self.canvas.setStrokeColor(HexColor(GRID))
                self.canvas.line(52, y + 17, 789, y + 17)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 7.5)
            self.canvas.drawString(54, y, name)
            self.canvas.setFont(FONT_REGULAR, 7.4)
            self.canvas.drawString(156, y, condition)
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_REGULAR, 7.1)
            self.canvas.drawString(505, y, evidence)
            y -= 25
        self.canvas.setFillColor(HexColor(RED))
        self.canvas.setFont(FONT_BOLD, 7.7)
        self.canvas.drawRightString(789, 57, "FAIL ANY DEVELOPMENT GATE -> STOP. DO NOT OPEN THE SEALED HOLDOUT.")

    def _audit_and_decision(self) -> None:
        identity = self.results["artifact_identity"]
        self._new_page(
            "Decision: proceed only to data and environment certification",
            "The architecture is specified; its accuracy, incremental value and profitability remain unproven.",
            kicker="11 // DECISION, LIMITATIONS, AND LINEAGE",
        )
        self._rounded_rect(38, 324, 370, 145, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(RED))
        self.canvas.setFont(FONT_BOLD, 11)
        self.canvas.drawString(54, 445, "Proven from this run")
        proven = [
            "Objective is footprint prediction, not executable return.",
            "Median signal is weak; uncertainty crosses zero.",
            "Costs exceed measured edge by more than 50x.",
            "Targets are redundant/degenerate; sparse scaling loses magnitude.",
            "Training is clipping-dominated and stopped before schedule decay.",
        ]
        y = 420
        for item in proven:
            self.canvas.setFillColor(HexColor(RED))
            self.canvas.circle(58, y + 2, 2, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_REGULAR, 8)
            self.canvas.drawString(67, y, item)
            y -= 21

        self._rounded_rect(426, 324, 377, 145, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(AMBER))
        self.canvas.setFont(FONT_BOLD, 11)
        self.canvas.drawString(442, 445, "Not yet proven")
        unproven = [
            "That DQN repairs accuracy or creates economic edge.",
            "That DQN beats the strongest frozen non-RL policy.",
            "That the transformer topology or L2 adds value.",
            "That counterfactual fills transfer at useful capacity.",
            "That any revised model is accurate or profitable.",
        ]
        y = 420
        for item in unproven:
            self.canvas.setFillColor(HexColor(AMBER))
            self.canvas.circle(446, y + 2, 2, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_REGULAR, 8)
            self.canvas.drawString(455, y, item)
            y -= 21

        self._rounded_rect(38, 160, 765, 139, LIGHT_AMBER, "#EBCB8B", 8)
        self.canvas.setFillColor(HexColor("#7A4A00"))
        self.canvas.setFont(FONT_BOLD, 10.5)
        self.canvas.drawString(54, 275, "Holdout integrity")
        self._wrapped_text(
            self.supplement["holdout_integrity_disclosure"]["detail"],
            54,
            255,
            733,
            FONT_REGULAR,
            8.2,
            11,
            "#5E3A00",
            max_lines=4,
        )
        self.canvas.setFillColor(HexColor("#7A4A00"))
        self.canvas.setFont(FONT_BOLD, 8.2)
        self.canvas.drawString(
            54,
            203,
            "Action: retire Jun 25-30; physically seal a later >=60-session period with hashes and access logs.",
        )
        self.canvas.setFillColor(HexColor(MUTED))
        self.canvas.setFont(FONT_REGULAR, 7)
        self.canvas.drawString(
            54,
            181,
            "Historical backtest outer_holdout_accessed=false remains true for artifact creation; it is not a current-session integrity claim.",
        )

        self._rounded_rect(38, 52, 765, 84, WHITE, GRID, 8)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 10)
        self.canvas.drawString(54, 113, "Artifact lineage")
        rows = [
            ("Dataset", identity["dataset_id"]),
            ("Experiment", identity["experiment_config_sha256"]),
            ("Checkpoint", f"step-{int(identity['checkpoint_step']):08d}"),
            ("Plan", self.dqn_plan["plan_id"]),
        ]
        y = 94
        for label, value in rows:
            shown = str(value)
            if len(shown) > 80:
                shown = shown[:38] + "..." + shown[-30:]
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_BOLD, 6.7)
            self.canvas.drawString(54, y, label.upper())
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_REGULAR, 6.7)
            self.canvas.drawString(124, y, shown)
            y -= 14
        self.canvas.setFillColor(HexColor(RED))
        self.canvas.setFont(FONT_BOLD, 8.5)
        self.canvas.drawRightString(789, 113, "FINAL DECISION: RESEARCH PLAN ONLY")
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 9.5)
        self.canvas.drawRightString(789, 91, "NOT IMPLEMENTED / NO DEPLOYMENT / NO CAPITAL")

    def _metric_card(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        label: str,
        note: str,
        color: str,
        *,
        dark: bool = False,
    ) -> None:
        fill = "#111E33" if dark else WHITE
        stroke = "#263650" if dark else GRID
        self._rounded_rect(x, y, w, h, fill, stroke, 8)
        self.canvas.setFillColor(HexColor(color))
        self.canvas.rect(x, y + h - 5, w, 5, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(WHITE if dark else INK))
        self.canvas.setFont(FONT_BOLD, 20)
        self.canvas.drawString(x + 14, y + h - 38, value)
        self.canvas.setFont(FONT_BOLD, 8.5)
        self.canvas.drawString(x + 14, y + 33, label)
        self.canvas.setFillColor(HexColor("#A9B8CB" if dark else MUTED))
        self.canvas.setFont(FONT_REGULAR, 7.2)
        self.canvas.drawString(x + 14, y + 17, note)

    def _chart_card(self, path: Path, x: float, y: float, w: float, h: float) -> None:
        self._rounded_rect(x, y, w, h, WHITE, GRID, 8)
        image = ImageReader(str(path))
        self.canvas.drawImage(
            image,
            x + 7,
            y + 7,
            width=w - 14,
            height=h - 14,
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
        stroke: str,
        radius: float,
    ) -> None:
        self.canvas.setFillColor(HexColor(fill))
        self.canvas.setStrokeColor(HexColor(stroke))
        self.canvas.setLineWidth(0.6)
        self.canvas.roundRect(x, y, w, h, radius, stroke=1, fill=1)

    def _tag(
        self,
        x: float,
        y: float,
        text: str,
        color: str,
        *,
        width: float,
    ) -> None:
        self.canvas.setFillColor(HexColor(color))
        self.canvas.roundRect(x, y, width, 15, 4, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setFont(FONT_BOLD, 6.3)
        self.canvas.drawCentredString(x + width / 2, y + 4.7, text)

    def _wrapped_text(
        self,
        text: str,
        x: float,
        y: float,
        max_width: float,
        font: str,
        size: float,
        leading: float,
        color: str,
        *,
        max_lines: int | None = None,
    ) -> float:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if pdfmetrics.stringWidth(candidate, font, size) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        if max_lines is not None and len(lines) > max_lines:
            lines = lines[:max_lines]
            while lines and pdfmetrics.stringWidth(lines[-1] + "...", font, size) > max_width:
                lines[-1] = lines[-1][:-1]
            if lines:
                lines[-1] += "..."
        self.canvas.setFillColor(HexColor(color))
        self.canvas.setFont(font, size)
        cursor = y
        for line in lines:
            self.canvas.drawString(x, cursor, line)
            cursor -= leading
        return cursor


def _build_charts(
    diagnostics: dict[str, Any],
    results: dict[str, Any],
    supplement: dict[str, Any],
    workdir: Path,
) -> dict[str, Path]:
    _set_plot_style()
    charts = {
        "score_heatmaps": workdir / "score-heatmaps.png",
        "forecast": workdir / "forecast-diagnostics.png",
        "delta": workdir / "delta-diagnostic.png",
        "training": workdir / "training-diagnostics.png",
        "sample_power": workdir / "sample-power.png",
        "economics": workdir / "economics.png",
    }
    for path in charts.values():
        if path.exists():
            raise FileExistsError(path)
    _plot_score_heatmaps(diagnostics, charts["score_heatmaps"])
    _plot_forecast(diagnostics, charts["forecast"])
    _plot_delta(diagnostics, charts["delta"])
    _plot_training(diagnostics, charts["training"])
    _plot_sample_power(diagnostics, supplement, charts["sample_power"])
    _plot_economics(diagnostics, charts["economics"])
    return charts


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.labelsize": 8,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.labelcolor": MUTED,
        }
    )


def _base_axis(ax: Any) -> None:
    ax.set_facecolor(WHITE)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)


def _save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def _plot_score_heatmaps(diagnostics: dict[str, Any], path: Path) -> None:
    score = diagnostics["scoring_diagnostics"]["untouched_development_forward"]
    arrays = [
        np.asarray(score["cell_pinball_improvement_percent"], dtype=float),
        np.asarray(score["cell_median_mae_improvement_percent"], dtype=float),
    ]
    titles = ("Pinball improvement (%)", "Median MAE improvement (%)")
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.8), constrained_layout=True)
    limit = max(45.0, max(float(np.max(np.abs(array))) for array in arrays))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = plt.get_cmap("RdYlGn")
    for ax, array, title in zip(axes, arrays, titles, strict=True):
        image = ax.imshow(array, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title, pad=10)
        ax.set_xticks(range(5), TARGET_SHORT, rotation=28, ha="right")
        ax.set_yticks(range(5), [f"H{index}" for index in range(1, 6)])
        ax.tick_params(length=0, labelsize=7.5)
        for row in range(5):
            for column in range(5):
                value = array[row, column]
                ax.text(
                    column,
                    row,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=7.3,
                    fontweight="semibold",
                    color=WHITE if abs(value) > 0.55 * limit else INK,
                )
        fig.colorbar(image, ax=ax, fraction=0.036, pad=0.025).ax.tick_params(labelsize=7)
    _save(fig, path)


def _plot_forecast(diagnostics: dict[str, Any], path: Path) -> None:
    forward = diagnostics["partition_metrics"]["untouched_development_forward"]
    cells = [forward["cells"][index][0] for index in range(5)]
    horizons = np.arange(1, 6)
    pearson = np.asarray([cell["pearson"] for cell in cells])
    spearman = np.asarray([cell["spearman"] for cell in cells])
    std_ratio = np.asarray([cell["prediction_to_actual_std_ratio"] for cell in cells])
    widths = np.asarray([cell["mean_interval_width"] for cell in cells]) * float(
        forward["tick_size"]
    )
    coverage = 100 * np.asarray([cell["interval_coverage"] for cell in cells])

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0), constrained_layout=True)
    ax = axes[0]
    _base_axis(ax)
    ax.plot(horizons, pearson, marker="o", color=BLUE, linewidth=2)
    ax.plot(horizons, spearman, marker="s", color=TEAL, linewidth=2)
    ax.text(5.05, pearson[-1], "Pearson", color=BLUE, fontsize=7, va="center")
    ax.text(5.05, spearman[-1] - 0.006, "Spearman", color=TEAL, fontsize=7, va="center")
    ax.set_title("Total-POC conditional correlation")
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon minute")
    ax.set_ylim(0, 0.11)

    ax = axes[1]
    _base_axis(ax)
    bars = ax.bar(horizons, std_ratio, color=AMBER, width=0.62)
    ax.axhline(1.0, color=RED, linewidth=1, linestyle="--")
    for bar, value in zip(bars, std_ratio, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.008, f"{value:.2f}", ha="center", fontsize=7)
    ax.set_title("Prediction std / actual std")
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon minute")
    ax.set_ylim(0, 1.08)

    ax = axes[2]
    _base_axis(ax)
    bars = ax.bar(horizons, widths, color=RED, alpha=0.88, width=0.62)
    ax.set_title("Interval width grows with horizon")
    ax.set_xticks(horizons)
    ax.set_xlabel("Horizon minute")
    ax.set_ylabel("Mean q10-q90 width (USDT)")
    ax2 = ax.twinx()
    ax2.plot(horizons, coverage, color=BLUE, marker="o", linewidth=1.8)
    ax2.set_ylim(70, 88)
    ax2.set_ylabel("Coverage (%)", color=BLUE)
    ax2.tick_params(axis="y", colors=BLUE, labelsize=7)
    for bar, value in zip(bars, widths, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 3, f"${value:.0f}", ha="center", fontsize=7)
    _save(fig, path)


def _plot_delta(diagnostics: dict[str, Any], path: Path) -> None:
    delta = diagnostics["partition_metrics"]["untouched_development_forward"][
        "cells"
    ][4][4]
    actual = delta["actual"]
    predicted = delta["prediction_median"]
    fig, ax = plt.subplots(figsize=(6.4, 2.45), constrained_layout=True)
    ax.set_facecolor(WHITE)
    ax.axvline(0, color=GRID, linewidth=1)
    series = [
        ("Actual delta", actual, TEAL),
        ("Predicted median", predicted, RED),
    ]
    for y, (label, values, color) in enumerate(series[::-1]):
        ax.plot([values["q05"], values["q95"]], [y, y], color=color, linewidth=7, solid_capstyle="round")
        ax.scatter(values["q50"], y, color=INK, s=38, zorder=3)
        ax.text(values["q50"], y + 0.2, f"median {values['q50']:+.3f}", ha="center", fontsize=7)
    ax.set_yticks([0, 1], ["Predicted median", "Actual delta"])
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel("q05 - q95 distribution across forward rows")
    ax.set_title("Delta head is biased and near-constant")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.tick_params(labelsize=7.5)
    _save(fig, path)


def _plot_training(diagnostics: dict[str, Any], path: Path) -> None:
    training = diagnostics["training_diagnostics"]
    epochs = training["epochs"]
    x = np.asarray([item["epoch"] for item in epochs])
    validation = np.asarray([item["primary_metric"] for item in epochs])
    train_pin = np.asarray([item["mean_training_window_pinball"] for item in epochs])
    grad = np.asarray([item["gradient_norm_before_clip"]["median"] for item in epochs])
    fig, axes = plt.subplots(2, 1, figsize=(8.7, 5.4), constrained_layout=True)
    ax = axes[0]
    _base_axis(ax)
    ax.plot(x, validation, color=BLUE, marker="o", markersize=3.5, linewidth=1.8)
    ax.axhline(
        training["empirical_baseline_validation_primary_metric"],
        color=GRAY,
        linestyle="--",
        linewidth=1.2,
    )
    best = training["best_epoch"]
    ax.scatter(best, validation[best - 1], color=TEAL, s=55, zorder=4)
    ax.annotate(
        f"selected epoch {best}",
        (best, validation[best - 1]),
        xytext=(best + 0.6, validation[best - 1] - 0.008),
        fontsize=7,
        color=TEAL,
    )
    ax.text(18.1, training["empirical_baseline_validation_primary_metric"], "empirical baseline", fontsize=7, color=GRAY, va="center")
    ax.set_title("Validation weighted pinball is volatile")
    ax.set_ylabel("Lower is better")
    ax.set_xlim(1, 19.5)
    ax.set_xticks(range(1, 19, 2))

    ax = axes[1]
    _base_axis(ax)
    ax.plot(x, grad, color=RED, marker="o", markersize=3.5, linewidth=1.8)
    ax.axhline(1.0, color=INK, linestyle="--", linewidth=1.1)
    ax.set_yscale("log")
    ax.set_title("Median pre-clip gradient norm by epoch")
    ax.set_ylabel("Log scale")
    ax.set_xlabel("Epoch")
    ax.set_xlim(1, 19.5)
    ax.set_xticks(range(1, 19, 2))
    ax.text(18.1, 1.0, "clip = 1", fontsize=7, color=INK, va="center")
    ax2 = ax.twinx()
    ax2.plot(x, train_pin, color=AMBER, linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("Training-window pinball", color=AMBER)
    ax2.tick_params(axis="y", colors=AMBER, labelsize=7)
    _save(fig, path)


def _plot_sample_power(
    diagnostics: dict[str, Any], supplement: dict[str, Any], path: Path
) -> None:
    train = diagnostics["partition_metrics"]["in_sample_train_replay"]
    h5 = train["cells"][4][0]
    nominal = train["row_count"]
    ess_target = h5["dependence"]["actual"]["heuristic_effective_sample_size"]
    ess_prediction = h5["dependence"]["prediction"][
        "heuristic_effective_sample_size"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.0), constrained_layout=True)
    ax = axes[0]
    _base_axis(ax)
    labels = ("Rows", "H5 target ESS", "Prediction ESS")
    values = np.asarray([nominal, ess_target, ess_prediction])
    colors = (BLUE, AMBER, RED)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_yscale("log")
    ax.set_title("Nominal vs heuristic effective sample")
    ax.set_ylabel("Count, log scale")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,.0f}", ha="center", fontsize=7)

    ax = axes[1]
    _base_axis(ax)
    labels = ("Bar context", "H5 labels", "Partial UTC warm-up")
    values = np.asarray(
        [
            diagnostics["model_contract"]["adjacent_bar_context_overlap_fraction"],
            diagnostics["model_contract"]["adjacent_five_minute_label_overlap_fraction"],
            supplement["session_boundary"]["partial_64_bar_context_fraction"],
        ]
    )
    bars = ax.barh(labels, 100 * values, color=(RED, AMBER, GRAY), height=0.55)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Percent")
    ax.set_title("Dependence and boundary effects")
    for bar, value in zip(bars, 100 * values, strict=True):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=7)
    _save(fig, path)


def _plot_economics(diagnostics: dict[str, Any], path: Path) -> None:
    econ = diagnostics["economic_diagnostics"]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.9), constrained_layout=True)
    ax = axes[0]
    _base_axis(ax)
    labels = ("Measured gross edge", "Base round-trip cost")
    values = (
        econ["median_long_flat"]["gross_round_trip_edge_bps"],
        econ["base_round_trip_cost_bps"],
    )
    bars = ax.bar(labels, values, color=(TEAL, RED), width=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("Basis points, log scale")
    ax.set_title("Cost hurdle vs measured edge")
    ax.tick_params(axis="x", rotation=16)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value * 1.18, f"{value:.3f}", ha="center", fontsize=8, fontweight="semibold")

    ax = axes[1]
    _base_axis(ax)
    policies = ("Long/flat", "Long/short")
    frictionless = 100 * np.asarray(
        [
            econ["median_long_flat"]["frictionless_compounded_return"],
            econ["median_long_short"]["frictionless_compounded_return"],
        ]
    )
    base = 100 * np.asarray(
        [
            econ["median_long_flat"]["base_cost_compounded_return"],
            econ["median_long_short"]["base_cost_compounded_return"],
        ]
    )
    x = np.arange(2)
    width = 0.34
    ax.bar(x - width / 2, frictionless, width, color=TEAL)
    ax.bar(x + width / 2, base, width, color=RED)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, policies)
    ax.set_ylabel("Compounded return (%)")
    ax.set_title("Frictionless appearance vs modeled reality")
    for pos, value in zip(x - width / 2, frictionless, strict=True):
        ax.text(pos, value + 2, f"{value:+.1f}%", ha="center", fontsize=7, color=TEAL)
    for pos, value in zip(x + width / 2, base, strict=True):
        ax.text(pos, value - 6, f"{value:+.1f}%", ha="center", fontsize=7, color=RED)
    ax.text(0.03, 0.95, "teal: frictionless\nred: base cost", transform=ax.transAxes, va="top", fontsize=7, color=MUTED)
    _save(fig, path)


def _validate_sources(
    diagnostics: dict[str, Any],
    results: dict[str, Any],
    protocol: dict[str, Any],
    supplement: dict[str, Any],
    dqn_plan: dict[str, Any],
    paths: dict[str, Path],
) -> None:
    if diagnostics.get("study_kind") != "chmt_q_deep_failure_analysis":
        raise ValueError("unexpected diagnostic study kind")
    if diagnostics.get("outer_holdout_used_in_failure_metrics") is not False:
        raise ValueError("failure metrics used the outer holdout")
    status = diagnostics.get("current_audit_holdout_integrity", {}).get("status")
    if status != "inadvertently_read_not_used":
        raise ValueError("revision report requires the current holdout disclosure")
    if supplement.get("holdout_integrity_disclosure", {}).get("status") != status:
        raise ValueError("supplement and diagnostics disagree on holdout integrity")
    if supplement.get("quantitative_scope", {}).get("sealed_values_used") is not False:
        raise ValueError("supplement claims use of sealed values")
    if results.get("outer_holdout_accessed") is not False:
        raise ValueError("historical backtest artifact did not preserve its holdout boundary")
    if protocol.get("sealed_outer_holdout", {}).get("will_access") is not False:
        raise ValueError("protocol did not freeze the original holdout")
    expected_rows = {
        "in_sample_train_replay": 25758,
        "selected_validation": 2862,
        "untouched_development_forward": 5724,
    }
    if diagnostics.get("row_counts") != expected_rows or results.get("row_counts") != expected_rows:
        raise ValueError("evaluated row counts differ from the frozen study")
    sources = diagnostics.get("source_artifacts", {})
    for name in ("results", "protocol"):
        expected = sources.get(name, {}).get("sha256")
        if expected != _sha256(paths[name]):
            raise ValueError(f"diagnostic source hash mismatch: {name}")
    if dqn_plan.get("schema_version") != 1:
        raise ValueError("unexpected Transformer-DQN plan schema")
    if dqn_plan.get("status") != "proposed_not_implemented_not_validated":
        raise ValueError("architecture plan must remain explicitly unvalidated")
    actions = dqn_plan.get("architecture", {}).get("actions", {}).get("values", [])
    action_ids = [action.get("id") for action in actions]
    targets = [action.get("target_position") for action in actions]
    if action_ids != [-1, 0, 1] or targets != [-1.0, 0.0, 1.0]:
        raise ValueError("target-position action space must be exactly short/flat/long")
    phases = [phase.get("phase") for phase in dqn_plan.get("training_program", [])]
    if phases != list(range(7)):
        raise ValueError("training program must contain ordered phases 0 through 6")
    validation = dqn_plan.get("validation_design", {})
    validation_text = json.dumps(validation, sort_keys=True).lower()
    for required in ("five rolling", "fold-local", "ten dqn", "physically sealed"):
        if required not in validation_text:
            raise ValueError(f"architecture validation design is missing: {required}")
    simulator = dqn_plan.get("architecture", {}).get("environment", {}).get(
        "simulator_class", ""
    )
    if "exogenous historical counterfactual simulator" not in simulator.lower():
        raise ValueError("architecture must select one explicit simulator class")
    confirmation = dqn_plan.get("go_no_go_gates", [])[-1].get("condition", "")
    if "60" not in confirmation or "one" not in confirmation.lower():
        raise ValueError("confirmation gate must require one champion and 60-day stages")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
