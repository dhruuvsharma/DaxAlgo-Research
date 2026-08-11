"""Generate the frozen FDT-N research PDF from complete multi-seed results."""

# ruff: noqa: E501 - long authored paper sentences remain readable as source text.

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from pypdf import PdfReader
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .config import BAR_FEATURE_NAMES, ROW_FEATURE_NAMES, TARGET_NAMES

plt.switch_backend("Agg")
matplotlib.rcParams["axes.unicode_minus"] = False

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
PURPLE = "#7A5AF8"
LIGHT_BLUE = "#EAF0FF"
LIGHT_TEAL = "#E5F7F3"
LIGHT_AMBER = "#FFF5DE"

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

TARGET_LABELS = {
    "poc_offset": "Total POC",
    "low_offset": "Low",
    "high_offset": "High",
    "buy_poc_offset": "Buy POC",
    "sell_poc_offset": "Sell POC",
    "log_volume_relative": "Log volume",
    "delta_fraction": "Delta fraction",
}

REFERENCES: tuple[tuple[str, str], ...] = (
    (
        "S01",
        "Vaswani et al. Attention Is All You Need. NeurIPS, 2017. "
        "https://arxiv.org/abs/1706.03762",
    ),
    (
        "S02",
        "Zhang and Sennrich. Root Mean Square Layer Normalization. NeurIPS, 2019. "
        "https://arxiv.org/abs/1910.07467",
    ),
    (
        "S03",
        "Shazeer. GLU Variants Improve Transformer. 2020. "
        "https://arxiv.org/abs/2002.05202",
    ),
    (
        "S04",
        "Shaw, Uszkoreit, and Vaswani. Self-Attention with Relative Position Representations. "
        "NAACL, 2018. https://arxiv.org/abs/1803.02155",
    ),
    (
        "S05",
        "Zeng et al. Are Transformers Effective for Time Series Forecasting? AAAI, 2023. "
        "https://doi.org/10.1609/aaai.v37i9.26317",
    ),
    (
        "S06",
        "Kim et al. Reversible Instance Normalization for Accurate Time-Series Forecasting "
        "against Distribution Shift. ICLR, 2022. https://openreview.net/forum?id=cGDAkQo1C0p",
    ),
    (
        "S07",
        "Lim et al. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series "
        "Forecasting. International Journal of Forecasting, 2021. "
        "https://doi.org/10.1016/j.ijforecast.2021.03.012",
    ),
    (
        "S08",
        "Nie et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. "
        "ICLR, 2023. https://arxiv.org/abs/2211.14730",
    ),
    (
        "S09",
        "Liu et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. "
        "ICLR, 2024. https://arxiv.org/abs/2310.06625",
    ),
    (
        "S10",
        "Zhang and Yan. Crossformer: Transformer Utilizing Cross-Dimension Dependency for "
        "Multivariate Time Series Forecasting. ICLR, 2023. "
        "https://openreview.net/forum?id=vSVLM2j9eie",
    ),
    (
        "S11",
        "Wang et al. TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous "
        "Variables. NeurIPS, 2024. https://arxiv.org/abs/2402.19072",
    ),
    (
        "S12",
        "Woo et al. Unified Training of Universal Time Series Forecasting Transformers. ICML, 2024. "
        "https://arxiv.org/abs/2402.02592",
    ),
    (
        "S13",
        "Zhang, Zohren, and Roberts. DeepLOB: Deep Convolutional Neural Networks for Limit Order "
        "Books. IEEE Transactions on Signal Processing, 2019. "
        "https://doi.org/10.1109/TSP.2019.2907260",
    ),
    (
        "S14",
        "Wallbridge. Transformers for Limit Order Books. 2020. "
        "https://arxiv.org/abs/2003.00130",
    ),
    (
        "S15",
        "Berti and Kasneci. TLOB: A Novel Transformer Model with Dual Attention for Price Trend "
        "Prediction with Limit Order Book Data. 2025. https://arxiv.org/abs/2502.15757",
    ),
    (
        "S16",
        "Dai et al. DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts "
        "Language Models. 2024. https://arxiv.org/abs/2401.06066",
    ),
    (
        "S17",
        "DeepSeek-AI. DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language "
        "Model. 2024. https://arxiv.org/abs/2405.04434",
    ),
    (
        "S18",
        "DeepSeek-AI. DeepSeek-V3 Technical Report. 2024/2025. "
        "https://arxiv.org/abs/2412.19437",
    ),
    (
        "S19",
        "Yuan et al. Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse "
        "Attention. 2025. https://arxiv.org/abs/2502.11089",
    ),
    (
        "S20",
        "Yang et al. Qwen2 Technical Report. 2024. https://arxiv.org/abs/2407.10671",
    ),
    (
        "S21",
        "Qwen Team. Qwen3 Technical Report. 2025. https://arxiv.org/abs/2505.09388",
    ),
    (
        "S22",
        "Sun, Cetin, and Tang. Learning to Predict Structural Adaptations of Large Language Models. "
        "2025. https://arxiv.org/abs/2501.06252",
    ),
    (
        "S23",
        "Akiba et al. Evolutionary Optimization of Model Merging Recipes. Nature Machine "
        "Intelligence, 2025. https://doi.org/10.1038/s42256-024-00975-8",
    ),
    (
        "S24",
        "Kimi Team. Kimi Linear: An Expressive, Efficient Attention Architecture. 2025. "
        "https://arxiv.org/abs/2510.26692",
    ),
    (
        "S25",
        "Kimi Team. Kimi K2: Open Agentic Intelligence. 2025/2026. "
        "https://arxiv.org/abs/2507.20534",
    ),
    (
        "S26",
        "Romano, Patterson, and Candes. Conformalized Quantile Regression. NeurIPS, 2019. "
        "https://arxiv.org/abs/1905.03222",
    ),
    (
        "S27",
        "Ntakaris et al. Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data. "
        "Journal of Forecasting, 2018. https://doi.org/10.1002/for.2543",
    ),
    (
        "S28",
        "Kolm, Turiel, and Westray. Deep Order Flow Imbalance: Extracting Alpha at Multiple "
        "Horizons from the Limit Order Book. Mathematical Finance, 2023. "
        "https://doi.org/10.1111/mafi.12413",
    ),
    (
        "D01",
        "Binance Data Vision. BTCUSDT spot daily individual-trades archive, 2024-01-01 through "
        "2024-01-14. https://data.binance.vision/?prefix=data/spot/daily/trades/BTCUSDT/",
    ),
    (
        "D02",
        "DaxAlgo Research. CHMT-Q BTC-USDT June 2026 Development Backtest. Local research report, "
        "2026-07-26. docs/research/numeric-market-transformer/okx-june-2026-development-backtest-report.md",
    ),
    (
        "D03",
        "DaxAlgo Research. BTCUSDT Fourteen-Day Exact-Row Dataset Manifest. 2026. Manifest and "
        "archive SHA-256 identities are reproduced in this paper's artifact ledger.",
    ),
)


class PendingResultsError(RuntimeError):
    """Raised when final report inputs do not yet contain complete evaluation metrics."""


@dataclass(frozen=True)
class Artifact:
    label: str
    path: Path
    sha256: str
    bytes: int
    seed: int | None = None


@dataclass(frozen=True)
class RunRecord:
    path: Path
    data: dict[str, Any]
    seed: int
    validation_loss: float
    checkpoint: Artifact
    splits: Artifact
    ridge: Artifact


@dataclass(frozen=True)
class BoundAudit:
    scope: str
    feature: str
    lower_bound: float | None
    lower_hits: int | None
    upper_bound: float | None
    upper_hits: int | None
    observations: int
    method: str


@dataclass(frozen=True)
class CorpusRecord:
    manifest_path: Path
    dataset_path: Path
    manifest: dict[str, Any]
    manifest_artifact: Artifact
    dataset_artifact: Artifact
    row_percentiles: tuple[float, float, float, float]
    archive_bytes: int
    clipping_audit: tuple[BoundAudit, ...]


@dataclass(frozen=True)
class DeploymentRecord:
    data: dict[str, Any]
    metadata: Artifact
    onnx: Artifact


@dataclass(frozen=True)
class ReportBundle:
    corpus: CorpusRecord
    runs: tuple[RunRecord, ...]
    primary: RunRecord
    deployment: DeploymentRecord
    paper: Artifact
    ledger: Artifact


def generate_pdf_report(
    results_paths: Sequence[Path],
    dataset_manifest_path: Path,
    dataset_path: Path,
    paper_path: Path,
    ledger_path: Path,
    model_metadata_path: Path,
    output_path: Path,
    workdir: Path,
    *,
    enforce_repo_paths: bool = True,
    watermark: str | None = None,
) -> dict[str, Any]:
    """Validate inputs, build figures, and atomically publish a multi-page PDF."""

    bundle = load_report_bundle(
        results_paths,
        dataset_manifest_path,
        dataset_path,
        paper_path,
        ledger_path,
        model_metadata_path,
    )
    if len(bundle.runs) < 3 and watermark is None:
        raise PendingResultsError(
            "FDT-N PDF not generated: final qualification requires at least three declared "
            "seeds; fewer complete seed results were supplied."
        )
    output = output_path.resolve()
    figure_dir = workdir.resolve()
    repo_root = _find_repo_root(bundle.paper.path)
    if enforce_repo_paths:
        _require_descendant(output, repo_root / "output" / "pdf", "PDF output")
        _require_descendant(figure_dir, repo_root / "tmp" / "pdfs", "PDF work directory")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output}")

    figure_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    charts = _build_charts(bundle, figure_dir)
    staged = figure_dir / "fdt-n-report.staged.pdf"
    if staged.exists():
        staged.unlink()
    _register_fonts()
    report = VisualReport(staged, bundle, charts, watermark=watermark)
    try:
        report.render()
        reader = PdfReader(str(staged))
        if len(reader.pages) != report.page_number or len(reader.pages) < 9 + len(bundle.runs):
            raise RuntimeError("rendered report has an unexpected page count")
        extracted = "\n".join(page.extract_text() or "" for page in reader.pages)
        if "[PENDING]" in extracted or "PLACEHOLDER" in extracted:
            raise RuntimeError("rendered report contains a pending or placeholder token")
        shutil.copyfile(staged, output)
    finally:
        if staged.exists():
            staged.unlink()

    return {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "pages": report.page_number,
        "seed_count": len(bundle.runs),
        "primary_seed": bundle.primary.seed,
        "primary_selection": "minimum_validation_loss_only",
        "model_metadata_sha256": bundle.deployment.metadata.sha256,
        "onnx_sha256": bundle.deployment.onnx.sha256,
        "figures": {name: str(path) for name, path in charts.items()},
    }


def load_report_bundle(
    results_paths: Sequence[Path],
    dataset_manifest_path: Path,
    dataset_path: Path,
    paper_path: Path,
    ledger_path: Path,
    model_metadata_path: Path,
) -> ReportBundle:
    """Load and fail-closed validate the complete reporting bundle."""

    if not results_paths:
        raise PendingResultsError(
            "FDT-N PDF not generated: final training/evaluation metrics are pending; "
            "no results JSON was supplied."
        )
    missing = [str(path) for path in results_paths if not path.is_file()]
    if missing:
        raise PendingResultsError(
            "FDT-N PDF not generated: final training/evaluation metrics are pending; "
            f"missing results JSON: {', '.join(missing)}"
        )

    manifest_path = _require_file(dataset_manifest_path, "dataset manifest")
    encoded_path = _require_file(dataset_path, "encoded dataset")
    source_paper = _require_file(paper_path, "research paper source")
    evidence = _require_file(ledger_path, "evidence ledger")
    repo_root = _find_repo_root(source_paper)
    manifest = _read_json(manifest_path)
    corpus = _validate_corpus(manifest_path, encoded_path, manifest)
    runs = tuple(
        _validate_run(path.resolve(), _read_json(path), corpus, repo_root) for path in results_paths
    )
    _validate_run_set(runs)
    primary = select_primary_run(runs)
    deployment = _validate_deployment_metadata(
        model_metadata_path, primary=primary, corpus=corpus
    )

    paper_text = source_paper.read_text(encoding="utf-8")
    ledger_text = evidence.read_text(encoding="utf-8")
    if not paper_text.startswith("# FDT-N:"):
        raise ValueError("research paper source is not the FDT-N paper")
    if not ledger_text.startswith("# FDT-N evidence"):
        raise ValueError("evidence ledger source is not the FDT-N ledger")
    return ReportBundle(
        corpus=corpus,
        runs=runs,
        primary=primary,
        deployment=deployment,
        paper=_artifact("Research paper", source_paper),
        ledger=_artifact("Evidence ledger", evidence),
    )


def select_primary_run(runs: Sequence[RunRecord]) -> RunRecord:
    """Select by validation loss only; test metrics are deliberately not consulted."""

    if not runs:
        raise ValueError("at least one run is required")
    return min(runs, key=lambda run: (run.validation_loss, run.seed))


def _validate_deployment_metadata(
    path: Path,
    *,
    primary: RunRecord,
    corpus: CorpusRecord,
) -> DeploymentRecord:
    try:
        metadata_path = _require_file(path, "selected model metadata")
        data = _read_json(metadata_path)
        _reject_pending_values(data, "model metadata")
        if data.get("schema_version") != 2:
            raise ValueError("metadata schema is not version 2")
        if data.get("model_kind") != "fdt-n-footprint-distribution-transformer":
            raise ValueError("metadata model kind is not FDT-N")
        if data.get("encoding_config") != primary.data["encoding_config"]:
            raise ValueError("metadata encoding configuration differs from the primary run")
        if data.get("model_config") != primary.data["model_config"]:
            raise ValueError("metadata model configuration differs from the primary run")
        if int(data.get("parameter_count", -1)) != int(primary.data["parameter_count"]):
            raise ValueError("metadata parameter count differs from the primary run")
        if tuple(data.get("row_feature_names", ())) != ROW_FEATURE_NAMES:
            raise ValueError("metadata row-feature order differs from the encoder")
        if tuple(data.get("bar_feature_names", ())) != BAR_FEATURE_NAMES:
            raise ValueError("metadata bar-feature order differs from the encoder")
        if tuple(data.get("target_names", ())) != TARGET_NAMES:
            raise ValueError("metadata target order differs from the evaluation")
        if data.get("quantiles") != [0.1, 0.5, 0.9]:
            raise ValueError("metadata quantile order is not q10/q50/q90")
        if int(data.get("minimum_history_bars", -1)) != int(
            primary.data["encoding_config"]["lookback_bars"]
        ):
            raise ValueError("metadata history requirement differs from the primary run")
        if int(data.get("maximum_horizon_bars", -1)) != int(
            primary.data["encoding_config"]["horizon_bars"]
        ):
            raise ValueError("metadata horizon differs from the primary run")

        checkpoint_hash = data.get("checkpoint_sha256")
        if checkpoint_hash != primary.checkpoint.sha256:
            raise ValueError("metadata checkpoint hash is not the validation-selected checkpoint")
        if data.get("model_version") != primary.checkpoint.sha256[:16]:
            raise ValueError("metadata model version is not derived from the selected checkpoint")
        if int(data.get("checkpoint_epoch", -1)) != int(primary.data["best_epoch"]):
            raise ValueError("metadata checkpoint epoch differs from the selected epoch")
        checkpoint_loss = _finite(
            data.get("checkpoint_validation_loss"), "metadata checkpoint validation loss"
        )
        if not math.isclose(
            checkpoint_loss, primary.validation_loss, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("metadata checkpoint loss differs from the primary validation loss")

        lineage = data.get("lineage")
        if not isinstance(lineage, dict):
            raise ValueError("metadata lineage is missing")
        expected_lineage = {
            "dataset": corpus.dataset_artifact,
            "dataset_manifest": corpus.manifest_artifact,
            "training_results": _artifact("Primary results", primary.path, seed=primary.seed),
        }
        for name, artifact in expected_lineage.items():
            record = lineage.get(name)
            if not isinstance(record, dict) or record.get("sha256") != artifact.sha256:
                raise ValueError(f"metadata lineage {name} hash does not match the report input")
            if record.get("file") != artifact.path.name:
                raise ValueError(f"metadata lineage {name} filename does not match the report input")

        coordinate = data.get("coordinate")
        if not isinstance(coordinate, dict):
            raise ValueError("metadata coordinate is missing")
        instrument = coordinate.get("instrument_key")
        if not isinstance(instrument, str) or instrument.strip().upper() != "BTCUSDT":
            raise ValueError("metadata instrument coordinate is not BTCUSDT")
        if coordinate.get("source") != "Binance":
            raise ValueError("metadata provider coordinate must use the Binance broker identity")
        if int(coordinate.get("interval_seconds", -1)) != int(
            primary.data["encoding_config"]["interval_seconds"]
        ):
            raise ValueError("metadata interval differs from the frozen corpus")
        row_size = _finite(coordinate.get("row_size"), "metadata row size", minimum=0.0)
        if not math.isclose(
            row_size,
            float(primary.data["encoding_config"]["row_size"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("metadata row size differs from the frozen corpus")

        parity = data.get("parity")
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise ValueError("PyTorch-to-ONNX Runtime parity is absent or failed")
        if parity.get("provider") != "CPUExecutionProvider":
            raise ValueError("parity was not measured with ONNX Runtime CPUExecutionProvider")
        maximum_error = _finite(
            parity.get("maximum_absolute_error"), "maximum parity error", minimum=0.0
        )
        tolerance = _finite(parity.get("tolerance"), "parity tolerance", minimum=0.0)
        if tolerance <= 0 or maximum_error > tolerance:
            raise ValueError("PyTorch-to-ONNX Runtime parity exceeds tolerance")
        sample = parity.get("sample")
        if not isinstance(sample, dict):
            raise ValueError("representative parity sample profile is missing")
        sample_width = _positive_int(sample.get("window_max_rows"), "parity sample width")
        if sample_width > int(primary.data["encoding_config"]["max_rows"]):
            raise ValueError("representative parity sample exceeds the row safety bound")
        if int(sample.get("valid_windows", -1)) != int(corpus.manifest["valid_windows"]):
            raise ValueError("parity sample profile does not cover the frozen valid-window set")
        _finite(sample.get("p95_window_max_rows"), "p95 parity sample width", minimum=0.0)
        latency = parity.get("cpu_latency_ms")
        if not isinstance(latency, dict) or int(latency.get("runs", -1)) != 50:
            raise ValueError("CPU latency evidence must contain exactly 50 timed runs")
        p50 = _finite(latency.get("p50"), "CPU latency p50", minimum=0.0)
        p95 = _finite(latency.get("p95"), "CPU latency p95", minimum=0.0)
        p99 = _finite(latency.get("p99"), "CPU latency p99", minimum=0.0)
        if not 0 <= p50 <= p95 <= p99:
            raise ValueError("CPU latency percentiles are inconsistent")
        if data.get("runtime_status") != "shadow-only-research":
            raise ValueError("deployment metadata is not shadow-only research")
        if data.get("creates_order_signals") is not False:
            raise ValueError("deployment metadata does not prohibit order signals")

        onnx_path = _require_file(metadata_path.parent / "fdt-n.onnx", "selected ONNX model")
        onnx_artifact = _artifact("Selected ONNX model", onnx_path, seed=primary.seed)
        if data.get("onnx_sha256") != onnx_artifact.sha256:
            raise ValueError("metadata ONNX hash does not match the selected graph")
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exception:
        raise PendingResultsError(
            "FDT-N PDF not generated: selected deployment evidence is missing, incomplete, or "
            f"does not bind to the validation-selected run: {exception}."
        ) from exception
    return DeploymentRecord(
        data=data,
        metadata=_artifact("Selected model metadata", metadata_path, seed=primary.seed),
        onnx=onnx_artifact,
    )


def _validate_corpus(
    manifest_path: Path,
    dataset_path: Path,
    manifest: dict[str, Any],
) -> CorpusRecord:
    if manifest.get("schema_version") != 2:
        raise ValueError("dataset manifest must use exact-row schema version 2")
    if manifest.get("source") != "Binance Data Vision spot daily individual trades":
        raise ValueError("dataset source must be live-parity Binance individual trades")
    if tuple(manifest.get("target_names", ())) != TARGET_NAMES:
        raise ValueError("dataset target order does not match FDT-N")
    if tuple(manifest.get("row_feature_names", ())) != ROW_FEATURE_NAMES:
        raise ValueError("dataset row-feature order does not match FDT-N")
    if tuple(manifest.get("bar_feature_names", ())) != BAR_FEATURE_NAMES:
        raise ValueError("dataset bar-feature order does not match FDT-N")
    sources = manifest.get("source_files")
    if not isinstance(sources, list) or len(sources) != 14:
        raise ValueError("dataset manifest must identify all 14 source archives")
    archive_bytes = 0
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"source archive {index} is malformed")
        archive_bytes += _positive_int(source.get("bytes"), f"source archive {index} bytes")
        digest = source.get("sha256")
        if not isinstance(digest, str) or not _is_sha256(digest):
            raise ValueError(f"source archive {index} does not have a SHA-256")

    try:
        with np.load(dataset_path, allow_pickle=False) as encoded:
            if int(encoded["schema_version"]) != 2:
                raise ValueError("encoded dataset is not exact-row schema version 2")
            config = json.loads(str(encoded["config"]))
            offsets = np.asarray(encoded["row_offsets"], dtype=np.int64)
            timestamps = np.asarray(encoded["timestamps_ms"], dtype=np.int64)
            row_features = np.asarray(encoded["row_features"], dtype=np.float64)
            bar_values = np.asarray(encoded["bar_values"], dtype=np.float64)
    except (KeyError, OSError, ValueError) as exception:
        raise ValueError(f"encoded dataset is incomplete: {exception}") from exception
    counts = np.diff(offsets)
    bars = _positive_int(manifest.get("bars"), "manifest bars")
    rows = _positive_int(manifest.get("encoded_price_rows"), "manifest rows")
    if len(timestamps) != bars or len(counts) != bars or int(offsets[-1]) != rows:
        raise ValueError("encoded dataset counts do not match the manifest")
    if row_features.shape != (rows, len(ROW_FEATURE_NAMES)):
        raise ValueError("encoded row-feature tensor does not match the frozen schema")
    if bar_values.shape != (bars, 19):
        raise ValueError("encoded bar-value tensor does not match the frozen schema")
    if not np.isfinite(row_features).all() or not np.isfinite(bar_values).all():
        raise ValueError("encoded dataset contains non-finite values")
    if np.any(counts <= 0) or int(counts.max()) != int(manifest["maximum_rows_per_bar"]):
        raise ValueError("encoded row offsets do not match the manifest")
    if config != manifest.get("encoding"):
        raise ValueError("encoded dataset configuration does not match the manifest")
    if config.get("max_rows") != 1024 or config.get("quantity_scale") != 1000.0:
        raise ValueError("encoded dataset is not the final exact-row live-parity corpus")
    if archive_bytes != 374_843_570:
        raise ValueError("source archive byte ledger does not match the frozen corpus")
    percentiles = tuple(float(value) for value in np.percentile(counts, (50, 95, 99)))
    row_percentiles = (percentiles[0], percentiles[1], percentiles[2], float(counts.max()))
    clipping_audit = _compute_clipping_audit(row_features, bar_values, timestamps, config)
    if clipping_audit[0].observations != rows:
        raise ValueError("row clipping audit does not cover every encoded price row")
    valid_windows = _positive_int(manifest.get("valid_windows"), "manifest valid windows")
    bar_audit = next(item for item in clipping_audit if item.scope == "bar_input")
    target_audit = next(item for item in clipping_audit if item.scope == "target")
    if bar_audit.observations != valid_windows * int(config["lookback_bars"]):
        raise ValueError("input clipping audit does not cover every valid context window")
    if target_audit.observations != valid_windows * int(config["horizon_bars"]):
        raise ValueError("target clipping audit does not cover every valid target window")
    return CorpusRecord(
        manifest_path=manifest_path,
        dataset_path=dataset_path,
        manifest=manifest,
        manifest_artifact=_artifact("Dataset manifest", manifest_path),
        dataset_artifact=_artifact("Encoded dataset", dataset_path),
        row_percentiles=row_percentiles,
        archive_bytes=archive_bytes,
        clipping_audit=clipping_audit,
    )


def _compute_clipping_audit(
    row_features: np.ndarray,
    bar_values: np.ndarray,
    timestamps: np.ndarray,
    config: dict[str, Any],
) -> tuple[BoundAudit, ...]:
    """Measure raw-coordinate clipping and the one irrecoverable encoded-row saturation."""

    lookback = _positive_int(config.get("lookback_bars"), "lookback bars")
    horizons = _positive_int(config.get("horizon_bars"), "horizon bars")
    interval_ms = _positive_int(config.get("interval_seconds"), "interval seconds") * 1_000
    price_scale = _finite(config.get("price_scale_ticks"), "price scale", minimum=0.0)
    gap_scale = _finite(config.get("gap_scale_ticks"), "gap scale", minimum=0.0)
    volume_scale = _finite(config.get("log_volume_scale"), "log-volume scale", minimum=0.0)
    cvd_scale = _finite(
        config.get("cumulative_delta_scale"), "cumulative-delta scale", minimum=0.0
    )
    if min(price_scale, gap_scale, volume_scale, cvd_scale) <= 0:
        raise ValueError("clipping audit scales must be strictly positive")

    candidates = np.arange(lookback - 1, len(timestamps) - horizons, dtype=np.int64)
    valid: list[int] = []
    for context_end in candidates:
        start = int(context_end) - lookback + 1
        stop = int(context_end) + horizons + 1
        if np.all(np.diff(timestamps[start:stop]) == interval_ms):
            valid.append(int(context_end))
    ends = np.asarray(valid, dtype=np.int64)
    if not len(ends):
        raise ValueError("clipping audit found no continuous context-plus-target windows")

    audits: list[BoundAudit] = [
        BoundAudit(
            scope="row_input",
            feature=ROW_FEATURE_NAMES[0],
            lower_bound=-2.0,
            lower_hits=int(np.count_nonzero(row_features[:, 0] <= -2.0)),
            upper_bound=2.0,
            upper_hits=int(np.count_nonzero(row_features[:, 0] >= 2.0)),
            observations=int(row_features.shape[0]),
            method="encoded_boundary_saturation",
        )
    ]

    starts = ends - lookback + 1
    anchors = bar_values[ends, 0]
    reference_width = min(16, lookback)
    reference_logs = np.stack(
        [bar_values[ends - offset, 10] for offset in range(reference_width)], axis=1
    )
    reference_logs = np.median(reference_logs, axis=1)
    reference_volumes = np.expm1(reference_logs)
    base_cumulative = bar_values[starts, 14]
    row_count_denominator = math.log1p(_positive_int(config.get("max_rows"), "maximum rows"))

    bar_bounds: tuple[tuple[str, float | None, float | None], ...] = (
        (BAR_FEATURE_NAMES[0], -2.0, 2.0),
        (BAR_FEATURE_NAMES[1], -2.0, 2.0),
        (BAR_FEATURE_NAMES[2], -2.0, 2.0),
        (BAR_FEATURE_NAMES[3], -2.0, 2.0),
        (BAR_FEATURE_NAMES[4], -2.0, 2.0),
        (BAR_FEATURE_NAMES[5], -2.0, 2.0),
        (BAR_FEATURE_NAMES[6], None, 2.0),
        (BAR_FEATURE_NAMES[7], None, 2.0),
        (BAR_FEATURE_NAMES[8], None, 2.0),
        (BAR_FEATURE_NAMES[9], None, 2.0),
        (BAR_FEATURE_NAMES[10], -2.0, 2.0),
        (BAR_FEATURE_NAMES[14], -2.0, 2.0),
        (BAR_FEATURE_NAMES[15], None, 1.0),
        (BAR_FEATURE_NAMES[16], None, 1.0),
        (BAR_FEATURE_NAMES[17], None, 1.0),
    )
    bar_hits = {name: [0, 0, 0] for name, _, _ in bar_bounds}
    for offset in range(lookback):
        values = bar_values[starts + offset]
        row_counts = np.maximum(values[:, 17], 1.0)
        raw_values = (
            (values[:, 0] - anchors) / price_scale,
            (values[:, 1] - values[:, 0]) / gap_scale,
            (values[:, 2] - values[:, 0]) / gap_scale,
            (values[:, 3] - values[:, 0]) / gap_scale,
            (values[:, 4] - values[:, 0]) / gap_scale,
            (values[:, 5] - values[:, 0]) / gap_scale,
            (values[:, 0] - values[:, 6]) / gap_scale,
            (values[:, 7] - values[:, 0]) / gap_scale,
            (values[:, 0] - values[:, 8]) / gap_scale,
            (values[:, 9] - values[:, 0]) / gap_scale,
            (values[:, 10] - reference_logs) / volume_scale,
            (
                (values[:, 14] - base_cumulative)
                / np.maximum(reference_volumes * lookback, 1.0)
                / cvd_scale
            ),
            values[:, 15] / row_counts,
            values[:, 16] / row_counts,
            np.log1p(values[:, 17]) / row_count_denominator,
        )
        for (name, lower, upper), raw in zip(bar_bounds, raw_values, strict=True):
            counter = bar_hits[name]
            counter[0] += int(np.count_nonzero(raw < lower)) if lower is not None else 0
            counter[1] += int(np.count_nonzero(raw > upper)) if upper is not None else 0
            counter[2] += int(raw.size)
    for name, lower, upper in bar_bounds:
        lower_hits, upper_hits, observations = bar_hits[name]
        audits.append(
            BoundAudit(
                scope="bar_input",
                feature=name,
                lower_bound=lower,
                lower_hits=lower_hits if lower is not None else None,
                upper_bound=upper,
                upper_hits=upper_hits if upper is not None else None,
                observations=observations,
                method="raw_pre_clip",
            )
        )

    target_bounds: tuple[tuple[str, int, float, float], ...] = (
        (TARGET_NAMES[0], 0, -2.0, 2.0),
        (TARGET_NAMES[1], 6, -2.0, 2.0),
        (TARGET_NAMES[2], 7, -2.0, 2.0),
        (TARGET_NAMES[3], 1, -2.0, 2.0),
        (TARGET_NAMES[4], 2, -2.0, 2.0),
        (TARGET_NAMES[5], 10, -2.0, 2.0),
        (TARGET_NAMES[6], 13, -1.0, 1.0),
    )
    target_hits = {name: [0, 0, 0] for name, _, _, _ in target_bounds}
    for horizon in range(1, horizons + 1):
        values = bar_values[ends + horizon]
        raw_values = (
            (values[:, 0] - anchors) / price_scale,
            (values[:, 6] - anchors) / price_scale,
            (values[:, 7] - anchors) / price_scale,
            (values[:, 1] - anchors) / price_scale,
            (values[:, 2] - anchors) / price_scale,
            (values[:, 10] - reference_logs) / volume_scale,
            values[:, 13],
        )
        for (name, _, lower, upper), raw in zip(target_bounds, raw_values, strict=True):
            valid_label = np.ones(raw.shape, dtype=np.bool_)
            if name == "buy_poc_offset":
                valid_label = values[:, 11] > 0
            elif name == "sell_poc_offset":
                valid_label = values[:, 12] > 0
            observed = raw[valid_label]
            counter = target_hits[name]
            counter[0] += int(np.count_nonzero(observed < lower))
            counter[1] += int(np.count_nonzero(observed > upper))
            counter[2] += int(observed.size)
    for name, _, lower, upper in target_bounds:
        lower_hits, upper_hits, observations = target_hits[name]
        audits.append(
            BoundAudit(
                scope="target",
                feature=name,
                lower_bound=lower,
                lower_hits=lower_hits,
                upper_bound=upper,
                upper_hits=upper_hits,
                observations=observations,
                method="raw_pre_clip",
            )
        )
    return tuple(audits)


def _validate_run(
    path: Path,
    data: dict[str, Any],
    corpus: CorpusRecord,
    repo_root: Path,
) -> RunRecord:
    try:
        if data.get("schema_version") != 2:
            raise PendingResultsError("requires complete schema-v2 evaluation output")
        if data.get("architecture") != "FDT-N":
            raise ValueError("architecture is not FDT-N")
        _reject_pending_values(data, "results")
        seed = _positive_int(data["train_config"]["seed"], "training seed", allow_zero=True)
        if data.get("encoding_config") != corpus.manifest.get("encoding"):
            raise ValueError("encoding configuration differs from the frozen dataset")
        _positive_int(data.get("parameter_count"), "parameter count")
        horizons = int(data["encoding_config"]["horizon_bars"])
        _validate_splits(data["splits"], corpus.manifest)
        _validate_distribution(data["validation"], horizons, "validation")
        _validate_distribution(data["test"], horizons, "test")
        if int(data["validation"]["examples"]) != int(data["splits"]["validation_examples"]):
            raise ValueError("validation metric count differs from split count")
        if int(data["test"]["examples"]) != int(data["splits"]["test_examples"]):
            raise ValueError("test metric count differs from split count")
        _validate_baselines(data["baselines"], horizons, data["splits"])
        _validate_comparisons(data["comparisons"], horizons)
        _validate_history(data)
        validation_loss = _finite(data["validation"]["loss"], "validation loss", minimum=0.0)
        checkpoint_path = _resolve_recorded_path(data["checkpoint"], path.parent, repo_root)
        split_path = _require_file(path.parent / "splits.npz", "split indices")
        ridge_path = _resolve_recorded_path(
            data["baselines"]["direct_ridge"]["artifact"], path.parent, repo_root
        )
        _validate_split_artifact(split_path, data["splits"])
    except PendingResultsError as exception:
        raise PendingResultsError(
            f"FDT-N PDF not generated: {path} is not a final evaluation bundle; {exception}. "
            "Final metrics remain pending."
        ) from exception
    except (KeyError, TypeError, ValueError) as exception:
        raise PendingResultsError(
            f"FDT-N PDF not generated: {path} is incomplete or incompatible: {exception}. "
            "Final metrics remain pending."
        ) from exception

    return RunRecord(
        path=path,
        data=data,
        seed=seed,
        validation_loss=validation_loss,
        checkpoint=_artifact("Checkpoint", checkpoint_path, seed=seed),
        splits=_artifact("Split indices", split_path, seed=seed),
        ridge=_artifact("Direct ridge", ridge_path, seed=seed),
    )


def _validate_run_set(runs: Sequence[RunRecord]) -> None:
    seeds = [run.seed for run in runs]
    if len(set(seeds)) != len(seeds):
        raise ValueError("multi-seed report inputs contain a duplicate seed")
    reference = runs[0]
    reference_splits = _load_splits(reference.splits.path)
    reference_training = {
        key: value for key, value in reference.data["train_config"].items() if key != "seed"
    }
    for run in runs[1:]:
        for key in ("encoding_config", "model_config", "parameter_count", "splits"):
            if run.data[key] != reference.data[key]:
                raise ValueError(f"seed {run.seed} has a different {key}")
        candidate_training = {
            key: value for key, value in run.data["train_config"].items() if key != "seed"
        }
        if candidate_training != reference_training:
            raise ValueError(f"seed {run.seed} has a different non-seed training configuration")
        candidate_splits = _load_splits(run.splits.path)
        for name in ("train", "validation", "test"):
            if not np.array_equal(candidate_splits[name], reference_splits[name]):
                raise ValueError(f"seed {run.seed} uses different {name} indices")
        if (
            run.data["baselines"]["persistence"]["test"]
            != reference.data["baselines"]["persistence"]["test"]
        ):
            raise ValueError(f"seed {run.seed} has different persistence baseline metrics")
        direct = run.data["baselines"]["direct_ridge"]
        reference_direct = reference.data["baselines"]["direct_ridge"]
        for key in ("feature_strategy", "feature_count", "alpha", "training_examples", "test"):
            if direct[key] != reference_direct[key]:
                raise ValueError(f"seed {run.seed} has different direct-ridge {key}")


def _validate_splits(splits: dict[str, Any], manifest: dict[str, Any]) -> None:
    purge = _positive_int(splits.get("purge_bars"), "purge bars")
    valid = _positive_int(manifest.get("valid_windows"), "valid windows")
    train_stop = int(valid * 0.70)
    validation_stop = int(valid * 0.85)
    expected = {
        "train_examples": max(0, train_stop - purge),
        "validation_examples": max(0, validation_stop - purge - (train_stop + purge)),
        "test_examples": max(0, valid - (validation_stop + purge)),
    }
    for name, value in expected.items():
        if int(splits.get(name, -1)) != value:
            raise ValueError(f"{name} does not match the frozen purged split")


def _validate_distribution(section: dict[str, Any], horizons: int, label: str) -> None:
    _positive_int(section.get("examples"), f"{label} examples")
    _finite(section.get("loss"), f"{label} loss", minimum=0.0)
    cells = section.get("by_horizon_target")
    if not isinstance(cells, dict) or set(cells) != {str(i) for i in range(1, horizons + 1)}:
        raise ValueError(f"{label} does not contain every horizon")
    for horizon in range(1, horizons + 1):
        targets = cells[str(horizon)]
        if not isinstance(targets, dict) or set(targets) != set(TARGET_NAMES):
            raise ValueError(f"{label} horizon {horizon} target order is incomplete")
        for target_name in TARGET_NAMES:
            metric = targets[target_name]
            _positive_int(metric.get("examples"), f"{label} {horizon} {target_name} examples")
            pinball = metric.get("pinball_normalized")
            if not isinstance(pinball, dict) or set(pinball) != {"q10", "q50", "q90"}:
                raise ValueError(f"{label} {horizon} {target_name} lacks quantile scores")
            for quantile in ("q10", "q50", "q90"):
                _finite(pinball[quantile], f"{label} {target_name} {quantile}", minimum=0.0)
            _finite(metric.get("median_mae_normalized"), "q50-forecast MAE", minimum=0.0)
            _finite(metric.get("median_bias_normalized"), "median bias")
            _finite(metric.get("central_80_coverage"), "coverage", minimum=0.0, maximum=1.0)
            _finite(metric.get("mean_interval_width_normalized"), "interval width", minimum=0.0)
            if target_name == "poc_offset":
                _finite(
                    metric.get("poc_directional_hit_rate"),
                    "POC directional hit rate",
                    minimum=0.0,
                    maximum=1.0,
                )
    pooled = section.get("pooled")
    if not isinstance(pooled, dict):
        raise ValueError(f"{label} pooled metrics are missing")
    for quantile in ("q10", "q50", "q90"):
        _finite(pooled["pinball_normalized"][quantile], f"{label} pooled {quantile}", minimum=0.0)
    for key, bounds in (
        ("median_mae_normalized", (0.0, None)),
        ("median_bias_normalized", (None, None)),
        ("central_80_coverage", (0.0, 1.0)),
        ("mean_interval_width_normalized", (0.0, None)),
        ("poc_directional_hit_rate", (0.0, 1.0)),
    ):
        _finite(pooled[key], f"{label} pooled {key}", minimum=bounds[0], maximum=bounds[1])
    for key in ("median_mae_normalized", "central_80_coverage"):
        values = section.get(key)
        if not isinstance(values, dict) or set(values) != set(TARGET_NAMES):
            raise ValueError(f"{label} pooled target {key} is incomplete")
        for target_name, value in values.items():
            _finite(
                value,
                f"{label} {target_name} {key}",
                minimum=0.0,
                maximum=1.0 if key.endswith("coverage") else None,
            )


def _validate_baselines(baselines: dict[str, Any], horizons: int, splits: dict[str, Any]) -> None:
    if not isinstance(baselines, dict) or set(baselines) != {"persistence", "direct_ridge"}:
        raise ValueError("required point baselines are missing")
    for name in ("persistence", "direct_ridge"):
        baseline = baselines[name]
        if baseline.get("target_space") != "encoded_normalized_targets":
            raise ValueError(f"{name} baseline uses an unexpected target space")
        _validate_point_metrics(baseline["test"], horizons, name)
    direct = baselines["direct_ridge"]
    if direct.get("feature_strategy") != "latest_mean_trend_bar_aggregates":
        raise ValueError("direct ridge feature strategy is unexpected")
    _positive_int(direct.get("feature_count"), "direct ridge feature count")
    _finite(direct.get("alpha"), "direct ridge alpha", minimum=0.0, strict_minimum=True)
    if int(direct.get("training_examples", -1)) != int(splits["train_examples"]):
        raise ValueError("direct ridge was not fitted on the training split only")


def _validate_point_metrics(section: dict[str, Any], horizons: int, label: str) -> None:
    _positive_int(section.get("examples"), f"{label} examples")
    cells = section.get("by_horizon_target")
    if not isinstance(cells, dict) or set(cells) != {str(i) for i in range(1, horizons + 1)}:
        raise ValueError(f"{label} baseline does not contain every horizon")
    for horizon in range(1, horizons + 1):
        targets = cells[str(horizon)]
        if not isinstance(targets, dict) or set(targets) != set(TARGET_NAMES):
            raise ValueError(f"{label} baseline horizon {horizon} is incomplete")
        for target_name in TARGET_NAMES:
            metric = targets[target_name]
            _positive_int(metric.get("examples"), f"{label} {target_name} examples")
            _finite(metric.get("median_mae_normalized"), "baseline MAE", minimum=0.0)
            _finite(metric.get("median_bias_normalized"), "baseline bias")
            if target_name == "poc_offset":
                _finite(
                    metric.get("poc_directional_hit_rate"),
                    "baseline POC hit rate",
                    minimum=0.0,
                    maximum=1.0,
                )
    for key in ("median_mae_normalized", "median_bias_normalized"):
        values = section.get(key)
        if not isinstance(values, dict) or set(values) != set(TARGET_NAMES):
            raise ValueError(f"{label} baseline {key} is incomplete")
        for value in values.values():
            _finite(
                value,
                f"{label} baseline {key}",
                minimum=0.0 if key.endswith("mae_normalized") else None,
            )


def _validate_comparisons(comparisons: dict[str, Any], horizons: int) -> None:
    required = {"model_vs_persistence", "model_vs_direct_ridge"}
    if not isinstance(comparisons, dict) or set(comparisons) != required:
        raise ValueError("paired model-baseline comparisons are missing")
    for name, comparison in comparisons.items():
        if comparison.get("difference") != "model_median_mae_minus_baseline_mae":
            raise ValueError(f"{name} uses an unexpected effect definition")
        if comparison.get("block_definition") != "contiguous_240_example_blocks":
            raise ValueError(f"{name} does not use frozen four-hour blocks")
        if int(comparison.get("resamples", 0)) < 2_000:
            raise ValueError(f"{name} has fewer than 2,000 bootstrap resamples")
        _finite(comparison.get("confidence_level"), "confidence level", minimum=0.95, maximum=0.95)
        _positive_int(comparison.get("blocks"), "bootstrap blocks")
        cells = comparison.get("by_horizon_target")
        if not isinstance(cells, dict) or set(cells) != {str(i) for i in range(1, horizons + 1)}:
            raise ValueError(f"{name} comparison horizons are incomplete")
        for horizon in range(1, horizons + 1):
            if set(cells[str(horizon)]) != set(TARGET_NAMES):
                raise ValueError(f"{name} horizon {horizon} targets are incomplete")
            for target_name in TARGET_NAMES:
                metric = cells[str(horizon)][target_name]
                point = _finite(metric.get("mae_difference_normalized"), "MAE difference")
                lower = _finite(metric.get("ci_95_lower"), "MAE lower interval")
                upper = _finite(metric.get("ci_95_upper"), "MAE upper interval")
                if not lower <= point <= upper:
                    raise ValueError(f"{name} interval does not contain its point estimate")


def _validate_history(data: dict[str, Any]) -> None:
    history = data.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("training history is missing")
    epochs = []
    for row in history:
        epoch = _positive_int(row.get("epoch"), "history epoch")
        epochs.append(epoch)
        _finite(row.get("train_loss"), "history train loss", minimum=0.0)
        _finite(row.get("validation_loss"), "history validation loss", minimum=0.0)
        _finite(row.get("learning_rate"), "history learning rate", minimum=0.0)
    if epochs != list(range(1, len(history) + 1)):
        raise ValueError("training history epochs are not contiguous")
    if int(data.get("epochs_completed", -1)) != len(history):
        raise ValueError("epochs_completed differs from training history")
    if int(data.get("best_epoch", -1)) not in epochs:
        raise ValueError("best epoch is outside training history")


def _validate_split_artifact(path: Path, splits: dict[str, Any]) -> None:
    values = _load_splits(path)
    for name, key in (
        ("train", "train_examples"),
        ("validation", "validation_examples"),
        ("test", "test_examples"),
    ):
        array = values[name]
        if array.ndim != 1 or len(array) != int(splits[key]) or np.any(np.diff(array) <= 0):
            raise ValueError(f"{name} split artifact is inconsistent")


def _load_splits(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {
            name: np.asarray(bundle[name], dtype=np.int64)
            for name in ("train", "validation", "test")
        }


def _build_charts(bundle: ReportBundle, workdir: Path) -> dict[str, Path]:
    paths = {
        "training": workdir / "training-history.png",
        "stability": workdir / "seed-stability.png",
        "pinball": workdir / "primary-pinball.png",
        "coverage_width": workdir / "primary-coverage-width.png",
        "comparison": workdir / "primary-mae-comparison.png",
        "poc_intervals": workdir / "primary-poc-intervals.png",
    }
    _plot_training(bundle, paths["training"])
    _plot_stability(bundle, paths["stability"])
    _plot_pinball(bundle.primary, paths["pinball"])
    _plot_coverage_width(bundle.primary, paths["coverage_width"])
    _plot_comparison(bundle.primary, paths["comparison"])
    _plot_poc_intervals(bundle.primary, paths["poc_intervals"])
    return paths


def _plot_training(bundle: ReportBundle, path: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))
    for run in bundle.runs:
        history = run.data["history"]
        epoch = [int(row["epoch"]) for row in history]
        validation = [row["validation_loss"] for row in history]
        width = 2.4 if run.seed == bundle.primary.seed else 1.2
        alpha = 1.0 if run.seed == bundle.primary.seed else 0.62
        axes[0].plot(epoch, validation, linewidth=width, alpha=alpha, label=f"seed {run.seed}")
        axes[1].plot(
            epoch,
            [row["train_loss"] for row in history],
            linewidth=width,
            alpha=alpha,
            label=f"seed {run.seed}",
        )
    axes[0].set_title("Validation weighted pinball")
    axes[1].set_title("Training weighted pinball")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Normalized loss")
        axis.legend(frameon=False, ncol=2, fontsize=8)
        _base_axis(axis)
    fig.suptitle(
        "Optimization history - primary seed shown with heavier line",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    _save_figure(fig, path)


def _plot_stability(bundle: ReportBundle, path: Path) -> None:
    _set_plot_style()
    rows = sorted(bundle.runs, key=lambda run: run.seed)
    labels = [str(run.seed) for run in rows]
    values = [
        [run.validation_loss for run in rows],
        [float(run.data["test"]["loss"]) for run in rows],
        [float(run.data["test"]["pooled"]["median_mae_normalized"]) for run in rows],
        [float(run.data["test"]["pooled"]["central_80_coverage"]) for run in rows],
    ]
    titles = (
        "Validation loss (selection only)",
        "Held-out test loss",
        "Held-out pooled q50-forecast MAE",
        "Held-out central 80% coverage",
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 6.0))
    for axis, series, title in zip(axes.flat, values, titles, strict=True):
        colors = [TEAL if run.seed == bundle.primary.seed else BLUE for run in rows]
        axis.bar(labels, series, color=colors, width=0.62)
        axis.set_title(title)
        axis.set_xlabel("Seed")
        _base_axis(axis)
        lower = min(series)
        upper = max(series)
        padding = max((upper - lower) * 0.35, abs(upper) * 0.08, 0.01)
        axis.set_ylim(max(0.0, lower - padding), upper + padding)
    fig.suptitle(
        "Every declared seed - teal marks validation-selected primary",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    _save_figure(fig, path)


def _plot_pinball(run: RunRecord, path: Path) -> None:
    matrices = []
    for quantile in ("q10", "q50", "q90"):
        matrices.append(
            _cell_matrix(
                run.data["test"],
                lambda cell, name=quantile: cell["pinball_normalized"][name],
            )
        )
    _set_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0), constrained_layout=True)
    maximum = max(float(np.max(matrix)) for matrix in matrices)
    for axis, matrix, quantile in zip(axes, matrices, ("q10", "q50", "q90"), strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=maximum)
        _heatmap_axis(axis, f"{quantile} pinball")
        _annotate_heatmap(axis, matrix)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Primary seed {run.seed} - held-out normalized pinball by horizon and target",
        fontsize=12,
    )
    _save_figure(fig, path)


def _plot_coverage_width(run: RunRecord, path: Path) -> None:
    coverage = _cell_matrix(run.data["test"], lambda cell: cell["central_80_coverage"])
    width = _cell_matrix(run.data["test"], lambda cell: cell["mean_interval_width_normalized"])
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True)
    distance = max(abs(float(coverage.min()) - 0.8), abs(float(coverage.max()) - 0.8), 0.05)
    coverage_image = axes[0].imshow(
        coverage,
        aspect="auto",
        cmap="RdBu",
        norm=TwoSlopeNorm(vmin=0.8 - distance, vcenter=0.8, vmax=0.8 + distance),
    )
    width_image = axes[1].imshow(width, aspect="auto", cmap="Purples", vmin=0.0)
    _heatmap_axis(axes[0], "Central 80% empirical coverage")
    _heatmap_axis(axes[1], "Mean normalized interval width")
    _annotate_heatmap(axes[0], coverage, percent=True)
    _annotate_heatmap(axes[1], width)
    fig.colorbar(coverage_image, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(width_image, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Primary seed {run.seed} - calibration and sharpness must be read together",
        fontsize=12,
    )
    _save_figure(fig, path)


def _plot_comparison(run: RunRecord, path: Path) -> None:
    names = ("model_vs_persistence", "model_vs_direct_ridge")
    titles = ("Model q50 MAE - persistence MAE", "Model q50 MAE - direct ridge MAE")
    matrices = [_comparison_matrix(run.data["comparisons"][name]) for name in names]
    limit = max(abs(float(matrix.min())) for matrix in matrices)
    limit = max(limit, max(abs(float(matrix.max())) for matrix in matrices), 1e-6)
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.0), constrained_layout=True)
    for axis, matrix, title in zip(axes, matrices, titles, strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
        _heatmap_axis(axis, title)
        _annotate_heatmap(axis, matrix, signed=True)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle(
        "Held-out point-baseline effects - negative values favor lower FDT-N q50-forecast MAE",
        fontsize=12,
    )
    _save_figure(fig, path)


def _plot_poc_intervals(run: RunRecord, path: Path) -> None:
    _set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4), sharey=True)
    for axis, name, title in (
        (axes[0], "model_vs_persistence", "Against persistence"),
        (axes[1], "model_vs_direct_ridge", "Against direct ridge"),
    ):
        comparison = run.data["comparisons"][name]["by_horizon_target"]
        points = []
        lower = []
        upper = []
        for horizon in range(1, 9):
            metric = comparison[str(horizon)]["poc_offset"]
            point = float(metric["mae_difference_normalized"])
            points.append(point)
            lower.append(point - float(metric["ci_95_lower"]))
            upper.append(float(metric["ci_95_upper"]) - point)
        y = np.arange(1, 9)
        axis.errorbar(
            points,
            y,
            xerr=np.asarray((lower, upper)),
            fmt="o",
            color=BLUE,
            ecolor=MUTED,
            capsize=3,
        )
        axis.axvline(0.0, color=RED, linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.set_xlabel("Normalized MAE difference")
        axis.set_yticks(y, [f"H{h}" for h in range(1, 9)])
        axis.invert_yaxis()
        _base_axis(axis)
    fig.suptitle(
        f"Primary seed {run.seed} total-POC paired four-hour sensitivity intervals",
        fontsize=12,
        x=0.02,
        ha="left",
    )
    _save_figure(fig, path)


def _cell_matrix(section: dict[str, Any], getter) -> np.ndarray:
    return np.asarray(
        [
            [getter(section["by_horizon_target"][str(horizon)][target]) for target in TARGET_NAMES]
            for horizon in range(1, 9)
        ],
        dtype=np.float64,
    )


def _comparison_matrix(comparison: dict[str, Any]) -> np.ndarray:
    return np.asarray(
        [
            [
                comparison["by_horizon_target"][str(horizon)][target]["mae_difference_normalized"]
                for target in TARGET_NAMES
            ]
            for horizon in range(1, 9)
        ],
        dtype=np.float64,
    )


def _heatmap_axis(axis, title: str) -> None:
    axis.set_title(title, fontsize=10)
    axis.set_xticks(range(len(TARGET_NAMES)), [TARGET_LABELS[name] for name in TARGET_NAMES])
    axis.tick_params(axis="x", rotation=48, labelsize=7)
    axis.set_yticks(range(8), [f"H{index}" for index in range(1, 9)])
    axis.tick_params(axis="y", labelsize=8)


def _annotate_heatmap(
    axis, matrix: np.ndarray, *, percent: bool = False, signed: bool = False
) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix[row, column])
            text = (
                f"{100 * value:.0f}%" if percent else f"{value:+.3f}" if signed else f"{value:.3f}"
            )
            axis.text(column, row, text, ha="center", va="center", fontsize=5.5, color=INK)


def _set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": WHITE,
            "figure.facecolor": PAPER,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
        }
    )


def _base_axis(axis) -> None:
    axis.grid(axis="y", linewidth=0.6, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def _save_figure(fig, path: Path) -> None:
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


class VisualReport:
    def __init__(
        self,
        output: Path,
        bundle: ReportBundle,
        charts: dict[str, Path],
        *,
        watermark: str | None,
    ) -> None:
        self.output = output
        self.bundle = bundle
        self.charts = charts
        self.watermark = watermark
        self.canvas = canvas.Canvas(
            str(output),
            pagesize=(PAGE_W, PAGE_H),
            pageCompression=1,
            invariant=1,
        )
        self.canvas.setTitle("FDT-N Volume-Footprint Transformer Research Paper")
        self.canvas.setAuthor("DaxAlgo Research")
        self.page_number = 0

    def render(self) -> None:
        self._cover()
        self._abstract_and_truth_labels()
        self._market_object()
        self._causal_task()
        self._data_representation()
        self._architecture()
        self._row_pool_and_fusion()
        self._temporal_decoder()
        self._targets_and_constraints()
        self._loss_and_joint_boundary()
        self._worked_example()
        self._training_protocol()
        self._baselines_ablations_and_chmt()
        self._evaluation_design()
        self._corpus()
        self._clipping_audit()
        self._training()
        self._stability_pages()
        self._probabilistic_scores()
        self._calibration()
        self._baseline_effects()
        self._poc_intervals()
        for run in sorted(self.bundle.runs, key=lambda item: item.seed):
            self._seed_scorecard(run)
            self._seed_cell_pages(run)
        self._deployment_evidence()
        self._runtime_safety()
        self._related_work()
        self._claims_and_conclusion()
        self._reproducibility_pages()
        self._limitations()
        self._references_pages()
        self.canvas.save()

    def _new_page(
        self,
        title: str,
        subtitle: str,
        *,
        dark: bool = False,
        kicker: str = "FDT-N RESEARCH",
    ) -> None:
        if self.page_number:
            self.canvas.showPage()
        self.page_number += 1
        self.canvas.setFillColor(HexColor(NAVY if dark else PAPER))
        self.canvas.rect(0, 0, PAGE_W, PAGE_H, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(TEAL if dark else BLUE))
        self.canvas.setFont(FONT_BOLD, 8)
        self.canvas.drawString(42, PAGE_H - 38, kicker)
        self.canvas.setFillColor(HexColor(WHITE if dark else INK))
        self.canvas.setFont(FONT_BOLD, 23)
        self.canvas.drawString(42, PAGE_H - 69, title)
        self.canvas.setFillColor(HexColor("#CBD5E1" if dark else MUTED))
        self.canvas.setFont(FONT_REGULAR, 9.5)
        self.canvas.drawString(42, PAGE_H - 88, subtitle)
        self._footer(dark)
        if self.watermark:
            self.canvas.saveState()
            self.canvas.setFillColor(HexColor("#CBD5E1"), alpha=0.18)
            self.canvas.setFont(FONT_BOLD, 31)
            self.canvas.translate(PAGE_W / 2, PAGE_H / 2)
            self.canvas.rotate(24)
            self.canvas.drawCentredString(0, 0, self.watermark)
            self.canvas.restoreState()

    def _footer(self, dark: bool) -> None:
        self.canvas.setStrokeColor(HexColor("#334155" if dark else GRID))
        self.canvas.line(42, 27, PAGE_W - 42, 27)
        self.canvas.setFillColor(HexColor("#94A3B8" if dark else MUTED))
        self.canvas.setFont(FONT_REGULAR, 7.2)
        self.canvas.drawString(
            42,
            14,
            "FDT-N research paper | Shadow-only artifact | No order, production, or capital claim",
        )
        self.canvas.drawRightString(PAGE_W - 42, 14, str(self.page_number))

    def _cover(self) -> None:
        primary = self.bundle.primary
        corpus = self.bundle.corpus.manifest
        self._new_page(
            "FDT-N volume-footprint transformer",
            "Self-contained research paper and frozen multi-seed evaluation | BTCUSDT spot",
            dark=True,
            kicker="DAXALGO RESEARCH // VERIFIED INPUT BUNDLE",
        )
        self.canvas.setFillColor(HexColor("#94A3B8"))
        self.canvas.setFont(FONT_REGULAR, 8.5)
        self.canvas.drawString(42, PAGE_H - 121, "EVIDENCE BOUNDARY")
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setFont(FONT_BOLD, 17)
        self.canvas.drawString(42, PAGE_H - 145, "FORECAST EVALUATION - NOT A TRADING SYSTEM")
        self._draw_wrapped(
            "The primary run is selected by validation loss only. Every supplied seed remains in "
            "the held-out scorecard; no test metric participates in run selection.",
            42,
            PAGE_H - 168,
            610,
            font=FONT_REGULAR,
            size=9.3,
            color="#CBD5E1",
            leading=13,
        )
        cards = [
            (str(len(self.bundle.runs)), "declared seeds", "all reported", BLUE),
            (str(primary.seed), "primary seed", "validation-only choice", TEAL),
            (f"{primary.validation_loss:.5f}", "validation loss", "selection criterion", PURPLE),
            (
                f"{primary.data['test']['loss']:.5f}",
                "held-out test loss",
                "reported after selection",
                AMBER,
            ),
        ]
        self._metric_cards(cards, y=265, dark=True)
        start = _utc_date(corpus["first_timestamp_ms"])
        end = _utc_date(corpus["last_timestamp_ms"])
        self.canvas.setFillColor(HexColor("#CBD5E1"))
        self.canvas.setFont(FONT_REGULAR, 8)
        self.canvas.drawString(42, 88, f"Frozen corpus: {start} through {end} UTC")
        self.canvas.drawString(42, 72, f"Results SHA-256: {_sha256(primary.path)}")

    def _abstract_and_truth_labels(self) -> None:
        self._new_page(
            "Abstract and evidence grammar",
            "Architecture facts, hypotheses, and measured results remain visibly separate.",
            kicker="ABSTRACT",
        )
        left_x, right_x, width = 42, 428, 368
        y = PAGE_H - 122
        y = self._narrative_block(
            "OBSERVED",
            "The market object",
            "A sealed footprint bar retains executed buy and sell volume at every traded price "
            "row, plus POCs, centroids, range, value area, volume, delta, CVD, imbalance, quality, "
            "and UTC time. The frozen corpus identity is measured independently of forecast skill.",
            left_x,
            y,
            width,
            TEAL,
        )
        y = self._narrative_block(
            "IMPLEMENTED",
            "The forecaster",
            "FDT-N embeds every accepted row, pools each bar with four learned queries, fuses the "
            "row summary with 23 aggregates, processes 64 sealed bars, and directly decodes eight "
            "future horizons. Each horizon emits q10/q50/q90 for seven target families.",
            left_x,
            y - 8,
            width,
            BLUE,
        )
        y = self._narrative_block(
            "DESIGN HYPOTHESIS",
            "Why test this hierarchy",
            "Full auction shape may preserve information discarded by aggregates; learned query "
            "pooling may summarize variable row sets efficiently; direct horizon queries may avoid "
            "autoregressive forecast-as-input error. None of these motivations is a result.",
            left_x,
            y - 8,
            width,
            PURPLE,
        )
        self._narrative_block(
            "EMPIRICAL RESULT",
            "What this edition measures",
            f"The report binds {len(self.bundle.runs)} declared seed results to one frozen dataset, "
            "selects the primary only by validation loss, publishes every held-out seed, and binds "
            "the selected checkpoint to a passed ONNX Runtime parity artifact. It remains a short, "
            "single-instrument development study.",
            left_x,
            y - 8,
            width,
            AMBER,
        )

        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 12)
        self.canvas.drawString(right_x, PAGE_H - 122, "Research questions")
        questions = [
            "RQ1 - Do exact row distributions improve proper held-out scores over aggregate-only inputs?",
            "RQ2 - Do four learned queries improve on simple pooling or convolutional structure?",
            "RQ3 - Do parallel horizon queries improve on separate heads and naive causal baselines?",
            "RQ4 - Are q10/q50/q90 calibrated overall and by horizon, target, and regime?",
            "RQ5 - Does any gain survive other symbols, venues, regimes, and runtime constraints?",
        ]
        cursor = self._bullet_list(questions, right_x, PAGE_H - 149, width, size=8.4, leading=12)
        self.canvas.setFillColor(HexColor(LIGHT_AMBER))
        self.canvas.roundRect(right_x, cursor - 127, width, 114, 8, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(AMBER))
        self.canvas.setFont(FONT_BOLD, 9)
        self.canvas.drawString(right_x + 14, cursor - 35, "SAFETY AND CLAIM BOUNDARY")
        self._draw_wrapped(
            "FDT-N is a research forecaster and chart-visualization candidate. It creates no order, "
            "position, sizing instruction, or execution signal. Forecast accuracy is not economic "
            "value; fees, latency, fills, impact, and a frozen decision policy require a separate study.",
            right_x + 14,
            cursor - 55,
            width - 28,
            size=8.3,
            leading=12,
            color=INK,
        )

    def _market_object(self) -> None:
        self._new_page(
            "Volume footprints from first principles",
            "Executed-flow observations describe an auction; they are not directional conclusions.",
            kicker="BEGINNER PRIMER",
        )
        self._narrative_block(
            "OBSERVED",
            "One price row",
            "At price row j inside sealed bar t, p is snapped price, v+ is buy-initiated "
            "ask-lifting volume, and v- is sell-initiated bid-hitting volume. Bid/ask imbalance and "
            "zero-side flags are explicit data fields, not inferred tiny volumes.",
            42,
            PAGE_H - 122,
            355,
            TEAL,
        )
        self._equation_box(
            "ROW IDENTITIES",
            [
                "r[t,j] = (p[t,j], v+[t,j], v-[t,j], imbalance flags, zero flags)",
                "total row volume: v[t,j] = v+[t,j] + v-[t,j]",
                "row delta: Delta[t,j] = v+[t,j] - v-[t,j]",
            ],
            42,
            340,
            355,
            95,
        )
        self._narrative_block(
            "INTERPRETATION",
            "Delta is descriptive",
            "Positive delta says aggressive buyers executed more volume at that row or bar. It is "
            "not automatically bullish: price response, passive absorption, regime, and later flow "
            "are separate objects.",
            42,
            222,
            355,
            PURPLE,
        )

        self._narrative_block(
            "OBSERVED",
            "From rows to a sealed bar",
            "Bar buy and sell volumes sum the corresponding row volumes. Total POC is the price row "
            "with maximum total volume; buy and sell POCs use side volume. A centroid is a weighted "
            "mean, not an argmax. Value area is a deterministic contiguous region grown around POC.",
            430,
            PAGE_H - 122,
            366,
            TEAL,
        )
        self._equation_box(
            "BAR IDENTITIES",
            [
                "V+ = sum_j v+[j] ; V- = sum_j v-[j] ; V = V+ + V-",
                "Delta = V+ - V- ; POC = p[argmax_j(v[j])]",
                "centroid = sum_j(p[j] * v[j]) / V",
                "CVD[t] = CVD[t-1] + Delta[t]",
            ],
            430,
            336,
            366,
            112,
        )
        self._narrative_block(
            "DATA CONTRACT",
            "Policies are part of the coordinate",
            "Tie handling, price snapping, aggressor classification, value-area expansion, CVD reset, "
            "bar interval, row size, and feed-quality decisions must be deterministic and versioned. "
            "Changing any of them changes the meaning of the tensor.",
            430,
            202,
            366,
            AMBER,
        )

    def _causal_task(self) -> None:
        self._new_page(
            "Causal cutoff and forecasting task",
            "Only information available before the cutoff may influence inputs, normalization, or selection.",
            kicker="PROBLEM DEFINITION",
        )
        timeline = [
            ("Trade events", "event time", BLUE),
            ("Price rows", "classified flow", TEAL),
            ("Bar end", "interval sealed", PURPLE),
            ("Quality gate", "grace + completeness", AMBER),
            ("Forecast cutoff", "inputs frozen", RED),
            ("Bars +1 ... +8", "targets only", BLUE),
        ]
        x, top, box_w, gap = 42, PAGE_H - 180, 112, 15
        for index, (title, note, color) in enumerate(timeline):
            left = x + index * (box_w + gap)
            self._rounded_box(left, top, box_w, 68, title, note, color)
            if index < len(timeline) - 1:
                self.canvas.setStrokeColor(HexColor(MUTED))
                self.canvas.line(left + box_w, top + 34, left + box_w + gap - 3, top + 34)

        self._equation_box(
            "DIRECT MULTI-HORIZON MAP",
            [
                "Y_hat[t+1:t+8] = F_theta(B[t-63], ..., B[t])",
                "output shape = [batch, 8 horizons, 7 targets, 3 quantiles]",
                "no realized target and no earlier forecast is fed back into a later horizon",
            ],
            42,
            330,
            470,
            100,
        )
        target_rows = [
            ["Price", "total POC, low, high, buy POC, sell POC", "cutoff-POC offsets"],
            ["Activity", "log total volume", "context-median relative"],
            ["Flow", "delta fraction", "bounded side imbalance"],
        ]
        self._table(
            534,
            330,
            262,
            ["Family", "Targets", "Coordinate"],
            target_rows,
            [0.20, 0.48, 0.32],
            row_height=39,
            font_size=6.4,
        )
        self._narrative_block(
            "OPEN PROVENANCE GATE",
            "Exchange timestamps are not receive-time proof",
            "The historical archives establish exchange event chronology, not when a live system "
            "received and accepted each event. A production study must retain receive time, source "
            "epoch, sequence, completeness decision, and acceptance-policy identity. This report "
            "does not claim that the archive alone proves live receive-time causality.",
            42,
            185,
            754,
            AMBER,
        )

    def _data_representation(self) -> None:
        self._new_page(
            "Exact-row representation and feature contract",
            "Accepted bars keep every observed row; the 1,024-row ceiling is an explicit rejection bound.",
            kicker="ENCODING",
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Ragged storage without silent compaction",
            "All row vectors are stored in one flat tensor with offsets. A window pads only to its "
            "largest observed bar and supplies a mask. Wide windows receive smaller batches under the "
            "row-token budget; no accepted row is ranked away, merged, or silently truncated.",
            42,
            PAGE_H - 122,
            360,
            BLUE,
        )
        row_groups = [
            ["Price geometry", "1", "POC-relative row offset"],
            ["Side shares", "4", "buy, sell, total, delta shares"],
            ["Log side ratios", "2", "buy and sell activity"],
            ["Auction flags", "4", "bid/ask imbalance and zero-side flags"],
        ]
        self._table(
            42,
            336,
            360,
            ["Row group", "Count", "Meaning"],
            row_groups,
            [0.31, 0.14, 0.55],
            row_height=32,
            font_size=7.1,
        )
        self._narrative_block(
            "BOUNDARY",
            "Exact rows does not mean unlimited rows",
            "A bar with more than 1,024 observed rows fails the current schema. The frozen maximum is "
            "reported separately; acceptance on other symbols or volatility regimes is not inferred.",
            42,
            165,
            360,
            RED,
        )

        self._narrative_block(
            "IMPLEMENTED",
            "Twenty-three bar features",
            "The aggregate branch carries anchor-relative total and side POCs, centroids, low/high and "
            "value-area gaps, relative log volume, side and delta fractions, CVD change, stacked "
            "imbalance fractions, row count, feed quality, and cyclic minute/weekday coordinates.",
            430,
            PAGE_H - 122,
            366,
            BLUE,
        )
        bar_groups = [
            ["Price / value area", "10", "POCs, centroids, range, VA"],
            ["Volume / flow", "7", "volume, sides, delta, CVD, stacks"],
            ["Width / quality", "2", "row count and feed quality"],
            ["Calendar", "4", "minute and weekday sin/cos"],
        ]
        self._table(
            430,
            336,
            366,
            ["Bar group", "Count", "Meaning"],
            bar_groups,
            [0.29, 0.14, 0.57],
            row_height=32,
            font_size=7.1,
        )
        self._narrative_block(
            "CONTINUITY",
            "Every context-plus-label interval must be regular",
            "A candidate window is accepted only when all 64 context bars and all eight target bars "
            "are present at the configured one-minute cadence. The POC anchor and recent median "
            "log-volume reference are computed only from the context.",
            430,
            165,
            366,
            TEAL,
        )

    def _architecture(self) -> None:
        self._new_page(
            "Information contract and architecture",
            "Every arrow is causal; forecasts are parallel marginal quantiles, "
            "not a joint candle path.",
            kicker="METHOD",
        )
        boxes = [
            ("64 sealed bars", "ragged exact rows + 23 bar features", BLUE),
            ("Row MLP", "11 features per observed price row", TEAL),
            ("4 auction queries", "masked cross-attention within each bar", PURPLE),
            ("Gated fusion", "row summary + deterministic aggregates", AMBER),
            ("3 causal blocks", "pre-RMSNorm attention + SwiGLU", BLUE),
            ("8 horizon queries", "parallel decoder; no forecast feedback", TEAL),
            ("7 x 3 outputs", "q10/q50/q90 direct marginals", PURPLE),
        ]
        x, y, width, height, gap = 38, PAGE_H - 190, 100, 72, 13
        for index, (title, subtitle, color) in enumerate(boxes):
            left = x + index * (width + gap)
            self._rounded_box(left, y, width, height, title, subtitle, color)
            if index < len(boxes) - 1:
                self.canvas.setStrokeColor(HexColor(MUTED))
                self.canvas.setLineWidth(1.2)
                self.canvas.line(
                    left + width, y + height / 2, left + width + gap - 3, y + height / 2
                )
                self.canvas.line(
                    left + width + gap - 8,
                    y + height / 2 + 4,
                    left + width + gap - 3,
                    y + height / 2,
                )
                self.canvas.line(
                    left + width + gap - 8,
                    y + height / 2 - 4,
                    left + width + gap - 3,
                    y + height / 2,
                )

        sections = [
            (
                "Causality",
                "UTC one-minute bars are exact and contiguous. Context-only POC and volume "
                "references "
                "are frozen before the first target interval.",
            ),
            (
                "Exact-row policy",
                "Accepted bars retain every observed row up to the 1,024-row reject bound. "
                "Batching "
                "reduces item count for wide candles; it never compacts an accepted row.",
            ),
            (
                "Output boundary",
                "Low and high contain total and side POCs quantilewise. Marginals do not define "
                "a joint "
                "price path, product-distribution delta quantile, or order instruction.",
            ),
        ]
        for index, (title, body) in enumerate(sections):
            left = 42 + index * 255
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_BOLD, 11)
            self.canvas.drawString(left, 255, title)
            self._draw_wrapped(body, left, 235, 218, size=8.6, leading=12, color=MUTED)

    def _row_pool_and_fusion(self) -> None:
        self._new_page(
            "Architecture I - row pooling and fusion",
            "Learned summaries operate on the observed row set; semantic query names remain hypotheses.",
            kicker="MODEL DERIVATION",
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Per-row encoder",
            "An identical two-layer MLP projects each 11-feature row independently to model width, "
            "then RMS-normalizes it. Rows do not mix until cross-attention, and padded positions are "
            "excluded by the row mask.",
            42,
            PAGE_H - 122,
            355,
            BLUE,
        )
        self._equation_box(
            "ROW EMBEDDING",
            [
                "e[t,j] = RMSNorm(W2 * GELU(W1 * x[t,j] + b1) + b2)",
                "same weights for every observed price row",
            ],
            42,
            336,
            355,
            82,
        )
        self._narrative_block(
            "DESIGN HYPOTHESIS",
            "Four auction queries",
            "Four learned queries are intended to summarize total, buy-side, sell-side, and imbalance "
            "structure. Attention pictures cannot prove those roles. Query swaps, feature permutation, "
            "interventions, and seed stability are required interpretation tests.",
            42,
            228,
            355,
            PURPLE,
        )
        self._equation_box(
            "MASKED QUERY POOLING",
            [
                "alpha[k,j] = softmax_j((Wq*q[k]) dot (Wk*e[j]) / sqrt(d_head) + mask[j])",
                "a[k] = sum_j alpha[k,j] * Wv*e[j]",
                "score-count term is O(K*n_rows), with K = 4",
            ],
            430,
            PAGE_H - 120,
            366,
            110,
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Gated row/aggregate fusion",
            "The four query outputs are flattened into one row-set vector. A separate projection "
            "embeds the 23 deterministic aggregates. A sigmoid feature gate mixes them, while a "
            "residual projection prevents the representation from being only a convex mixture.",
            430,
            338,
            366,
            BLUE,
        )
        self._equation_box(
            "FUSION",
            [
                "g = sigmoid(Wg * concat(a_bar, b_bar) + bg)",
                "u = g * a_bar + (1-g) * b_bar + Wr * concat(a_bar, b_bar)",
            ],
            430,
            220,
            366,
            84,
        )
        self._draw_wrapped(
            "Required ablations: aggregates-only, rows-only, concatenation, addition, mean/max "
            "pooling, one query versus four queries, and matched-capacity alternatives.",
            430,
            112,
            366,
            size=8.2,
            leading=11,
            color=MUTED,
        )

    def _temporal_decoder(self) -> None:
        model = self.bundle.primary.data["model_config"]
        self._new_page(
            "Architecture II - causal history and direct horizons",
            "Dense attention is proportionate at 64 history positions; no generated forecast is recycled.",
            kicker="MODEL DERIVATION",
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Causal temporal stack",
            f"Three pre-RMSNorm blocks process 64 fused bar tokens at width {model['d_model']}. Each "
            "block uses masked multi-head attention, SwiGLU, residual connections, and LayerScale. "
            "The triangular mask preserves an online interpretation for every intermediate token.",
            42,
            PAGE_H - 122,
            365,
            BLUE,
        )
        self._equation_box(
            "TEMPORAL BLOCK",
            [
                "RMSNorm(x) = gain * x / sqrt(mean(x^2) + epsilon)",
                "Z = H + gamma_A * MHA(RMSNorm(H), causal_mask)",
                "H_next = Z + gamma_F * SwiGLU(RMSNorm(Z))",
                "mask[i,j] = 0 when j <= i; -infinity otherwise",
            ],
            42,
            332,
            365,
            122,
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Parallel horizon decoder",
            "Eight learned horizon queries first self-attend to one another, then cross-attend to the "
            "history memory. Horizon attention has no causal triangle because every query contains "
            "only learned identity - never a target realization or an earlier prediction.",
            42,
            188,
            365,
            TEAL,
        )

        self._equation_box(
            "DIRECT, NOT AUTOREGRESSIVE",
            [
                "z0[h] = learned_query[h] + learned_horizon_embedding[h]",
                "(Y_hat[1], ..., Y_hat[8]) = F_theta(X[t-63:t])",
                "NOT Y_hat[h] = f(X, Y_hat[1:h-1])",
            ],
            432,
            PAGE_H - 120,
            364,
            108,
        )
        self._narrative_block(
            "RATIONALE",
            "Why exact dense attention",
            "With L=64, the temporal sequence has only 4,096 position pairs per layer and head. "
            "The row-query term is linear in observed rows, and the decoder has only eight queries. "
            "Sparse attention, recurrent long-memory state, KV-cache compression, and expert routing "
            "solve different scale problems and add unjustified complexity here.",
            432,
            342,
            364,
            PURPLE,
        )
        self._equation_box(
            "ATTENTION SCORE COMPLEXITY",
            [
                "O(L*K*n_bar) + O(N_temporal*L^2)",
                "+ O(N_decoder*H*L) + O(N_decoder*H^2)",
                "L=64, K=4, H=8",
            ],
            432,
            220,
            364,
            92,
        )

    def _targets_and_constraints(self) -> None:
        self._new_page(
            "Direct targets, quantiles, and physical constraints",
            "Five price marginals share one cutoff anchor; volume and flow use separate monotone coordinates.",
            kicker="OUTPUT CONTRACT",
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Context-only normalization",
            "Let p_anchor be the final context POC, S_p=64 row ticks, S_v=4, and logV_ref the "
            "median log-volume over at most the final 16 context bars. No future observation enters "
            "these references.",
            42,
            PAGE_H - 122,
            370,
            BLUE,
        )
        self._equation_box(
            "SEVEN DIRECT TARGETS",
            [
                "price target k: y[k,h] = clip((p[k,t+h]-p_anchor)/S_p, -2, 2)",
                "volume: y[V,h] = clip((log(1+V[t+h])-logV_ref)/S_v, -2, 2)",
                "flow: y[Delta,h] = clip((V+ - V-)/(V+ + V-), -1, 1)",
            ],
            42,
            334,
            370,
            104,
        )
        self._narrative_block(
            "WHY DIRECT",
            "No composition of uncertain price forecasts",
            "Low, high, total POC, buy POC, and sell POC are all forecast directly from the same "
            "cutoff POC. The model does not predict a future POC and then add a separately uncertain "
            "gap to create another target.",
            42,
            207,
            370,
            PURPLE,
        )

        self._narrative_block(
            "IMPLEMENTED",
            "Ordered q10 / q50 / q90",
            "For each horizon and target, the head predicts a center plus positive softplus spreads. "
            "This prevents within-target quantile crossing before physical price constraints are applied.",
            438,
            PAGE_H - 122,
            358,
            TEAL,
        )
        self._equation_box(
            "NON-CROSSING PARAMETERIZATION",
            [
                "Q50 = center",
                "Q10 = center - softplus(a) - 1e-4",
                "Q90 = center + softplus(b) + 1e-4",
            ],
            438,
            348,
            358,
            92,
        )
        self._equation_box(
            "QUANTILEWISE PRICE CONSTRAINTS",
            [
                "low = min(raw_low, total_POC)",
                "high = max(raw_high, total_POC)",
                "buy_POC = clamp(raw_buy_POC, low, high)",
                "sell_POC = clamp(raw_sell_POC, low, high)",
            ],
            438,
            236,
            358,
            114,
        )
        self._draw_wrapped(
            "The operations preserve quantile order and guarantee low <= every POC <= high at each "
            "quantile. Delta-fraction quantiles pass through tanh. Structural order does not imply "
            "calibration, and constraint intervention rates must be measured.",
            438,
            102,
            358,
            size=8.2,
            leading=11,
            color=MUTED,
        )

    def _loss_and_joint_boundary(self) -> None:
        self._new_page(
            "Pinball loss and the marginal-versus-joint boundary",
            "Proper marginal scores answer a narrower question than coherent candle-path probability.",
            kicker="PROBABILITY",
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Pinball loss",
            "For residual e = y - q_hat at quantile tau, underprediction above the target quantile "
            "is penalized differently from overprediction. At q90, one unit of underprediction costs "
            "nine times one unit of overprediction.",
            42,
            PAGE_H - 122,
            365,
            BLUE,
        )
        self._equation_box(
            "WEIGHTED MASKED OBJECTIVE",
            [
                "rho_tau(e) = max(tau*e, (tau-1)*e)",
                "Loss = sum(mask * horizon_weight * target_weight * rho) / sum(weights)",
                "horizon_weight[h] = 1/sqrt(h)",
                "target weights = (1.0, .7, .7, .8, .8, .5, .7)",
            ],
            42,
            326,
            365,
            120,
        )
        self._narrative_block(
            "EVALUATION RULE",
            "Training weights cannot hide weak cells",
            "The paper publishes unweighted q10/q50/q90 pinball, q50-forecast MAE and bias, coverage, and "
            "width for every horizon-target cell and every declared seed.",
            42,
            184,
            365,
            TEAL,
        )

        self._narrative_block(
            "CRITICAL LIMIT",
            "Marginal quantiles are not a joint candle",
            "FDT-N estimates seven conditional marginals. Even when each 80% interval is calibrated, "
            "their Cartesian product is not an 80% joint region. Quantilewise price ordering is a "
            "physical constraint, not a learned dependence model.",
            432,
            PAGE_H - 122,
            364,
            RED,
        )
        self._equation_box(
            "MONOTONE PHYSICAL INVERSES",
            [
                "price[k,h,tau] = p_anchor + S_p * row_size * Q[k,h,tau]",
                "volume[h,tau] = exp(logV_ref + S_v*Q[V,h,tau]) - 1",
            ],
            432,
            328,
            364,
            88,
        )
        self._narrative_block(
            "NOT IMPLEMENTED AS A QUANTILE",
            "Physical delta requires joint information",
            "q50(volume) times q50(delta fraction) is only a plug-in point diagnostic. Multiplying "
            "same-quantile endpoints does not create a calibrated delta interval. A direct delta head, "
            "joint model, or separately validated conservative map is required.",
            432,
            215,
            364,
            AMBER,
        )

    def _worked_example(self) -> None:
        self._new_page(
            "Worked footprint and forecast example",
            "All numbers on this page are illustrative arithmetic, not model results.",
            kicker="BEGINNER WORKED EXAMPLE",
        )
        rows = [
            ["100.50", "8", "0", "8", "+8", "1", "0"],
            ["100.25", "32", "6", "38", "+26", "0", "0"],
            ["100.00", "20", "30", "50", "-10", "0", "0"],
            ["99.75", "5", "22", "27", "-17", "0", "0"],
            ["99.50", "0", "7", "7", "-7", "0", "1"],
        ]
        self._table(
            42,
            PAGE_H - 125,
            430,
            ["Price", "Buy", "Sell", "Total", "Delta", "ZeroBid", "ZeroAsk"],
            rows,
            [0.18, 0.12, 0.12, 0.14, 0.14, 0.15, 0.15],
            row_height=34,
            font_size=7.2,
        )
        self._narrative_block(
            "DERIVATION",
            "Bar summaries",
            "V+ = 65, V- = 65, V = 130, and Delta = 0. Total POC is 100.00, buy POC is "
            "100.25, and sell POC is 100.00. The volume-weighted total centroid is 100.025. Starting "
            "at POC volume 50, adding the 38-volume row above and 27-volume row below reaches 115, "
            "which exceeds the 70% value-area target of 91.",
            42,
            235,
            430,
            TEAL,
        )

        self._narrative_block(
            "ROW ENCODING",
            "The 100.25 row",
            "With row size 0.25, the row sits one tick above POC. Its buy, sell, total, and delta "
            "shares are 32/130=0.2462, 6/130=0.0462, 38/130=0.2923, and 26/130=0.2000.",
            500,
            PAGE_H - 122,
            296,
            BLUE,
        )
        self._narrative_block(
            "ILLUSTRATIVE FORECAST",
            "Direct median decoding",
            "From cutoff POC 100.00, offsets of POC +1, low -1, high +4, buy POC +1, and sell POC "
            "0 decode to 100.25, 99.75, 101.00, 100.25, and 100.00. If the context volume reference "
            "is 100 and the relative log-volume median is 0.2, volume is 101*exp(0.2)-1 = 122.36.",
            500,
            334,
            296,
            PURPLE,
        )
        self._narrative_block(
            "PINBALL ARITHMETIC",
            "One realized POC",
            "For illustrative q10/q50/q90 POC offsets (-1, 1, 3) and realization 2 ticks, pinball "
            "terms are 0.3, 0.5, and 0.1, averaging 0.3 tick. A delta-fraction median of 0.10 times "
            "122.36 gives plug-in delta 12.24; it is not a delta quantile.",
            500,
            188,
            296,
            AMBER,
        )

    def _training_protocol(self) -> None:
        run = self.bundle.primary
        config = run.data["train_config"]
        self._new_page(
            "Training, selection, and frozen protocol",
            "All declared seeds share one budget; only validation loss selects the primary run.",
            kicker="EXPERIMENT DESIGN",
        )
        rows = [
            ["Optimizer", "AdamW", "implemented"],
            ["Peak learning rate", f"{config['learning_rate']:.2e}", "implemented"],
            ["Weight decay", f"{config['weight_decay']:.2e}", "implemented"],
            ["Warm-up", f"{100 * config['warmup_fraction']:.1f}% of steps", "implemented"],
            ["Gradient clip", f"{config['gradient_clip']:.2f}", "implemented"],
            ["Epoch budget", str(config["epochs"]), "same for every seed"],
            ["Batch item cap", str(config["batch_size"]), "same for every seed"],
            ["Patience", str(config["patience"]), "validation only"],
            ["Row-token reference", str(config["row_token_reference"]), "exact rows retained"],
        ]
        self._table(
            42,
            PAGE_H - 123,
            356,
            ["Setting", "Value", "Evidence status"],
            rows,
            [0.39, 0.23, 0.38],
            row_height=31,
            font_size=6.8,
        )
        self._narrative_block(
            "IMPLEMENTED",
            "Purged chronological split",
            "Windows remain chronological: 70% train, 15% validation, 15% test, with 72-bar "
            "purges around boundaries. Standardization and direct-ridge fitting use training windows "
            "only. Random window shuffling never defines a partition.",
            428,
            PAGE_H - 122,
            368,
            BLUE,
        )
        self._narrative_block(
            "SELECTION RULE",
            "No held-out test selection",
            f"The primary is seed {run.seed}, selected by minimum validation loss "
            f"({run.validation_loss:.6f}); ties break by seed number. Every seed must share the same "
            "model, data, split, and non-seed training configuration. Every seed's held-out metrics "
            "remain visible regardless of rank.",
            428,
            335,
            368,
            TEAL,
        )
        self._narrative_block(
            "REQUIRED FOR FINAL QUALIFICATION",
            "Freeze before opening the outer test",
            "Freeze sources, hashes, venue, instrument, row size, classification, value area, quality, "
            "gap handling, wall-clock boundaries, model, optimization, loss, calibration, and all "
            "baselines. Preserve at least three declared seeds. Open a later outer holdout once, then "
            "publish every target-horizon cell, missing-label count, clipping rate, and failure.",
            428,
            211,
            368,
            AMBER,
        )
        self._draw_wrapped(
            "This report validates the declared multi-seed bundle and selected deployment artifact, "
            "but one 14-day corpus and one chronological split remain development evidence, not broad qualification.",
            428,
            91,
            368,
            size=8.2,
            leading=11,
            color=MUTED,
        )

    def _baselines_ablations_and_chmt(self) -> None:
        self._new_page(
            "Baselines and ablations",
            "Implemented point comparators are reported honestly; richer distributional controls remain required.",
            kicker="COMPARATORS",
        )
        implemented = [
            ["Persistence", "causal point", "same test windows", "implemented"],
            ["Direct ridge", "causal point", "train-only fit", "implemented"],
            ["Rolling / seasonal", "point", "time-aware", "required"],
            ["Quantile linear", "distribution", "matched labels", "required"],
            ["Boosted trees", "distribution", "aggregate inputs", "required"],
            ["Aggregate MLP", "distribution", "matched budget", "required"],
            ["GRU / TCN", "distribution", "causal sequence", "required"],
            ["DeepLOB-like", "distribution", "structural control", "required"],
            ["Time-series transformer", "distribution", "causal adapter", "required"],
        ]
        self._table(
            42,
            PAGE_H - 123,
            430,
            ["Family", "Output", "Fairness condition", "Status"],
            implemented,
            [0.25, 0.19, 0.34, 0.22],
            row_height=30,
            font_size=6.4,
        )
        self._narrative_block(
            "REQUIRED / PROPOSED",
            "Ablations that test the architecture claims",
            "Aggregate-only versus full rows; row-only versus gated fusion; one/four queries versus "
            "mean/max pooling; feature-group removals; causal transformer versus TCN/GRU; learned "
            "positions versus relative age; parallel decoder versus independent heads; equal versus "
            "1/sqrt(h) weights; lookbacks 16/32/64/128; and matched parameter-capacity controls.",
            500,
            PAGE_H - 122,
            296,
            PURPLE,
        )
        self._narrative_block(
            "BOUNDARY",
            "Point baselines do not test calibration",
            "Persistence and ridge compare model medians with point estimates. They cannot establish "
            "whether q10/q90 are sharp or calibrated. A claim about distribution quality needs "
            "task-matched quantile baselines and proper scores.",
            500,
            287,
            296,
            AMBER,
        )

        self._new_page(
            "Prior negative evidence - CHMT-Q",
            "A cautionary development study, not a matched FDT-N baseline and not reusable data.",
            kicker="TRANSFER BOUNDARY",
        )
        self._narrative_block(
            "OBSERVED PRIOR RESULT",
            "What CHMT-Q actually studied",
            "CHMT-Q used aggregate one-minute bars, half-second L1 tokens, fixed-depth L2 snapshots, "
            "and quality features on an OKX BTC-USDT period. It did not consume FDT-N's ragged "
            "footprint rows, used five horizons and five targets, and had a different coordinate, "
            "source contract, dates, and restricted dataset.",
            42,
            PAGE_H - 122,
            365,
            TEAL,
        )
        cards = [
            ("worse H1-H3", "POC q50 MAE", "tied H4-H5", RED),
            ("52.67%", "direction proxy", "MCC 0.0475", AMBER),
            ("0", "qualified trades", "all POC intervals crossed 0", BLUE),
            ("sealed", "outer holdout", "2026-06-25 to 06-30", PURPLE),
        ]
        self._metric_cards(cards, y=238)
        self._narrative_block(
            "DECISION",
            "Do not deploy; allocate no capital",
            "CHMT-Q's preregistered uncertainty policy made zero trades. The transferable lesson is "
            "narrow: a modest proper-score improvement does not imply useful point accuracy, candle "
            "direction, trades, or economic value. Its scores must never be shown beside FDT-N as if "
            "both models saw the same inputs, labels, venue, or period.",
            431,
            PAGE_H - 122,
            365,
            RED,
        )
        self._draw_wrapped(
            "The restricted CHMT-Q data and sealed outer holdout are neither opened nor copied for "
            "this study. The local CHMT-Q report is cited as prior negative evidence only.",
            431,
            203,
            365,
            size=8.3,
            leading=12,
            color=MUTED,
        )

    def _evaluation_design(self) -> None:
        comparison = self.bundle.primary.data["comparisons"]["model_vs_persistence"]
        self._new_page(
            "Evaluation definitions and dependence treatment",
            "Proper scores, calibration, width, and point errors answer different questions.",
            kicker="METHODS",
        )
        self._narrative_block(
            "PRIMARY DISTRIBUTIONAL METRICS",
            "Every horizon-target cell",
            "Report q10, q50, and q90 pinball; empirical quantile calibration; central-80% coverage; "
            "mean interval width; and, where implemented, interval score. Coverage without width can "
            "reward diffuse intervals, while width without coverage can reward unsafe narrowness.",
            42,
            PAGE_H - 122,
            365,
            BLUE,
        )
        self._equation_box(
            "CENTRAL INTERVAL SCORE (alpha = 0.20)",
            [
                "IS = (upper-lower)",
                "+ (2/alpha)*(lower-y) when y < lower",
                "+ (2/alpha)*(y-upper) when y > upper",
            ],
            42,
            318,
            365,
            102,
        )
        self._narrative_block(
            "SECONDARY METRICS",
            "Point and domain diagnostics",
            "Q50-forecast MAE and signed bias; total-POC direction as a secondary diagnostic; side-label "
            "availability; invariant violations; missing outputs; and conditional slices by horizon, "
            "volume, delta, volatility, session, quality, clipping, venue, and regime.",
            42,
            190,
            365,
            TEAL,
        )

        self._narrative_block(
            "DEPENDENCE-AWARE DESCRIPTION",
            "Paired four-hour block sensitivity intervals",
            f"Model-minus-baseline absolute errors stay paired within {comparison['blocks']} "
            f"contiguous 240-example blocks and are resampled {comparison['resamples']:,} times. "
            "This preserves within-block structure better than IID resampling.",
            432,
            PAGE_H - 122,
            364,
            PURPLE,
        )
        self._narrative_block(
            "RIGOR LIMIT",
            "Adjacent boundary windows still overlap",
            "Forecast windows on either side of a four-hour block boundary share context and target "
            "bars. The blocks are therefore not independent. The displayed ranges are descriptive "
            "sensitivity intervals - not independent-block confidence intervals, causal effects, "
            "p-values, or formal generalization bounds.",
            432,
            296,
            364,
            RED,
        )
        self._narrative_block(
            "LATER CONFIRMATORY STUDY",
            "Predeclare the dependence unit",
            "Choose the resampling or loss-differential unit before opening the outer test, justify it "
            "from sampling diagnostics, state any horizon correction and multiplicity policy, and "
            "publish effect sizes and intervals rather than a winner label alone.",
            432,
            163,
            364,
            AMBER,
        )

    def _corpus(self) -> None:
        corpus = self.bundle.corpus
        manifest = corpus.manifest
        self._new_page(
            "Frozen corpus, split, and provenance",
            "Manifest identities establish what was evaluated; they do not establish "
            "predictive value.",
            kicker="DATA",
        )
        p50, p95, p99, maximum = corpus.row_percentiles
        cards = [
            (f"{manifest['bars']:,}", "one-minute bars", "14 contiguous days", BLUE),
            (
                f"{manifest['encoded_price_rows']:,}",
                "exact price rows",
                f"p50 {p50:.0f} | p99 {p99:.2f}",
                TEAL,
            ),
            (f"{manifest['valid_windows']:,}", "valid 64+8 windows", "before purged split", PURPLE),
            (f"{maximum:.0f}", "maximum rows/bar", f"p95 {p95:.0f} | bound 1,024", AMBER),
        ]
        self._metric_cards(cards, y=PAGE_H - 185)
        splits = self.bundle.primary.data["splits"]
        total = sum(
            int(splits[name]) for name in ("train_examples", "validation_examples", "test_examples")
        )
        x, y, width, height = 42, 294, PAGE_W - 84, 28
        cursor = x
        for key, label, color in (
            ("train_examples", "TRAIN", BLUE),
            ("validation_examples", "VALIDATION", TEAL),
            ("test_examples", "TEST", PURPLE),
        ):
            value = int(splits[key])
            segment = width * value / total
            self.canvas.setFillColor(HexColor(color))
            self.canvas.rect(cursor, y, segment, height, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, 7.5)
            self.canvas.drawCentredString(cursor + segment / 2, y + 10, f"{label} {value:,}")
            cursor += segment
        self.canvas.setFillColor(HexColor(MUTED))
        self.canvas.setFont(FONT_REGULAR, 8)
        self.canvas.drawString(42, 278, "72-bar purge on each boundary | no calibration partition")

        details = [
            ("Source", "Binance Data Vision spot daily individual trades; live @trade granularity"),
            (
                "Raw provenance",
                f"14 external archives; {corpus.archive_bytes:,} original bytes; "
                "hashes in manifest",
            ),
            (
                "Local raw state",
                "ZIP copies truncated after encoding; raw bytes recoverable, not locally retained",
            ),
            (
                "Row/quantity",
                "2.5 USDT rows; SizeScale=1000; ties-to-even; zero scaled trades dropped",
            ),
            ("Manifest SHA-256", corpus.manifest_artifact.sha256),
            ("Dataset SHA-256", corpus.dataset_artifact.sha256),
        ]
        self._key_value_rows(details, 42, 236, PAGE_W - 84, row_height=25)

    def _clipping_audit(self) -> None:
        inputs = [item for item in self.bundle.corpus.clipping_audit if item.scope != "target"]
        targets = [item for item in self.bundle.corpus.clipping_audit if item.scope == "target"]
        self._new_page(
            "Input clipping and row-bound saturation",
            "Raw bar coordinates are tested before clipping; the already encoded row coordinate has a narrower audit.",
            kicker="TAIL-LOSS AUDIT",
        )
        input_rows = []
        for item in inputs:
            input_rows.append(
                [
                    "row" if item.scope == "row_input" else "bar",
                    item.feature.replace("_", " "),
                    _format_bound(item.lower_bound),
                    _format_bound_rate(item.lower_hits, item.observations),
                    _format_bound(item.upper_bound),
                    _format_bound_rate(item.upper_hits, item.observations),
                    f"{item.observations:,}",
                    "bound saturation" if item.method == "encoded_boundary_saturation" else "raw pre-clip",
                ]
            )
        self._table(
            42,
            PAGE_H - 124,
            PAGE_W - 84,
            ["Input", "Feature", "Low", "Low hit rate", "High", "High hit rate", "N", "Method"],
            input_rows,
            [0.07, 0.25, 0.07, 0.14, 0.07, 0.14, 0.13, 0.13],
            row_height=22,
            font_size=6.2,
        )
        self._draw_wrapped(
            "A raw pre-clip hit uses a strict inequality, so a value exactly equal to the bound is not "
            "counted as lost tail information. The frozen NPZ does not retain the row price before its "
            "initial clip; its row rate therefore counts encoded values at either boundary and may include "
            "exact-bound observations. Exact row pre-clip loss is explicitly unimplemented.",
            42,
            77,
            PAGE_W - 84,
            size=7.9,
            leading=10.5,
            color=MUTED,
        )

        self._new_page(
            "Target clipping audit",
            "Every valid horizon-target label is evaluated in its raw coordinate before the training clip.",
            kicker="TAIL-LOSS AUDIT",
        )
        target_rows = [
            [
                TARGET_LABELS[item.feature],
                _format_bound(item.lower_bound),
                _format_bound_rate(item.lower_hits, item.observations),
                _format_bound(item.upper_bound),
                _format_bound_rate(item.upper_hits, item.observations),
                f"{item.observations:,}",
            ]
            for item in targets
        ]
        self._table(
            42,
            PAGE_H - 125,
            530,
            ["Target", "Low", "Low clipped", "High", "High clipped", "Valid labels"],
            target_rows,
            [0.25, 0.10, 0.19, 0.10, 0.19, 0.17],
            row_height=35,
            font_size=7.1,
        )
        maximum = max(
            (
                (
                    max(item.lower_hits or 0, item.upper_hits or 0) / item.observations,
                    item,
                )
                for item in targets
            ),
            key=lambda pair: pair[0],
        )
        self._narrative_block(
            "MEASURED",
            "Largest single target-bound fraction",
            f"{TARGET_LABELS[maximum[1].feature]} reaches {100 * maximum[0]:.4f}% at its more active "
            "bound. The table gives every lower and upper fraction separately; no pooled average hides "
            "a target or direction.",
            603,
            PAGE_H - 122,
            193,
            TEAL,
        )
        self._narrative_block(
            "DENOMINATOR",
            "All continuous valid windows",
            "Each price, volume, and delta target is counted over every valid horizon label. Buy-POC "
            "and sell-POC denominators exclude labels with no relevant side volume, matching the target mask.",
            603,
            318,
            193,
            BLUE,
        )
        self._narrative_block(
            "INTERPRETATION",
            "Descriptive corpus audit",
            "Overlapping windows repeat bars, so these fractions describe the model's training/evaluation "
            "coordinate exposure. They are not independent statistical observations or a robustness proof.",
            603,
            195,
            193,
            AMBER,
        )

    def _training(self) -> None:
        primary = self.bundle.primary
        self._new_page(
            "Optimization history",
            "Validation traces are selection evidence; held-out test metrics remain outside "
            "checkpoint choice.",
            kicker="TRAINING",
        )
        self.canvas.drawImage(
            ImageReader(str(self.charts["training"])),
            42,
            102,
            width=535,
            height=360,
            preserveAspectRatio=True,
            mask="auto",
        )
        config = primary.data["train_config"]
        model = primary.data["model_config"]
        rows = [
            ("Primary seed", str(primary.seed)),
            ("Selection", "validation loss only"),
            ("Device class", str(primary.data["device"])),
            ("Elapsed", f"{primary.data['elapsed_seconds'] / 3600.0:.3f} hours"),
            ("Best epoch", str(primary.data["best_epoch"])),
            ("Epochs completed", str(primary.data["epochs_completed"])),
            ("Peak learning rate", f"{config['learning_rate']:.2e}"),
            ("Weight decay", f"{config['weight_decay']:.2e}"),
            ("Batch item cap", str(config["batch_size"])),
            ("Row-token reference", str(config["row_token_reference"])),
            ("Bootstrap resamples", f"{config['bootstrap_resamples']:,}"),
            ("d_model / heads", f"{model['d_model']} / {model['attention_heads']}"),
            ("Parameters", f"{primary.data['parameter_count']:,}"),
        ]
        self._key_value_rows(rows, 602, 450, 194, row_height=28, value_width=92)

    def _stability_pages(self) -> None:
        runs = sorted(self.bundle.runs, key=lambda item: item.seed)
        chunks = [runs[index : index + 8] for index in range(0, len(runs), 8)]
        for page_index, chunk in enumerate(chunks, start=1):
            suffix = f" ({page_index}/{len(chunks)})" if len(chunks) > 1 else ""
            self._new_page(
                f"Seed stability{suffix}",
                "Primary selection uses validation loss only; all held-out summaries remain "
                "visible.",
                kicker="MULTI-SEED",
            )
            if page_index == 1:
                self.canvas.drawImage(
                    ImageReader(str(self.charts["stability"])),
                    42,
                    204,
                    width=480,
                    height=296,
                    preserveAspectRatio=True,
                    mask="auto",
                )
                table_x, table_y, table_w = 540, 463, 256
            else:
                table_x, table_y, table_w = 42, 463, PAGE_W - 84
            rows = []
            for run in chunk:
                test = run.data["test"]["pooled"]
                rows.append(
                    [
                        str(run.seed),
                        "PRIMARY" if run.seed == self.bundle.primary.seed else "reported",
                        f"{run.validation_loss:.5f}",
                        f"{run.data['test']['loss']:.5f}",
                        f"{test['median_mae_normalized']:.4f}",
                        f"{100 * test['central_80_coverage']:.1f}%",
                    ]
                )
            self._table(
                table_x,
                table_y,
                table_w,
                ["Seed", "Status", "Val loss", "Test loss", "Test MAE", "80% cov"],
                rows,
                [0.11, 0.20, 0.18, 0.18, 0.17, 0.16],
                row_height=25,
                font_size=7.0,
            )
            if page_index == 1:
                summaries = _stability_summary(runs)
                self._draw_wrapped(
                    "Stability summary: " + " | ".join(summaries),
                    42,
                    173,
                    PAGE_W - 84,
                    size=8.6,
                    leading=12,
                    color=MUTED,
                )

    def _probabilistic_scores(self) -> None:
        run = self.bundle.primary
        self._new_page(
            "Primary probabilistic scores",
            "Every horizon-target-quantile cell is reported in normalized target space; "
            "lower pinball is better.",
            kicker="HELD-OUT TEST",
        )
        self.canvas.drawImage(
            ImageReader(str(self.charts["pinball"])),
            39,
            93,
            width=758,
            height=405,
            preserveAspectRatio=True,
            mask="auto",
        )
        pooled = run.data["test"]["pooled"]["pinball_normalized"]
        self._draw_wrapped(
            f"Primary seed {run.seed} pooled pinball: q10 {pooled['q10']:.5f}, "
            f"q50 {pooled['q50']:.5f}, q90 {pooled['q90']:.5f}. "
            "These are marginal distribution scores, not a coherent joint path or economic result.",
            42,
            69,
            PAGE_W - 84,
            size=8.2,
            color=MUTED,
            leading=11,
        )

    def _calibration(self) -> None:
        run = self.bundle.primary
        pooled = run.data["test"]["pooled"]
        self._new_page(
            "Calibration and sharpness",
            "Coverage without width can reward diffuse intervals; width without coverage can "
            "reward unsafe narrowness.",
            kicker="HELD-OUT TEST",
        )
        self.canvas.drawImage(
            ImageReader(str(self.charts["coverage_width"])),
            42,
            132,
            width=655,
            height=350,
            preserveAspectRatio=True,
            mask="auto",
        )
        cards = [
            (f"{100 * pooled['central_80_coverage']:.1f}%", "pooled coverage", "nominal 80%", BLUE),
            (
                f"{pooled['mean_interval_width_normalized']:.4f}",
                "pooled width",
                "normalized",
                PURPLE,
            ),
            (f"{pooled['median_mae_normalized']:.4f}", "pooled q50 MAE", "normalized", TEAL),
            (
                f"{100 * pooled['poc_directional_hit_rate']:.1f}%",
                "POC sign hit",
                "secondary metric",
                AMBER,
            ),
        ]
        self._metric_cards(cards, y=90)

    def _baseline_effects(self) -> None:
        run = self.bundle.primary
        self._new_page(
            "Point-baseline comparisons",
            "Persistence and direct ridge are point baselines; they cannot establish "
            "quantile calibration.",
            kicker="PAIRED EFFECTS",
        )
        self.canvas.drawImage(
            ImageReader(str(self.charts["comparison"])),
            42,
            112,
            width=755,
            height=386,
            preserveAspectRatio=True,
            mask="auto",
        )
        comparison = run.data["comparisons"]["model_vs_persistence"]
        self._draw_wrapped(
            f"Effects are model q50-forecast MAE minus baseline MAE. Negative cells favor FDT-N. "
            f"Intervals use {comparison['resamples']:,} paired whole-block resamples over "
            f"{comparison['blocks']} contiguous four-hour blocks. Adjacent block-boundary "
            "windows still overlap, so these are descriptive sensitivity intervals. Cell "
            "multiplicity is descriptive.",
            42,
            81,
            PAGE_W - 84,
            size=8.3,
            color=MUTED,
            leading=11,
        )

    def _poc_intervals(self) -> None:
        self._new_page(
            "Total-POC effect intervals",
            "Paired four-hour sensitivity intervals preserve within-block dependence; adjacent "
            "boundary windows overlap.",
            kicker="PRIMARY RUN DETAIL",
        )
        self.canvas.drawImage(
            ImageReader(str(self.charts["poc_intervals"])),
            42,
            115,
            width=675,
            height=390,
            preserveAspectRatio=True,
            mask="auto",
        )
        self._draw_wrapped(
            "The dashed zero line marks equal q50-forecast MAE. An interval crossing zero is compatible "
            "with either ordering under this descriptive resampling design. The intervals are not "
            "independent-block or formal generalization bounds. No p-value or trading inference "
            "is made.",
            729,
            420,
            72,
            size=7.4,
            leading=10,
            color=MUTED,
        )

    def _seed_scorecard(self, run: RunRecord) -> None:
        selected = run.seed == self.bundle.primary.seed
        self._new_page(
            f"Held-out scorecard - seed {run.seed}",
            "Example-weighted summaries across all eight horizons; every declared seed receives "
            "a full page.",
            kicker="PRIMARY RUN" if selected else "DECLARED SEED",
        )
        pooled = run.data["test"]["pooled"]
        cards = [
            (
                f"{run.validation_loss:.5f}",
                "validation loss",
                "selection input",
                TEAL if selected else BLUE,
            ),
            (f"{run.data['test']['loss']:.5f}", "held-out test loss", "post-selection", PURPLE),
            (
                f"{100 * pooled['poc_directional_hit_rate']:.1f}%",
                "POC sign hit",
                "secondary",
                AMBER,
            ),
            (
                "PRIMARY" if selected else "REPORTED",
                "run status",
                "validation-only rule",
                TEAL if selected else MUTED,
            ),
        ]
        self._metric_cards(cards, y=PAGE_H - 184)
        summaries = _target_summaries(run)
        rows = []
        for target_name in TARGET_NAMES:
            values = summaries[target_name]
            rows.append(
                [
                    TARGET_LABELS[target_name],
                    f"{values['q10']:.4f}",
                    f"{values['q50']:.4f}",
                    f"{values['q90']:.4f}",
                    f"{values['mae']:.4f}",
                    f"{values['bias']:+.4f}",
                    f"{100 * values['coverage']:.1f}%",
                    f"{values['width']:.4f}",
                ]
            )
        self._table(
            42,
            305,
            PAGE_W - 84,
            ["Target", "q10 PB", "q50 PB", "q90 PB", "Q50 MAE", "Bias", "80% cov", "Width"],
            rows,
            [0.18, 0.11, 0.11, 0.11, 0.14, 0.11, 0.11, 0.13],
            row_height=28,
            font_size=7.5,
        )
        self._draw_wrapped(
            "PB = normalized pinball. Target summaries weight horizon cells by valid example "
            "count. "
            "The seven marginals are not a joint candle distribution.",
            42,
            72,
            PAGE_W - 84,
            size=8.2,
            color=MUTED,
            leading=11,
        )

    def _seed_cell_pages(self, run: RunRecord) -> None:
        rows = []
        for horizon in range(1, 9):
            for target_name in TARGET_NAMES:
                metric = run.data["test"]["by_horizon_target"][str(horizon)][target_name]
                pinball = metric["pinball_normalized"]
                rows.append(
                    [
                        str(horizon),
                        TARGET_LABELS[target_name],
                        f"{metric['examples']:,}",
                        f"{pinball['q10']:.5f}",
                        f"{pinball['q50']:.5f}",
                        f"{pinball['q90']:.5f}",
                        f"{metric['median_mae_normalized']:.5f}",
                        f"{metric['median_bias_normalized']:+.5f}",
                        f"{100 * metric['central_80_coverage']:.2f}%",
                        f"{metric['mean_interval_width_normalized']:.5f}",
                    ]
                )
        chunks = [rows[index : index + 19] for index in range(0, len(rows), 19)]
        for index, chunk in enumerate(chunks, start=1):
            self._new_page(
                f"Seed {run.seed} held-out cells ({index}/{len(chunks)})",
                "Every horizon, target, and reported quantile score; no cell is hidden by pooling.",
                kicker=(
                    "PRIMARY RUN DETAIL"
                    if run.seed == self.bundle.primary.seed
                    else "DECLARED SEED DETAIL"
                ),
            )
            self._table(
                42,
                PAGE_H - 124,
                PAGE_W - 84,
                [
                    "H",
                    "Target",
                    "N",
                    "q10 PB",
                    "q50 PB",
                    "q90 PB",
                    "Q50 MAE",
                    "Bias",
                    "80% cov",
                    "Width",
                ],
                chunk,
                [0.04, 0.15, 0.07, 0.09, 0.09, 0.09, 0.11, 0.11, 0.10, 0.15],
                row_height=18,
                font_size=5.8,
            )
            self._draw_wrapped(
                "PB = normalized pinball. Coverage is empirical q10-to-q90 inclusion; width is in "
                "the encoded target coordinate. These are marginal forecast cells, not joint paths.",
                42,
                76,
                PAGE_W - 84,
                size=7.8,
                leading=10.5,
                color=MUTED,
            )

    def _deployment_evidence(self) -> None:
        metadata = self.bundle.deployment.data
        parity = metadata["parity"]
        latency = parity["cpu_latency_ms"]
        sample = parity["sample"]
        coordinate = metadata["coordinate"]
        primary = self.bundle.primary
        self._new_page(
            "Selected model artifact and runtime evidence",
            "Metadata lineage binds the validation-selected results, checkpoint, ONNX graph, parity, and coordinate.",
            kicker="DEPLOYMENT ARTIFACT",
        )
        cards = [
            (str(primary.seed), "selected seed", "validation-only", TEAL),
            (str(metadata["checkpoint_epoch"]), "checkpoint epoch", "selected validation epoch", BLUE),
            (
                f"{parity['maximum_absolute_error']:.2e}",
                "PyTorch vs ORT max error",
                f"tolerance {parity['tolerance']:.1e}",
                PURPLE,
            ),
            (f"{latency['p95']:.3f} ms", "CPU latency p95", "50 timed runs", AMBER),
        ]
        self._metric_cards(cards, y=PAGE_H - 218)
        details = [
            ("Runtime status", metadata["runtime_status"]),
            ("Creates order signals", str(metadata["creates_order_signals"]).lower()),
            ("Coordinate", f"{coordinate['source']} / {coordinate['instrument_key']}"),
            (
                "Interval / row size",
                f"{coordinate['interval_seconds']} seconds / {coordinate['row_size']} USDT",
            ),
            ("History / horizon", f"{metadata['minimum_history_bars']} / {metadata['maximum_horizon_bars']} bars"),
            ("Representative width", f"{sample['window_max_rows']} rows (p95 {sample['p95_window_max_rows']:.2f})"),
            ("Latency p50 / p95 / p99", f"{latency['p50']:.3f} / {latency['p95']:.3f} / {latency['p99']:.3f} ms"),
            ("ONNX Runtime provider", parity["provider"]),
            (
                "Python / PyTorch",
                f"{metadata['environment']['python']} / {metadata['environment']['torch']}",
            ),
            (
                "ONNX / ORT",
                f"{parity['onnx_version']} / {parity['onnxruntime_version']}",
            ),
        ]
        self._key_value_rows(details, 42, 274, 365, row_height=27, value_width=218)
        hashes = [
            ("Checkpoint SHA-256", metadata["checkpoint_sha256"]),
            ("ONNX SHA-256", metadata["onnx_sha256"]),
            ("Primary results SHA-256", metadata["lineage"]["training_results"]["sha256"]),
            ("Metadata SHA-256", self.bundle.deployment.metadata.sha256),
        ]
        self._key_value_rows(hashes, 431, 274, 365, row_height=42, value_width=242)
        self._draw_wrapped(
            "The runtime coordinate uses the product broker identity 'Binance'. Training provenance "
            "is separately frozen as Binance Data Vision spot daily individual trades. Passed fixture "
            "parity and representative CPU timing qualify this exact exported artifact for shadow-only "
            "research; they do not establish live end-to-end latency, drift resistance, or trading safety.",
            431,
            90,
            365,
            size=8.1,
            leading=11,
            color=MUTED,
        )

    def _runtime_safety(self) -> None:
        self._new_page(
            "Runtime contract, fail-closed behavior, and governance",
            "Forecast display must never become an order path; native footprint rendering survives abstention.",
            kicker="PRODUCT SAFETY",
        )
        self._narrative_block(
            "IMPLEMENTED SHARED SEAM",
            "Explicit coordinate and status",
            "Requests bind instrument, source, interval, row size, cutoff, and target interval. Results "
            "distinguish Available, Unavailable, InsufficientHistory, and Failed. Available values carry "
            "provider/model/version identity and ordered q10/q50/q90 for all seven target families.",
            42,
            PAGE_H - 122,
            365,
            BLUE,
        )
        failures = [
            "model, schema, feature order, coordinate, row size, or tick-size mismatch",
            "insufficient, discontinuous, stale, incomplete, or rejected-quality history",
            "more than 1,024 rows, non-finite tensors, or runtime timeout/cancellation/crash",
            "unordered quantiles or low/high/POC/volume/delta invariant failure",
            "artifact-hash mismatch or unqualified physical-delta reconstruction",
        ]
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 10)
        self.canvas.drawString(42, 330, "Provider abstains on at least")
        self._bullet_list(failures, 42, 310, 365, size=7.8, leading=11)
        self._narrative_block(
            "PASSED ARTIFACT GATE",
            "Export parity is evidence, not promotion",
            "The selected ONNX graph passed the recorded PyTorch-versus-ONNX Runtime tolerance and "
            "50-run representative CPU timing. Those completed facts replace a future-parity placeholder, "
            "but the artifact remains shadow-only and creates no order signals.",
            431,
            PAGE_H - 122,
            365,
            TEAL,
        )
        governance = [
            "retain source, data, split, result, checkpoint, ONNX, metadata, and report hashes",
            "monitor input saturation, row counts, feed quality, abstention, latency, and calibration drift",
            "prohibit online weight updates, runtime model merging, and silent coordinate fallback",
            "keep a reversible version switch and null-provider fallback",
            "label forecasts as predictions and expose cutoff, horizon, age, model, and unavailable reason",
            "preserve the invariant that no live order-execution path is introduced",
        ]
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 10)
        self.canvas.drawString(431, 330, "Monitoring and governance")
        self._bullet_list(governance, 431, 310, 365, size=7.8, leading=11)

    def _related_work(self) -> None:
        self._new_page(
            "Related work - transferable mechanisms",
            "A cited component motivates an experiment; it does not establish FDT-N performance.",
            kicker="LITERATURE BOUNDARY",
        )
        sections = [
            (
                "Time-series forecasting",
                "TFT motivates direct quantiles, gates, and known-versus-observed covariates [S07]. "
                "PatchTST treats regular history as patches [S08]; here one sealed bar is already a "
                "domain patch, while strict channel independence is rejected. iTransformer, Crossformer, "
                "TimeXer, and Moirai motivate structured comparators, not native ragged-row support [S09-S12].",
                BLUE,
            ),
            (
                "Simple and normalized controls",
                "Strong linear baselines can outperform complex forecasters [S05], so persistence, ridge, "
                "seasonal, and quantile-linear controls are mandatory. Reversible instance normalization "
                "is an ablation only; the implemented coordinate uses causal physical scales [S06].",
                TEAL,
            ),
            (
                "Limit-order-book models",
                "DeepLOB, TransLOB, and TLOB motivate structural processing before temporal processing "
                "[S13-S15]. Their usual displayed-book and future-midprice tasks differ from executed-flow "
                "footprint marginals. FI-2010 and deep order-flow imbalance inform evaluation and baselines, "
                "not a footprint benchmark or P&L claim [S27-S28].",
                PURPLE,
            ),
            (
                "Transformer components",
                "Scaled attention, RMSNorm, SwiGLU, and relative-position work provide mature component "
                "precedents [S01-S04]. Qwen component choices are large-language-model precedent only "
                "[S20-S21]; language reasoning, grouped-query cache optimization, and agentic behavior do "
                "not transfer to a fixed supervised tensor map.",
                AMBER,
            ),
        ]
        positions = [(42, PAGE_H - 122), (431, PAGE_H - 122), (42, 290), (431, 290)]
        for (title, body, color), (x, y) in zip(sections, positions, strict=True):
            self._narrative_block("TRANSFER", title, body, x, y, 365, color)

        self._new_page(
            "Related work - rejected scale mechanisms and calibration",
            "The short exact sequence makes giant-model machinery disproportionate to the identified problem.",
            kicker="LITERATURE BOUNDARY",
        )
        rejected = [
            ["MoE / expert routing", "giant capacity specialization", "no demonstrated bottleneck", "reject v1"],
            ["Latent / grouped-query attention", "autoregressive KV-cache width", "no generated token cache", "reject"],
            ["Native sparse attention", "very long contexts", "64 bars / 4,096 pairs", "reject"],
            ["Kimi linear recurrence", "long-memory efficiency", "reset and gap semantics dominate", "defer"],
            ["Runtime singular-value scaling", "task adaptation", "breaks frozen hash lineage", "reject"],
            ["Evolutionary model merging", "combine qualified experts", "no qualified expert pool", "defer offline"],
            ["LLM reasoning / agents", "language generation and tools", "fixed supervised map", "out of scope"],
        ]
        self._table(
            42,
            PAGE_H - 124,
            PAGE_W - 84,
            ["Mechanism", "Original problem", "FDT-N condition", "Decision"],
            rejected,
            [0.24, 0.27, 0.31, 0.18],
            row_height=35,
            font_size=6.7,
        )
        self._narrative_block(
            "PROBABILISTIC CALIBRATION",
            "Conformal correction is optional, not universal",
            "Conformalized quantile regression can correct intervals on a held-out calibration set "
            "under its assumptions [S26]. Markets are serially dependent and nonstationary. Any later "
            "chronological conformal study must report empirical coverage under that exact procedure, "
            "not a universal guarantee. This frozen split has no separate calibration partition.",
            42,
            190,
            754,
            TEAL,
        )

    def _claims_and_conclusion(self) -> None:
        primary = self.bundle.primary
        pooled = primary.data["test"]["pooled"]
        persistence = primary.data["baselines"]["persistence"]["test"]["pooled"]
        ridge = primary.data["baselines"]["direct_ridge"]["test"]["pooled"]
        model_mae = pooled["median_mae_normalized"]
        persistence_mae = persistence["median_mae_normalized"]
        ridge_mae = ridge["median_mae_normalized"]
        persistence_change = 100.0 * (model_mae / persistence_mae - 1.0)
        ridge_change = 100.0 * (model_mae / ridge_mae - 1.0)
        persistence_cells = [
            cell
            for horizon in primary.data["comparisons"]["model_vs_persistence"][
                "by_horizon_target"
            ].values()
            for cell in horizon.values()
        ]
        ridge_cells = [
            cell
            for horizon in primary.data["comparisons"]["model_vs_direct_ridge"][
                "by_horizon_target"
            ].values()
            for cell in horizon.values()
        ]
        persistence_point_wins = sum(
            cell["mae_difference_normalized"] < 0.0 for cell in persistence_cells
        )
        ridge_point_wins = sum(cell["mae_difference_normalized"] < 0.0 for cell in ridge_cells)
        persistence_interval_wins = sum(cell["ci_95_upper"] < 0.0 for cell in persistence_cells)
        ridge_interval_wins = sum(cell["ci_95_upper"] < 0.0 for cell in ridge_cells)
        self._new_page(
            "Claims held, supported conclusion, and next experiment",
            "A complete artifact can narrow uncertainty without authorizing deployment or capital.",
            kicker="CONCLUSION",
        )
        held = [
            "full rows are better than aggregates until the registered ablation is run",
            "learned queries are semantically interpretable without intervention evidence",
            "the seven marginals form a coherent joint candle or physical-delta distribution",
            "results generalize across symbols, venues, intervals, sessions, or regimes",
            "forecast accuracy implies a directional, tradable, profitable, or risk-adjusted edge",
            "artifact parity implies production latency, operational safety, or capital suitability",
            "the model is state of the art, production-ready, or robust to distribution shift",
        ]
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 11)
        self.canvas.drawString(42, PAGE_H - 126, "The paper does not claim")
        self._bullet_list(held, 42, PAGE_H - 150, 365, size=7.9, leading=11)
        self._narrative_block(
            "SUPPORTED MECHANICAL CONCLUSION",
            "What FDT-N is",
            "FDT-N is a compact hierarchical direct multi-horizon transformer. It maps every accepted "
            "footprint row plus deterministic bar aggregates into ordered marginal quantiles for seven "
            "future auction targets, without autoregressive forecast feedback. The selected exported "
            "artifact is hash-bound, parity-checked, and shadow-only.",
            431,
            PAGE_H - 122,
            365,
            TEAL,
        )
        self._narrative_block(
            "EMPIRICAL SCOPE",
            "What this frozen study can say",
            f"Validation-selected seed {primary.seed} had pooled normalized q50-forecast MAE {model_mae:.5f}: "
            f"{abs(persistence_change):.2f}% below persistence ({persistence_mae:.5f}) but "
            f"{ridge_change:.2f}% above train-only ridge ({ridge_mae:.5f}). It beat persistence in "
            f"{persistence_point_wins}/56 cells by point estimate and {persistence_interval_wins}/56 by "
            f"descriptive interval; corresponding ridge counts were {ridge_point_wins}/56 and "
            f"{ridge_interval_wins}/56. Central 80% coverage was "
            f"{pooled['central_80_coverage']:.4f}. Thus FDT-N did not beat the stronger linear control.",
            431,
            299,
            365,
            BLUE,
        )
        self._narrative_block(
            "NEXT EVIDENCE",
            "What remains before promotion",
            "Run task-matched distributional baselines and registered ablations; preserve a later untouched "
            "outer period; add cross-symbol, cross-venue, and regime tests; validate live receive-time "
            "provenance and end-to-end latency; then conduct a separate execution study with frozen costs, "
            "risk, and abstention policy. Until then: no deployment and no capital allocation.",
            431,
            172,
            365,
            AMBER,
        )

    def _references_pages(self) -> None:
        chunks = [REFERENCES[index : index + 11] for index in range(0, len(REFERENCES), 11)]
        for page_index, chunk in enumerate(chunks, start=1):
            self._new_page(
                f"References ({page_index}/{len(chunks)})",
                "Primary locators are included; citations transfer mechanisms, not market-performance claims.",
                kicker="REFERENCES",
            )
            y = PAGE_H - 119
            for identifier, reference in chunk:
                self.canvas.setFillColor(HexColor(BLUE))
                self.canvas.setFont(FONT_BOLD, 7.1)
                self.canvas.drawString(42, y, f"[{identifier}]")
                y = self._draw_wrapped(
                    reference,
                    78,
                    y,
                    PAGE_W - 120,
                    size=7.7,
                    leading=10.2,
                    color=INK,
                )
                y -= 9

    def _reproducibility_pages(self) -> None:
        artifacts = [
            self.bundle.corpus.manifest_artifact,
            self.bundle.corpus.dataset_artifact,
            self.bundle.paper,
            self.bundle.ledger,
            self.bundle.deployment.metadata,
            self.bundle.deployment.onnx,
        ]
        for run in sorted(self.bundle.runs, key=lambda item: item.seed):
            artifacts.extend(
                [
                    _artifact("Results JSON", run.path, seed=run.seed),
                    run.checkpoint,
                    run.splits,
                    run.ridge,
                ]
            )
        chunks = [artifacts[index : index + 13] for index in range(0, len(artifacts), 13)]
        for index, chunk in enumerate(chunks, start=1):
            suffix = f" ({index}/{len(chunks)})" if len(chunks) > 1 else ""
            self._new_page(
                f"Reproducibility ledger{suffix}",
                "Full SHA-256 identities bind the source narrative, data, splits, baselines, "
                "and every seed.",
                kicker="LINEAGE",
            )
            rows = [
                [
                    item.label,
                    "shared" if item.seed is None else str(item.seed),
                    f"{item.bytes:,}",
                    item.sha256,
                ]
                for item in chunk
            ]
            self._table(
                42,
                PAGE_H - 124,
                PAGE_W - 84,
                ["Artifact", "Seed", "Bytes", "SHA-256"],
                rows,
                [0.20, 0.08, 0.13, 0.59],
                row_height=26,
                font_size=6.8,
            )
            self._draw_wrapped(
                "Primary selection rule: minimum validation loss, tie-broken by seed number; no test "
                "score enters selection. Results record seeds, device class, configuration, elapsed "
                "time, and one post-selection test evaluation per seed. Exact GPU model and original "
                "command lines were not frozen, and the repository had unrelated concurrent dirty work; "
                "there is no separate FDT-N outer holdout. These are disclosed reproducibility limits.",
                42,
                92,
                PAGE_W - 84,
                size=7.6,
                color=MUTED,
                leading=9.5,
            )

    def _limitations(self) -> None:
        self._new_page(
            "Decision boundary and required next evidence",
            "A complete PDF is a reporting artifact, not model promotion or authorization.",
            dark=True,
            kicker="LIMITATIONS",
        )
        columns = [
            (
                "What this report establishes",
                [
                    "Exact identities for the frozen data and every run",
                    "Validation-only primary selection across declared seeds",
                    "Per-cell marginal scores and point-baseline effects",
                    "Bound clipping audit and selected ONNX parity evidence",
                ],
                TEAL,
            ),
            (
                "What it does not establish",
                [
                    "A joint candle or physical-delta distribution",
                    "Cross-venue, cross-symbol, or cross-regime generality",
                    "Economic value after fills, latency, fees, and impact",
                    "Production latency, operational safety, or capital suitability",
                ],
                AMBER,
            ),
            (
                "Required before promotion",
                [
                    "Independent later outer holdout and calibration period",
                    "Task-matched distributional baselines and ablations",
                    "Live provider latency, memory, drift, and failure tests",
                    "Separate execution study with frozen risk and cost policy",
                ],
                BLUE,
            ),
        ]
        for index, (title, items, color) in enumerate(columns):
            left = 42 + index * 258
            self.canvas.setFillColor(HexColor(color))
            self.canvas.roundRect(left, 132, 226, 330, 10, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(NAVY))
            self.canvas.setFont(FONT_BOLD, 12)
            self.canvas.drawString(left + 16, 430, title)
            y = 392
            for item in items:
                self.canvas.setFont(FONT_BOLD, 10)
                self.canvas.drawString(left + 16, y, "+")
                y = (
                    self._draw_wrapped(
                        item,
                        left + 32,
                        y,
                        177,
                        size=8.4,
                        leading=12,
                        color=NAVY,
                    )
                    - 18
                )

    def _narrative_block(
        self,
        label: str,
        title: str,
        body: str,
        x: float,
        y: float,
        width: float,
        color: str,
    ) -> float:
        label_width = pdfmetrics.stringWidth(label, FONT_BOLD, 6.5) + 14
        self.canvas.setFillColor(HexColor(color))
        self.canvas.roundRect(x, y - 12, label_width, 15, 4, stroke=0, fill=1)
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setFont(FONT_BOLD, 6.5)
        self.canvas.drawString(x + 7, y - 7, label)
        self.canvas.setFillColor(HexColor(INK))
        self.canvas.setFont(FONT_BOLD, 10.5)
        self.canvas.drawString(x, y - 31, title)
        cursor = self._draw_wrapped(
            body,
            x,
            y - 49,
            width,
            size=8.2,
            leading=11.3,
            color=MUTED,
        )
        return cursor - 8

    def _equation_box(
        self,
        title: str,
        lines: Sequence[str],
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> float:
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setStrokeColor(HexColor(GRID))
        self.canvas.roundRect(x, y - height, width, height, 7, stroke=1, fill=1)
        self.canvas.setFillColor(HexColor(BLUE))
        self.canvas.setFont(FONT_BOLD, 7.1)
        self.canvas.drawString(x + 12, y - 20, title)
        cursor = y - 38
        for line in lines:
            cursor = self._draw_wrapped(
                line,
                x + 12,
                cursor,
                width - 24,
                font=FONT_REGULAR,
                size=7.4,
                leading=10,
                color=INK,
            )
            cursor -= 3
        return y - height

    def _bullet_list(
        self,
        items: Sequence[str],
        x: float,
        y: float,
        width: float,
        *,
        size: float,
        leading: float,
    ) -> float:
        cursor = y
        for item in items:
            self.canvas.setFillColor(HexColor(BLUE))
            self.canvas.setFont(FONT_BOLD, size)
            self.canvas.drawString(x, cursor, "-")
            cursor = self._draw_wrapped(
                item,
                x + 13,
                cursor,
                width - 13,
                size=size,
                leading=leading,
                color=INK,
            )
            cursor -= 7
        return cursor

    def _metric_cards(
        self,
        cards: Sequence[tuple[str, str, str, str]],
        *,
        y: float,
        dark: bool = False,
    ) -> None:
        gap = 13
        width = (PAGE_W - 84 - gap * (len(cards) - 1)) / len(cards)
        for index, (value, label, note, color) in enumerate(cards):
            left = 42 + index * (width + gap)
            background = "#17233F" if dark else WHITE
            self.canvas.setFillColor(HexColor(background))
            self.canvas.roundRect(left, y, width, 102, 9, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(color))
            self.canvas.rect(left, y + 96, width, 6, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(WHITE if dark else INK))
            self.canvas.setFont(FONT_BOLD, 18 if len(value) < 12 else 13)
            self.canvas.drawString(left + 13, y + 62, value)
            self.canvas.setFont(FONT_BOLD, 8)
            self.canvas.drawString(left + 13, y + 39, label.upper())
            self.canvas.setFillColor(HexColor("#94A3B8" if dark else MUTED))
            self.canvas.setFont(FONT_REGULAR, 7.2)
            self.canvas.drawString(left + 13, y + 21, note)

    def _rounded_box(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        subtitle: str,
        color: str,
    ) -> None:
        self.canvas.setFillColor(HexColor(WHITE))
        self.canvas.setStrokeColor(HexColor(color))
        self.canvas.setLineWidth(1.3)
        self.canvas.roundRect(x, y, width, height, 8, stroke=1, fill=1)
        self.canvas.setFillColor(HexColor(color))
        self.canvas.setFont(FONT_BOLD, 8.3)
        self.canvas.drawString(x + 8, y + height - 19, title)
        self._draw_wrapped(
            subtitle, x + 8, y + height - 34, width - 16, size=6.7, leading=9, color=MUTED
        )

    def _key_value_rows(
        self,
        rows: Sequence[tuple[str, str]],
        x: float,
        y: float,
        width: float,
        *,
        row_height: float,
        value_width: float | None = None,
    ) -> None:
        label_width = width - (value_width or width * 0.72)
        for index, (label, value) in enumerate(rows):
            top = y - index * row_height
            self.canvas.setFillColor(HexColor(WHITE if index % 2 == 0 else "#EEF2F7"))
            self.canvas.rect(x, top - row_height + 2, width, row_height - 2, stroke=0, fill=1)
            self.canvas.setFillColor(HexColor(MUTED))
            self.canvas.setFont(FONT_BOLD, 7.4)
            self.canvas.drawString(x + 9, top - 16, label.upper())
            self.canvas.setFillColor(HexColor(INK))
            self.canvas.setFont(FONT_REGULAR, 7.2)
            text = value
            max_width = width - label_width - 18
            while pdfmetrics.stringWidth(text, FONT_REGULAR, 7.2) > max_width and len(text) > 12:
                text = text[:-1]
            if text != value:
                text = text[:-3] + "..."
            self.canvas.drawString(x + label_width, top - 16, text)

    def _table(
        self,
        x: float,
        y: float,
        width: float,
        headers: Sequence[str],
        rows: Sequence[Sequence[str]],
        fractions: Sequence[float],
        *,
        row_height: float,
        font_size: float,
    ) -> None:
        widths = [width * fraction for fraction in fractions]
        self.canvas.setFillColor(HexColor(NAVY))
        self.canvas.rect(x, y - row_height, width, row_height, stroke=0, fill=1)
        cursor = x
        for header, cell_width in zip(headers, widths, strict=True):
            self.canvas.setFillColor(HexColor(WHITE))
            self.canvas.setFont(FONT_BOLD, font_size)
            self.canvas.drawString(cursor + 6, y - row_height + 9, header)
            cursor += cell_width
        for row_index, row in enumerate(rows):
            top = y - row_height * (row_index + 1)
            self.canvas.setFillColor(HexColor(WHITE if row_index % 2 == 0 else "#EEF2F7"))
            self.canvas.rect(x, top - row_height, width, row_height, stroke=0, fill=1)
            cursor = x
            for value, cell_width in zip(row, widths, strict=True):
                self.canvas.setFillColor(HexColor(INK))
                self.canvas.setFont(FONT_REGULAR, font_size)
                text = str(value)
                while (
                    pdfmetrics.stringWidth(text, FONT_REGULAR, font_size) > cell_width - 10
                    and len(text) > 8
                ):
                    text = text[:-1]
                if text != str(value):
                    text = text[:-3] + "..."
                self.canvas.drawString(cursor + 6, top - row_height + 9, text)
                cursor += cell_width

    def _draw_wrapped(
        self,
        text: str,
        x: float,
        y: float,
        width: float,
        *,
        font: str = FONT_REGULAR,
        size: float = 8.5,
        leading: float = 12,
        color: str = INK,
    ) -> float:
        words: list[str] = []
        for word in text.split():
            if pdfmetrics.stringWidth(word, font, size) <= width:
                words.append(word)
                continue
            chunk = ""
            for character in word:
                candidate = chunk + character
                if chunk and pdfmetrics.stringWidth(candidate, font, size) > width:
                    words.append(chunk)
                    chunk = character
                else:
                    chunk = candidate
            if chunk:
                words.append(chunk)
        lines: list[str] = []
        line = ""
        for word in words:
            candidate = word if not line else f"{line} {word}"
            if pdfmetrics.stringWidth(candidate, font, size) <= width:
                line = candidate
            else:
                if line:
                    lines.append(line)
                line = word
        if line:
            lines.append(line)
        self.canvas.setFillColor(HexColor(color))
        self.canvas.setFont(font, size)
        cursor = y
        for line in lines:
            self.canvas.drawString(x, cursor, line)
            cursor -= leading
        return cursor


def _format_bound(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:g}"


def _format_bound_rate(hits: int | None, observations: int) -> str:
    if hits is None:
        return "-"
    percentage = 100.0 * hits / observations
    if hits and percentage < 0.0001:
        return "<0.0001%"
    return f"{percentage:.4f}%"


def _target_summaries(run: RunRecord) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for target_name in TARGET_NAMES:
        cells = [
            run.data["test"]["by_horizon_target"][str(horizon)][target_name]
            for horizon in range(1, 9)
        ]
        weights = np.asarray([cell["examples"] for cell in cells], dtype=np.float64)
        output[target_name] = {
            "q10": _weighted([cell["pinball_normalized"]["q10"] for cell in cells], weights),
            "q50": _weighted([cell["pinball_normalized"]["q50"] for cell in cells], weights),
            "q90": _weighted([cell["pinball_normalized"]["q90"] for cell in cells], weights),
            "mae": _weighted([cell["median_mae_normalized"] for cell in cells], weights),
            "bias": _weighted([cell["median_bias_normalized"] for cell in cells], weights),
            "coverage": _weighted([cell["central_80_coverage"] for cell in cells], weights),
            "width": _weighted([cell["mean_interval_width_normalized"] for cell in cells], weights),
        }
    return output


def _weighted(values: Sequence[float], weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def _stability_summary(runs: Sequence[RunRecord]) -> list[str]:
    metrics = {
        "test loss": [float(run.data["test"]["loss"]) for run in runs],
        "pooled MAE": [float(run.data["test"]["pooled"]["median_mae_normalized"]) for run in runs],
        "coverage": [float(run.data["test"]["pooled"]["central_80_coverage"]) for run in runs],
    }
    output = []
    for name, values in metrics.items():
        array = np.asarray(values, dtype=np.float64)
        output.append(f"{name} mean {array.mean():.4f}, sd {array.std(ddof=0):.4f}")
    return output


def _register_fonts() -> None:
    global FONT_REGULAR, FONT_BOLD
    regular = Path(r"C:\Windows\Fonts\segoeui.ttf")
    semibold = Path(r"C:\Windows\Fonts\seguisb.ttf")
    if regular.is_file() and semibold.is_file():
        if "SegoeUI" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("SegoeUI", regular))
        if "SegoeUI-Semibold" not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont("SegoeUI-Semibold", semibold))
        FONT_REGULAR = "SegoeUI"
        FONT_BOLD = "SegoeUI-Semibold"


def _artifact(label: str, path: Path, *, seed: int | None = None) -> Artifact:
    resolved = _require_file(path, label)
    return Artifact(label, resolved, _sha256(resolved), resolved.stat().st_size, seed)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exception:
        raise ValueError(f"cannot read JSON {path}: {exception}") from exception
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _require_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _resolve_recorded_path(value: Any, run_dir: Path, repo_root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("recorded artifact path is missing")
    raw = Path(value)
    candidates = (
        [raw] if raw.is_absolute() else [repo_root / raw, run_dir / raw.name, run_dir / raw]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ValueError(f"recorded artifact does not exist: {value}")


def _find_repo_root(paper_path: Path) -> Path:
    for parent in paper_path.resolve().parents:
        if (parent / "tools" / "volume-footprint-transformer").is_dir():
            return parent
    raise ValueError("cannot locate repository root from research paper path")


def _require_descendant(path: Path, directory: Path, label: str) -> None:
    allowed = directory.resolve()
    try:
        path.relative_to(allowed)
    except ValueError as exception:
        raise ValueError(f"{label} must be under {allowed}: {path}") from exception


def _reject_pending_values(value: Any, path: str) -> None:
    if value is None:
        raise PendingResultsError(f"{path} contains a null metric")
    if isinstance(value, float) and not math.isfinite(value):
        raise PendingResultsError(f"{path} contains a non-finite metric")
    if isinstance(value, str) and ("[PENDING]" in value or "PLACEHOLDER" in value.upper()):
        raise PendingResultsError(f"{path} contains a pending marker")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_pending_values(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_pending_values(item, f"{path}[{index}]")


def _finite(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and (number <= minimum if strict_minimum else number < minimum):
        raise ValueError(f"{label} is below its allowed minimum")
    if maximum is not None and number > maximum:
        raise ValueError(f"{label} is above its allowed maximum")
    return number


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
        raise ValueError(f"{label} must be an integer")
    integer = int(value)
    if integer < 0 if allow_zero else integer <= 0:
        raise ValueError(f"{label} must be {'non-negative' if allow_zero else 'positive'}")
    return integer


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_date(timestamp_ms: int) -> str:
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).strftime("%Y-%m-%d")
