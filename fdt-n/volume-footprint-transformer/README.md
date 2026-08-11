# FDT-N volume-footprint transformer

This Pro-private tool builds, trains, evaluates, and exports FDT-N: a compact hierarchical
transformer for direct next-N footprint-bar forecasts. It consumes the complete sealed footprint
shape (every price row and all chart aggregates) rather than the chart's current 16-feature linear
summary.

The tool is research-only. A checkpoint is not evidence of predictive value. Runtime activation
requires a chronological held-out report, exact preprocessing parity, and an instrument/timeframe
match. It never creates an order or execution signal.

Generated data and runs belong under `tmp/volume-footprint-transformer/`. Reviewed models and the
final research PDF belong under `output/models/` and `output/pdf/` respectively.

## Frozen PDF research paper

The `report` command accepts at least three complete schema-v2 `results.json` files, the exact-row
dataset and manifest, the selected export's `metadata.json`, and the paper/evidence sources. Every
run must use the same dataset, architecture, split indices, and non-seed training configuration. The
primary seed is selected solely by minimum validation loss (seed number breaks an exact tie);
held-out test metrics never participate in selection. The PDF is a self-contained paper with the
beginner derivations, architecture and evaluation rationale, clipping audit, prior CHMT-Q boundary,
complete references, primary figures, every held-out horizon-target-quantile cell for every seed,
stability summary, selected ONNX parity/latency evidence, and full artifact hashes.

Install the `report` optional dependency group before rendering. By default, the command writes
`output/pdf/fdt-n-volume-footprint-transformer-research-paper.pdf` and stages figures under
`tmp/pdfs/fdt-n-volume-footprint-transformer/`.

The command fails with exit code 2 when any results JSON is missing, legacy, incomplete, contains a
pending/non-finite metric, omits a required baseline/comparison, mixes training budgets, or does not
match the frozen dataset. It also refuses a metadata/checkpoint/ONNX lineage mismatch, failed parity,
non-50-run CPU latency evidence, a non-shadow runtime status, or fewer than three final seeds. It
never creates a placeholder PDF.

## Anonymous publication manuscript

The `publication` command applies the same evidence and lineage gates to a shorter, conventional A4
academic manuscript. It renders the source at
`docs/research/volume-footprint-transformer/publication-manuscript.md`, generates the six numbered
figures from frozen results, and writes
`output/pdf/fdt-n-volume-footprint-transformer-publication-manuscript.pdf`. The manuscript leads with
the cross-seed negative ridge comparison and is intentionally anonymous; author, affiliation, ORCID,
funding, competing-interest, and venue metadata must be supplied before non-blind publication.
