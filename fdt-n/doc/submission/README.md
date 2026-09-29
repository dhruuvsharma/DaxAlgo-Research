# FDT-N submission package

This directory contains the submission source for FDT-N: Footprint Distribution Transformer--Narrow.

- `main.tex` is the manuscript source.
- `references.bib` is the bibliography.
- `build_figures.py` deterministically regenerates the vector figures under `assets/`.
- `assets/` contains the generated figure PDFs embedded by the manuscript.

The paper explains the complete Inputs $\rightarrow$ Transformer Processing $\rightarrow$ Outputs
path, includes an artifact-bound forward-pass example, and reports the frozen 14-day BTCUSDT
development benchmark. Dhruv Sharma is the named author.

The final reviewed PDF is written to `../reports/fdt-n-volume-footprint-transformer-final-submission.pdf`.
