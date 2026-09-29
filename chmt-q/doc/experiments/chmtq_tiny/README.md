# CHMT-Q Tiny

This directory contains the reproducible runner for a compact CHMT-Q engineering pilot. The model
keeps the published Bars/L1/L2/quality inputs, hierarchical side/rank-before-time book path,
hard masks, quality-bounded modality gates, five horizon queries, and `[B,5,17]` raw output. It uses
width 64 and 282,838 trainable parameters rather than the reference model's width 128 and 2,089,554
parameters.

The default run deliberately reuses only the already accessed June development corpus:

- training: 1--18 June 2026;
- validation and checkpoint selection: 19--20 June 2026;
- development-forward evaluation: 21--24 June 2026;
- 25--30 June is excluded.

This is a retrospective implementation check, not a new confirmatory result. Raw Arrow tensors,
checkpoints, and result JSON remain under `C:\DaxAlgoData`; they must not be committed here.

Run from this directory with `python run.py`. A quick pipeline check can use
`--epochs 1 --limit-train-batches 2 --limit-eval-batches 2` and a separate output directory.

## New BTC pilot

[`protocol-july-august-2026.json`](protocol-july-august-2026.json) freezes the next study before
target compilation: 1--21 July for training, 22--28 July for validation, and 29 July--4 August for
development-forward evaluation. Five seeds and the test-open rule are fixed there.

[`download_okx.py`](download_okx.py) acquires one UTC day from the official OKX archive and keeps
the payload outside Git. Because the trade ZIP labels use Asia/Shanghai days, it acquires trade
labels D and D+1, while the L2 archive uses label D. The downloader validates remote metadata,
archive integrity, member schema, the initial L2 snapshot, timestamps, and SHA-256 before recording
the files in a local manifest. Use `--plan` to inspect a day without network or filesystem writes.
