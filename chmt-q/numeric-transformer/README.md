# DaxAlgo numeric transformer research implementation

This Pro-private package now contains two deliberately separate paths:

- the original point-forecast prototype, retained as a regression baseline; and
- the selected **CHMT-Q v1** architecture, verified engineered-data path, development-fold runner,
  fixed-shape export core, and source-readiness audit.

It contains no market data, trained weights, promoted model package, or terminal integration. The
CHMT-Q code accepts already compiled causal tensors; it does not reinterpret raw ticks or fabricate
missing quote/depth/footprint streams.

The design authorities are the
[`architecture`](../../docs/architecture/numeric-market-transformer-architecture.md) and
[`research specification`](../../docs/research/numeric-market-transformer/research-spec.md). The
literal implementation has 2,089,554 parameters: the earlier 2,089,518 ledger omitted 36 learned
per-head Q/K temperatures while also requiring them independently from the 36 elapsed-time decay
parameters. The code preserves both required mechanisms and asserts the corrected count.

## Current boundary

Implemented and covered by tests:

- fixed bar/L1/L2/quality tensor and supervision contracts;
- hierarchical L2, causal bar/L1/L2 encoders, hard availability gates, direct horizon decoder, and
  monotone quantile/presence outputs;
- masked pinball, median Huber, presence, and scheduled auxiliary objectives;
- a strict single-instrument, content-addressed Arrow IPC dataset contract with
  shard/schema/hash/chronology checks and a path-contained, byte-hash-verified local evidence
  inventory for readiness, rights, and provenance gates;
- expanding session walk-forward folds with exact purge/embargo records and separately sealed outer
  holdout access;
- exact median/MAD scaler artifacts fitted through deterministic bounded-memory disk spooling and
  cryptographically bound to the dataset, split, fold, and ordered training example IDs;
- AdamW grouping, accumulation, warm-up/cosine scheduling, finite checks, clipping, deterministic
  initialization, exact effective-batch mask denominators, and verified immutable checkpoints;
- immutable experiment/run identities, epoch-boundary resume, early stopping, checkpoint selection,
  normalized and physical-unit metrics, structured epoch logs, zero-residual and empirical baselines,
  attention-logit/gate/mask telemetry, and sealed development-only train/evaluate commands;
- split-conformal calibration and fail-closed physical reconstruction primitives, plus bounded
  overfit and random-target qualification controls;
- normalized-input raw-core ONNX export/parity helpers; and
- a read-only Parquet audit for the observed BTC/USD corpus; and
- resumable, hash-manifested OKX trade/400-level L2 archive acquisition and streaming audit.

Not implemented here are the production .NET causal tensor compiler and provenance-complete live
source capture, the full registered neural/linear/tree baseline suite, final multi-fold/three-seed
candidate study, dedicated calibration-slice orchestration, scaler-wrapped/inverse-target ONNX
packaging, signed promotion (including the evaluator that loads and scores the frozen candidate),
serving, or UI integration. Those stages require valid causal source data and later frozen candidate
evidence; they are not silently simulated by Python.

The nine files currently under `D:\LSE Data` are **not training-ready for CHMT-Q**. They have partial
calendar coverage, pervasive indistinguishable duplicates, a constant zero `side`, and no availability
clock, sequence/source epoch, bid/ask sizes, depth, or provably genuine footprint labels. The audit
therefore fails closed instead of inventing semantics. They remain useful for source auditing and
bounded pipeline experiments only.

The source host is now attributable: every file retains a Windows download marker for
`londonstrategicedge.com`, and the provider's default vault filename is `crypto_BTC_USD_tick.parquet`.
That attribution does not repair the contract. The shared LSE adapter documents the feed as L1-only,
without sizes or depth, and explicitly does not treat `price` plus `volume` as a verified trade tape.
The provider SDK calls vault history a raw tape but does not define the historical `side` code,
availability/receive clocks, source epoch, sequence identity, quantity units, or export completeness.
The provider UI also declares a 2,500,000-tick maximum per download; every local shard ends exactly at
that cap before its named endpoint.

## OKX historical source acquisition

The standard-library OKX acquirer keeps official compressed archives outside the repository, resumes
partial transfers, verifies OKX ETag/MD5 plus local SHA-256, and writes an immutable-source manifest.
For a requested UTC window it includes the prior L2 day for the 500 ms input lookback and the next
UTC+8 trade partition for complete end-of-window labels. L1 and one-minute bars are derived later from
400-level updates and genuine trades; redundant one-minute candle archives are not required.

```powershell
.venv\Scripts\python -m daxalgo_numeric_transformer.okx_history acquire `
  --root "C:\DaxAlgoData\okx\BTC-USDT\2026-06" `
  --start 2026-06-01 --end 2026-06-30 --workers 3
.venv\Scripts\python -m daxalgo_numeric_transformer.okx_history audit `
  --root "C:\DaxAlgoData\okx\BTC-USDT\2026-06"
```

The audit fully streams each ZIP/tar.gz to its footer, validates every trade row, and reconstructs a
bounded semantic prefix plus the final row of every L2 day. OKX archives still lack independent local
receive/availability timestamps and L2 sequence IDs, so acquisition does not by itself satisfy the
strict provenance-complete training gate.

## Setup

Use Python 3.11 or newer from this directory:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev,data]"
```

Add the optional exporter/runtime only for ONNX verification:

```powershell
.venv\Scripts\python -m pip install -e ".[dev,data,onnx]"
```

No hand-authored dependency lock is claimed. A reviewed training run must freeze its resolved
environment as part of the immutable run identity and pass that exact lock file to both experiment
creation and execution.

## Review commands

Inspect the reference model and corrected parameter ledger:

```powershell
.venv\Scripts\python -m daxalgo_numeric_transformer chmtq-inspect
```

Run a bounded synthetic optimization through the selected architecture:

```powershell
.venv\Scripts\python -m daxalgo_numeric_transformer chmtq-smoke --device cpu --seed 17 --steps 3
```

Run the stronger non-promotable trainer controls:

```powershell
.venv\Scripts\python -m daxalgo_numeric_transformer qualify
```

Audit the BTC/USD source without writing to it. Exit code `3` means the audit completed but blocked
training; `--output` is optional and refuses to overwrite an existing report.

```powershell
.venv\Scripts\python -m daxalgo_numeric_transformer audit-data --input "D:\LSE Data"
```

## Compiled-dataset workflow

The development training command accepts only an already-engineered Arrow dataset emitted from the
reviewed causal .NET compiler contract. It never accepts the raw Parquet corpus. The canonical
dataset manifest binds exactly one `instrument_id` and three reviewable local evidence
files. Opening rejects missing or changed bytes, symbolic links, path escapes, non-regular files, and
row instruments that differ from the manifest before any training/evaluation batch is available.
The readiness and provenance records also bind the declared source-manifest and builder digests, all
schema digests, and the exact ordered `example_id`-to-provenance-digest inventory. This proves that
those declarations cannot drift independently inside the dataset; it does not independently prove
external source or builder bytes that are not included in the local evidence inventory.
The fail-closed workflow is:

1. `lock-environment --output <lock.json>`;
2. `verify-dataset --input <dataset> --dataset-id <sha256> --instrument-id BTC-USD`;
3. `build-split ... --output <split.json>`;
4. `fit-scaler ... --split <split.json> --fold <n> --output <scaler.json>`;
5. `init-experiment ... --instrument-id BTC-USD --scaler <scaler.json>
   --dependency-lock <lock.json> --output <experiment.json>`;
6. review the dataset evidence and every generated immutable artifact, then run `qualify`;
7. `train --dataset <dataset> --config <experiment.json> --split <split.json>
   --scaler <scaler.json> --dependency-lock <lock.json> --output-root <runs>`; and
8. `evaluate` the selected checkpoint on the development validation partition.

Datasets whose strict readiness or provenance gates remain false may be used only through an
explicit, content-addressed local-research authorization. `authorize-local-research` requires a
named authorizer, reason, explicit training-rights confirmation, and acknowledgements of incomplete
readiness and provenance. It binds the unchanged dataset manifest, capability evidence, and compiler
report; it does not change any dataset gate. Pass the resulting
`--local-research-authorization <authorization.json>` to every subsequent dataset command. The
authorization permits local development training and validation only: promotion, export, serving,
and outer-holdout access remain false. The authorization digest is part of the experiment, run, and
checkpoint identities.

Use `train --progress` for bounded live optimizer-window, validation, checkpoint, and epoch events.
Progress output is intended for a visible terminal and may be captured to a durable log without
changing the immutable run artifacts.

For large folds, point `fit-scaler --spool-directory` at reviewed local scratch storage with enough
free space for exact float64 statistics. `--max-chunk-values` bounds working memory without changing
the exact scaler artifact identity. Exact empirical-baseline quantiles use a bounded SQLite order
index under the operating-system temporary directory, so that volume also needs sufficient space.

`train --resume <checkpoint>` resumes only the latest verified epoch-boundary checkpoint for the same
run. If first-epoch event publication succeeds but its first checkpoint does not, repeat the identical
`train` command without `--resume`; recovery verifies and quarantines the orphan/staging evidence,
then deterministically replays epoch one. Training and development-evaluation APIs cannot materialize
the outer holdout. Its dedicated
data-access gate byte-verifies a frozen-candidate file and its freeze-evidence file, then atomically
creates the one canonical split/candidate/evidence-bound consumption ledger inside the verified
dataset before returning any holdout iterator. This guards accidental repeat access; it is not the
deferred promotion evaluator and cannot defend against deliberate removal or copying of governed
dataset state. Each run records the exact source-tree hash, dependency-lock hash,
dataset/split/scaler identities, baseline scores, denominators, telemetry, and selected checkpoint.

Run all local checks:

```powershell
.venv\Scripts\python -m pytest
.venv\Scripts\python -m ruff check .
```

ONNX tests skip cleanly when the optional dependencies are absent. Every synthetic checkpoint or
ONNX file is non-promotable and must not be shown as a live forecast or treated as evaluation
evidence.

## Code map

- `daxalgo_numeric_transformer/chmt_q/config.py`, `contracts.py`, `output.py`, and `loss.py`: stable
  model and objective contract.
- `daxalgo_numeric_transformer/chmt_q/model/`: CHMT-Q implementation.
- `daxalgo_numeric_transformer/chmt_q/data/`: source audit, verified Arrow loader, and fold-bound
  scaler.
- `daxalgo_numeric_transformer/chmt_q/scaling.py`, `splits.py`, `baselines.py`, and `calibration.py`:
  preprocessing and evaluation artifacts.
- `daxalgo_numeric_transformer/chmt_q/training/`: optimizer, exact step loop, walk-forward manifests,
  checkpoints, and experiment runner.
- `daxalgo_numeric_transformer/chmt_q/export.py`: fixed batch-one raw-core ONNX boundary.
- `daxalgo_numeric_transformer/chmt_q/qualification.py`: bounded synthetic controls.
- `tests/test_chmt_q_*.py`: executable review surface and invariants.
