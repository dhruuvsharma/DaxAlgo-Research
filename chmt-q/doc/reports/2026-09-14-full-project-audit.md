# CHMT-Q Full Project Analysis — Synthesized Audit Report

**Date:** September 14, 2026
**Scope:** Complete audit of `chmt-q` — architecture, data pipeline, training loop, backtest
scripts, and documentation/test consistency — synthesized from five independent audit passes with
live verification against the frozen June 2026 study artifacts.
**Verification sources:** `ChmtQTransformer(ChmtQConfig.reference())` live instantiation,
`chmtq-inspect`, `python -m pytest -q`, and direct reads of
`C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1\`.

---

## 1. Executive summary

The repository is in strong overall shape. The model implementation matches its published contract
exactly (2,089,554 parameters, verified by live instantiation), the backtest scripts reproduce
every headline number in the documentation from the frozen artifacts, and the test suite is green
(231 passed, 2 platform skips). The audit found **no correctness defects** — only minor issues
(I-1..I-9, all low severity) and a set of documentation-consistency findings, chiefly the
superseded 2,089,518-parameter design ledger that several documents still carry without a
disambiguation note.

**Headline study facts (all traced to frozen artifacts):**

| Fact | Value |
|---|---|
| Market / period | OKX BTC-USDT, June 2026 |
| Partitions | 25,758 train / 2,862 selection / 5,724 forward examples |
| Seed / fold / selected checkpoint | 17 / fold-01 / epoch 10 @ step 2,010 |
| Selection primary pinball | 0.21076511027987574 vs baseline 0.22244570920702006 |
| Forward edge | +2.2558% (0.25139 vs 0.25719); bootstrap 95% CI [0.00047, 0.01261] |
| H5 Pearson correlation | 0.0774; std ratio 0.1513 (under-dispersed) |
| Median long/flat policy | 484 trades, +2.1607% frictionless, −68.07% @ 12 bps/side |
| Break-even cost | 0.2227 bps one-way (53.89× base cost) |
| Interval policies | zero trades |

The one-line scientific conclusion: the model delivers a **statistically real but economically
unusable** aggregate pinball edge. The +2.26% forward improvement is significant (CI excludes
zero), but it does not translate into usable location forecasts (H5 physical MAE improvement of
only 0.0119% over the empirical marginal), and the trading policy's break-even cost of 0.2227 bps
is far below realistic execution costs.

---

## 2. Architecture audit

**Verified by live instantiation** — `ChmtQTransformer(ChmtQConfig.reference()).parameter_count`:

- Parameter count: **2,089,554** (exactly matches docs; the 2,089,518 figure is an earlier design
  ledger, see §6).
- Contract shapes honored: bars `[B,64,16]`, L1 `[B,240,16]`, L2 `[B,240,2,10,12]`, quality
  `[B,16]`, 5 horizon queries, output `[B,5,17]`.
- Width 128, 4 attention heads, FF width 256, 184-token fused memory.
- Causality probes passed: no information leakage from future bars/quotes/depth into past-token
  representations.

**Minor issues (all low severity):**

| ID | Finding |
|---|---|
| I-1 | Per-head Q/K temperatures add 36 params not in the superseded ledger (documented in `config.py:28`, handled) |
| I-2 | Some shape checks use Python bools on tensors — triggers ONNX `TracerWarning`s during export tests (benign; tests pass) |
| I-3 | Hard availability gates treat missing data as masked, not zero — correct, but worth a comment where mask semantics are subtle |
| I-4..I-9 | Naming, docstring, and dead-code nits across `model/` — cosmetic only |

No architectural defects were found that would invalidate the study's claims.

## 3. Data pipeline audit

- Partitioning (25,758 / 2,862 / 5,724) is consistent across artifacts and docs; no overlap or
  leakage between selection and forward windows detected.
- Normalization is per-fold, fit on train only, applied frozen to selection/forward — correct
  practice.
- Quality/masking path verified: unavailable L1/L2 observations are masked at input, never
  imputed as zeros.
- No pipeline bugs found.

## 4. Training loop audit

- Seed 17, fold-01 run is fully reproducible from `run.json` + `config.json` + epoch records.
- Epoch/step bookkeeping is consistent: 18 epochs, selected epoch 10 @ step 2,010, matching the
  `selections` records and `evaluation-fold-01-seed-17.json`.
- Early stopping and selection rule implemented as documented (primary = mean normalized pinball
  across horizons/targets on the selection partition vs empirical marginal baseline).
- No training-loop defects found.

## 5. Backtest / evaluation audit

Every headline number in `doc/okx-june-2026-development-backtest-report.md` and the manuscript
was **traced to the frozen artifacts and reproduced exactly**:

- Selection primary 0.21076511027987574 vs 0.22244570920702006 ✓
- Forward 0.25139 vs 0.25719 (+2.2558%) ✓; bootstrap CI [0.00047, 0.01261] ✓
- H5 Pearson 0.0774, std ratio 0.1513 ✓
- Median long/flat: 484 trades, +2.1607% frictionless, −68.07% @ 12 bps/side, break-even
  0.2227 bps one-way (53.89× base) ✓
- Interval policies: zero trades ✓ (intervals too narrow to cross entry thresholds — consistent
  with the under-dispersion diagnosis)

**Artifact schema quirks to know when consuming these files directly:**

- `evaluation-fold-01-seed-17.json` is UTF-16 encoded.
- Attention `max_abs_by_head` grids are stored as bracketed comma strings.
- The run `config.json` key is `training` (not `training_config`).
- Policy returns live at `metrics.net_compounded_return` (no top-level `net_return`).
- All metric grids are `[horizon][target]`, 5×5, over targets
  `total_poc_offset, buy_poc_offset, sell_poc_offset, log_volume_residual, delta_fraction`.

## 6. Docs & tests consistency audit

**Test suite:** 233 collected; **231 passed, 2 skipped** (both skips are Windows symlink-privilege
gates in `test_chmt_q_research_authorization.py:249` and `test_chmt_q_walk_forward.py:315`, not
product defects). 137 warnings, all benign ONNX `TracerWarning`s from Python-bool shape checks.

**Stale 2,089,518 parameter count — locations and status:**

| Location | Status |
|---|---|
| `doc/AGENTS.md:17` | ✅ correctly disambiguates (2,089,554 current, 2,089,518 earlier) |
| `numeric-transformer/README.md:16` | ✅ correctly disambiguates |
| `numeric-transformer/.../chmt_q/config.py:28` | ✅ comment explains the 36-param delta |
| `doc/from-first-principles.md:17, 607, 1037` | ⚠️ presents 2,089,518 as "designed" ledger — intentional (design-vs-implementation distinction) but easy to misread |
| `doc/README.md:53` | ✅ flags the earlier tutorial as superseded |
| `doc/publication/chmtq-arxiv-manuscript.tex:1545` | ✅ footnote explains the delta |
| `doc/SESSION-CONTEXT.md:99` | ✅ explicitly warns not to use 2,089,518 |

**Other consistency findings (from the docs/tests audit, 10 total):** minor wording drift between
the backtest report and manuscript on friction assumptions; one outdated cross-reference in
`from-first-principles.md`; the remaining findings are formatting/typo level. None affect claims.

## 7. Consolidated recommendations

1. **No code changes required.** The implementation, pipeline, training, and evaluation are
   consistent and verified.
2. Consider adding a one-line note at the top of `doc/from-first-principles.md` table rows
   (lines 607, 1037) clarifying that the 2,089,518 ledger is the *design-era* figure, superseded
   by the implemented 2,089,554 — the distinction exists but is buried in prose.
3. The 2 Windows symlink skips are environmental; consider marking them `xfail`/`skipif` cleanly
   (they already self-skip) so CI logs stay quiet.
4. For future consumers of the frozen artifacts, the schema quirks in §5 are the main friction;
   the visualizer (`research-backtest/visualize_chmtq_study.py`) demonstrates the correct
   handling of each.

---

## Appendix A — Verification commands used

```
cd numeric-transformer
python -m pytest -q                     # 231 passed, 2 skipped
python -c "from daxalgo_numeric_transformer.chmt_q.model import ChmtQTransformer, ChmtQConfig; \
           print(ChmtQTransformer(ChmtQConfig.reference()).parameter_count)"  # 2089554
chmtq-inspect                           # contract/shape report OK
python -m ruff check .                  # clean
```

## Appendix B — Frozen artifact map

```
C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1\
├── runs\25be2f446a75ad37120bba956ea86e46c0ecefa004ab3343d6745cdbee102fb3\
│   ├── run.json, config.json
│   ├── epochs\epoch-0001..0018.json
│   ├── selections\, checkpoints\
├── evaluation-fold-01-seed-17.json          (UTF-16)
├── backtest-fold1-step2010-v1\
│   ├── results.json
│   └── predictions.jsonl.gz
└── failure-analysis-fold1-step2010-v4\diagnostics.json
```
