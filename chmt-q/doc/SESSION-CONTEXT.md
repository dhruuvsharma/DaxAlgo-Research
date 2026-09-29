# CHMT-Q session handoff

**Updated:** 2026-08-08 Europe/London

**Repository:** `D:\Github\DaxAlgo Docs`

**Remote:** `https://github.com/dhruuvsharma/DaxAlgo-Docs.git`

**Branch:** `main`

**Canonical workspace:** `chmt-q/` at the repository root

## User direction

- Continue all CHMT-Q work only in the DaxAlgo Docs repository.
- Do not write, commit, or push CHMT-Q changes in `DaxAlgo-Terminal-Pro`.
- Keep every publishable CHMT-Q document, figure, evidence summary, report, and context handoff in
  this folder.
- Preserve unrelated dirty files elsewhere in this repository.

## What CHMT-Q is

CHMT-Q is the **Causal Hierarchical Market Transformer with Quantiles**. “Causal” means temporal
availability and masking, not causal inference. “Q” means quantiles, not Q-learning.

Implemented input contract:

- bars: `[B,64,16]`;
- level 1: `[B,240,16]`;
- level 2: `[B,240,2,10,12]`;
- quality: `[B,16]`;
- five learned horizon queries;
- output: `[B,5,17]`, comprising five targets × q10/q50/q90 plus two presence logits;
- main width 128, four heads, feed-forward width 256, and 2,089,554 trainable parameters.

The five targets are total, buy, and sell point-of-control offsets, residual log volume, and delta
fraction. They are not OHLC candles, executable returns, or trade actions.

## Training and artifact state

- Dataset: OKX BTC-USDT spot, June 2026 UTC, local non-promotable research use.
- Compiled examples: 42,930 across 30 sessions.
- Training: 25,758 examples from 18 sessions.
- Selection: 2,862 examples from two sessions.
- Development-forward evaluation: 5,724 examples from four sessions.
- Seed: 17; one fold; selected epoch 10 at optimizer step 2,010.
- Weight artifact: 8.071 MiB; resumable checkpoint: 24.284 MiB.
- The model has no proven live-terminal integration or production deployment status.

## Current empirical result

On the frozen four-session development-forward partition:

- weighted normalized pinball: model `0.2513909348275368` versus empirical baseline
  `0.25719276632184623`;
- relative improvement: `2.2558299664809045%`;
- H5 total-POC median MAE: 532.841 versus 532.905 ticks, only 0.0119% better;
- H5 Pearson: 0.0774; Spearman: 0.0753; direct-sign MCC: 0.0481;
- predicted/realized standard-deviation ratio: 0.1513;
- every H5 q10–q90 total-POC interval crossed zero, so the registered interval policy made zero
  trades;
- diagnostic median long-flat: 484 trades, +2.1607% frictionless, −68.0682% at 12 bps per side;
- gross long-flat edge: 0.4453 bps per round trip; base cost was 53.89× break-even.

Conclusion: CHMT-Q improved an aggregate proper score but did not establish useful conditional
location forecasts, executable edge, or profitability. It remains research-only and
non-promotable.

## Holdout integrity

The June 25–30 partition was unused and unscored in the historical backtest. A later schema/count
projection traversed all 42,930 rows and physically read projected target fields in those shards.
No sealed values were printed, retained, analyzed, or used. The period is nevertheless
**process-accessed and retired**. It must never be described as sealed, untouched, or confirmatory.
A new future study needs a newly acquired access-controlled holdout.

## Transformer–DQN status

The Dueling Double-DQN revision is a research plan only. Its status is
`proposed_not_implemented_not_validated`. No DQN has been built, trained, backtested, or integrated.
The proposed actions are target positions `{-1,0,+1}` and evaluation must use raw net PnL separate
from shaped reward. Eligibility requires direct-return utility, strong simple baselines, realistic
costs, at least five rolling folds, multiple Transformer and DQN seeds, a new ≥60-session sealed
holdout, and ≥60 shadow days. Reinforcement learning cannot create missing information or eliminate
transaction costs.

## Publication package

The venue-neutral negative-results manuscript contains 7,165 words, 8 figures, 10 tables, and 30
sequential cited references. It was rendered to a 21-page PDF and an editable DOCX.

- DOCX SHA-256: `a16854543dea66a51a3efa19e4d9a2117e34a93ef56a984eb9f687b28c4e7e21`
- PDF SHA-256: `d161468e1c59c0163422e84aaefbd660aa6eb7d749ef70d95f4e126ee6cc8ac5`
- Accessibility audit at creation: zero high and zero medium findings.
- Submission still needs real author/affiliation/ORCID details, funding and competing-interest
  declarations, venue formatting, and a human verification of all claims.

The earlier `from-first-principles` tutorial predates the implemented run and contains a stale
2,089,518 parameter count. Do not use it to override the current publication or exact implementation
ledger.

## arXiv-format derivative

On 8 August 2026, a single-column arXiv-format derivative was added at
publication/chmtq-arxiv-manuscript.tex and publication/chmtq-arxiv-manuscript.pdf, with an upload
bundle at publication/chmtq-arxiv-source.zip. It preserves the empirical conclusions while expanding
the formal information set, tensor path, attention and fusion equations, quantile-loss derivation,
evaluation estimands, cost arithmetic, dependence analysis, parameter ledger, and clearly prospective
Double-DQN mathematics. The PDF is 21 US-Letter pages with 8 figures, 12 tables, and 30 references.

- LaTeX SHA-256: 2c33935e7071c0ae5ca90d0a2e60603dbddd39b30020fae306de0a0c5e6a66c0
- PDF SHA-256: 6deeaaf73ab31c8677945993d5e80a45acbb3f1d81a48f24ced7f31fc7904bf9
- Source archive SHA-256: 30924ba19b90abf845dd086b91a168aa052d2fdbd3291ef8770a5ad176e253a4

The derivative names Dhruv Sharma as author. It still requires affiliation, ORCID, correspondence,
funding, competing-interest, contribution, venue, and generative-tool declarations. The Markdown
manuscript and publication verifier remain the empirical claim boundary.

## Transformer architecture submission

On 8 August 2026, a separate architecture-first submission was added at
publication/chmtq-transformer-architecture-paper.tex and
publication/chmtq-transformer-architecture-paper.pdf, with the upload bundle
publication/chmtq-transformer-architecture-source.zip. Its title and running header are exactly
**CHMT-Q: Causal Hierarchical Market Transformer with Quantiles**, and the author is Dhruv Sharma.

This mechanism-only paper traces the complete computation:

- four value tensors and four authoritative masks, including every indexed feature;
- robust scaling, clock-boundary ownership, physical-time encoding, and structural embeddings;
- Q/K-normalized attention, total safe softmax, SwiGLU, and LayerScale;
- 64 unpatched bar tokens, 60 L1 tokens, and 60 hierarchical L2 tokens;
- quality-aware fusion into [B,184,128];
- two-block decoding from five learned horizon queries;
- the exact [B,5,17] raw index map, ordered q10/q50/q90 construction, presence probabilities,
  inverse target scaling, and physical reconstruction;
- the complete composite loss and 2,089,554-parameter derivation.

The final PDF is 16 US-Letter pages with 3 figures, 9 tables, 73 labeled equations, and 13
references. All pages were rendered and visually inspected. The three computational diagrams use a
shared minimalist visual system. The source archive contains only the current TeX and its three
referenced PNGs and compiles independently to 16 pages. The final critic gauntlet scores were
architecture/mathematics 100/100, writing/structure 99/100, and scientific integrity 98/100, with
no critical defects. After the minimalist figure redesign, the diagram-only re-audit scored
architecture semantics 98/100 and visual quality 100/100.

- LaTeX SHA-256: cd667f412f7e6710925672dea3917d41b3c9cd75872cbfc9c62ac5e07e664c88
- PDF SHA-256: 1e886c3cb9486f4432d427767878636fba73467dade2a7187d6eb2eab2d2b9a9
- Source archive SHA-256: 2bdeb17092489fa7f4bd5e335e328bac30d0418e45ec1eb652041666abe8ce3e

The paper reports architecture rather than empirical performance. It states the research-only,
non-promotable status and does not claim a full predictive distribution, calibrated coverage,
live deployment, executable edge, or profitability. publication/manuscript.md and its verifier
remain the empirical claim boundary. Before external submission, the author should confirm the
affiliation and correspondence line and supply any venue-required ORCID, funding,
competing-interest, contribution, and generative-tool declarations.

## Local evidence and immutable identifiers

The verifier expects local frozen study artifacts under:

`C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1`

Important identifiers:

- dataset: `565c1d8e912ee426a1a92572d5be481ec5168d54d1ae53bd77fdd70d888e4df6`;
- run: `25be2f446a75ad37120bba956ea86e46c0ecefa004ab3343d6745cdbee102fb3`;
- study protocol: `885fcd0fee23115a4a5480ec94d95f436e251ec0caf796f9686867ed7e72f5d0`;
- prediction tape: `27d9ffbe41b18868b8a45bf9c48c936908de7394dd6ecfec7062cc531cc7e1f0`;
- results: `083da170146585cf1a16dde56e2da02697194640ed7b053d3298d8e3775bee6a`;
- failure diagnostics: `ece5b18f8a8455f7edac8491b6d23362fa14f618129b1193b1febf14bc89505e`;
- DQN plan: `d5e81f203619c55754c940df7e92ec5995e4db41b3a7d6f3db4137318f7a502c`.

Do not commit the raw archives, Arrow dataset, 26 MiB prediction tape, weights, or checkpoint without
an explicit provenance and release decision.

## Repository state at handoff

Before consolidation, unrelated user work existed at:

- `pro/claude-code/Strategy_Protection_Pipeline_V2.md` (modified);
- `pro/architecture/adr-unified-strategy-execution-pipeline.md` (untracked);
- `pro/claude-code/spec/daxq-format-v2.md` (untracked).

These files are outside CHMT-Q scope and must remain untouched. The CHMT-Q corpus was first
consolidated from `pro/research/numeric-market-transformer/`, `pro/architecture/`, and `output/pdf/`,
then moved to the repository-root `chmt-q/` folder for direct access. The stale Microsoft Word lock
file `~$om-first-principles.docx` was removed.

## Continuation protocol

1. Read `AGENTS.md`, this file, and `README.md`.
2. Inspect `git status --short --branch` and isolate CHMT-Q changes from unrelated work.
3. Treat `publication/manuscript.md` plus immutable evidence as the current truth source.
4. Run `publication/verify_publication.py` after publication changes; render and inspect every DOCX
   and PDF page when layout changes.
5. Keep claims bounded: one venue, one instrument, one month, one seed, one fold, four forward days.
6. Commit and push only the DaxAlgo Docs repository when the user explicitly requests it.
