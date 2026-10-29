# CHMT-Q research workspace

This is the single canonical folder for the CHMT-Q documentation corpus. Start with the current publication package and
[`SESSION-CONTEXT.md`](SESSION-CONTEXT.md).

## Current authoritative artifacts

| Artifact | Purpose |
|---|---|
| [Publication manuscript](publication/manuscript.md) | Current empirical narrative and claim boundary |
| [Transformer architecture paper](publication/chmtq-transformer-architecture-paper.pdf) | Final arXiv-style mechanics paper: exact inputs, masks, tensor flow, outputs, and derivations |
| [Transformer architecture LaTeX](publication/chmtq-transformer-architecture-paper.tex) | Editable source for the architecture paper |
| [Transformer architecture source archive](publication/chmtq-transformer-architecture-source.zip) | Independently compiled upload bundle with the exact three figures |
| [Empirical arXiv-format PDF](publication/chmtq-arxiv-manuscript.pdf) | Single-column negative-results derivative with detailed mathematics |
| [Empirical arXiv LaTeX source](publication/chmtq-arxiv-manuscript.tex) | Editable source for the empirical derivative |
| [Empirical arXiv source archive](publication/chmtq-arxiv-source.zip) | Upload-ready source and figure bundle for the empirical derivative |
| [Submission PDF](publication/chmtq-negative-results-manuscript.pdf) | 21-page review/submission artifact |
| [Editable manuscript](publication/chmtq-negative-results-manuscript.docx) | Editable Word version |
| [Publication verifier](publication/verify_publication.py) | Evidence hashes, exact metrics, DOCX, and PDF checks |
| [Session handoff](SESSION-CONTEXT.md) | Repository scope, model facts, evidence limits, and continuation state |
| [Architecture specification](architecture/numeric-market-transformer-architecture.md) | System, capture, compiler, serving, and integration design |
| [Research specification](research-spec.md) | Selected CHMT-Q tensor, objective, and training contract |
| [June development report](okx-june-2026-development-backtest-report.md) | Historical development evaluation with updated holdout disclosure |

The architecture paper is a mechanism-only submission derivative. It does not
replace or weaken the empirical record: publication/manuscript.md and its
verifier remain the empirical claim boundary. The empirical arXiv-format files
preserve that record in a separate submission layout.

## Supporting and historical material

| Path | Contents |
|---|---|
| `from-first-principles.*` | Earlier beginner-first design tutorial in Markdown, DOCX, and PDF |
| `from-first-principles-assets/` | Fifty-six tutorial figures |
| `visual-guide.*` | Earlier visual guide in Markdown, DOCX, and PDF |
| `publication/assets/` | Eight current publication figures |
| `evidence/` | Frozen failure-analysis supplement and proposed DQN plan |
| `reports/` | Standalone visual backtest, failure-analysis, and DQN-plan PDFs |

The earlier tutorial contains a superseded 2,089,518-parameter design ledger and predates the final
implemented run. Use the current publication and session context for empirical or implementation
claims. The correct implemented parameter count is 2,089,554.

## Data boundary

Raw OKX archives, compiled Arrow shards, prediction tapes, trained weights, and resumable
checkpoints are intentionally not stored here. They remain local research artifacts because their
provenance/readiness does not support public redistribution or promotion. Hashes and regeneration
contracts are recorded in the publication.

## Verification

Create a local virtual environment from `requirements-publication.txt`, then run the publication
verifier. The verifier also expects the frozen local study directory at
`C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1`.
