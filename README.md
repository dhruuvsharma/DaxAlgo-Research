# DaxAlgo Research

Private research workspace: the code, experiments, and (in time) the papers and design records
behind DaxAlgo's model work. Extracted from the product repositories on 2026-08-11 so research
stops shipping inside an installer build.

**This repository ships nothing.** Nothing here is referenced by, built with, or installed
alongside DaxAlgo Terminal. Anything that graduates into the product moves to the appropriate
edition repository and is maintained there.

## Programmes

| Path | What |
|---|---|
| `chmt-q/numeric-transformer` | CHMT-Q numeric market transformer — model, training, and evaluation code |
| `chmt-q/research-backtest` | CHMT-Q research backtest harness |
| `fdt-n/volume-footprint-transformer` | FDT-N volume-footprint transformer — model, training, backtest, and instrument/timeframe screening |

## Where the rest of it lives

Documentation, papers, and evidence ledgers for these programmes are in the private
**DaxAlgo-Docs** repository under `chmt-q/` and `fdt-n/`. They are being consolidated here; until
that lands, treat the Docs copies as authoritative for written material and this repository as
authoritative for code.

The Hyperion expert-model work moved to the **Hyperion** repository instead — it is an agent
platform, not a market-model research programme.

## Conventions

- Frozen result sets that back a published claim are committed (for example FDT-N's BTC
  profitability study and common-timeframe screen), so the numbers stay reproducible.
- Checkpoints, raw datasets, and training runs are **not** committed; see `.gitignore`.
- Each programme keeps its own README describing how to reproduce its results.
