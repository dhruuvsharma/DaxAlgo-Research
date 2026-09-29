# CHMT-Q Tiny retrospective BTC result

**Status:** development evidence from an already studied June 2026 corpus. This is not a sealed,
confirmatory, executable-return, trading, profitability, or promotion result.

## Run contract

- Model: CHMT-Q Tiny, 282,838 trainable parameters (86.46% fewer than the 2,089,554-parameter
  reference model).
- Inputs: bars `[64,16]`, L1 `[240,16]`, L2 `[240,2,10,12]`, quality `[16]`.
- Output: five horizons by 17 raw values, reconstructed as five q10/q50/q90 targets plus two
  presence slots.
- Training: 1--18 June (25,758 examples), validation: 19--20 June (2,862),
  development-forward: 21--24 June (5,724).
- Seed 17; eight epochs completed; epoch 5 selected by validation pinball.
- Presence loss weight was frozen to zero because every valid training presence label was positive.
  The two shape-preserving presence slots are therefore inactive and unvalidated.

## Aggregate result

| Partition | Tiny | Train-only empirical marginal | Relative improvement |
|---|---:|---:|---:|
| Validation | 0.202214 | 0.222446 | 9.0951% |
| Development-forward | 0.242370 | 0.257193 | 5.7633% |

The metric is mask-normalized, horizon-and-target-weighted normalized pinball loss; lower is better.
Tiny was lower than the baseline in all 25 horizon-target cells. Its aggregate q10--q90 coverage was
80.19%, versus 83.68% for the empirical baseline. Descriptively, Tiny's development score was 3.59%
lower than the previously recorded 0.251391 score of the full CHMT-Q model, but this is one seed and
not a new independent market sample.

## Concrete input/output trace

For `okx-btc-usdt-20260621T000500Z`, only four early-session bar rows were valid, while all 240 L1
snapshots and all 4,800 L2 side-rank cells were valid. This produced 124 eligible memory tokens. The
quality-bounded modality gates were bars 0.3630, L1 0.2245, and L2 0.4125, summing to one. The raw
output shape was `[5,17]`.

The first-horizon physical output illustrates the q10/q50/q90 contract:

| Target | q10 | q50 | q90 | Realized |
|---|---:|---:|---:|---:|
| Total POC offset (ticks) | -116.149 | 8.017 | 133.046 | 0.500 |
| Buy POC offset (ticks) | -115.785 | 15.923 | 141.257 | 0.500 |
| Sell POC offset (ticks) | -97.812 | 26.542 | 160.080 | 72.500 |
| Log-volume residual | -0.656 | -0.098 | 0.853 | -0.099 |
| Delta fraction | -0.866 | -0.071 | 0.799 | 0.441 |

This row is an execution trace, not a representative-performance claim.

## Verification and artifacts

- All 168 Arrow shards were schema-, size-, and SHA-256-verified: 42,930 rows and 1,724,748,416
  compressed bytes.
- Batch-one latency was 40.65 ms on the local GTX 1650 while the GPU was shared with another user
  process; two repeated inference calls were bit-identical (`max_abs_difference = 0`).
- Checkpoint SHA-256: `351ca13c779863b5f99cd74491506b1814ac487efc2a7e408219f8a51a63eeb5`.
- Result JSON SHA-256: `7ac87484ee6ff102acb8412fe48b942afe16305963c3784211b3caf44cbdbd61`.
- Full result and checkpoint remain outside Git at
  `C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-tiny-retrospective-v2\seed-17`.

The separately frozen July--August protocol requires five seeds and one-time development-forward
scoring before any broader conclusion.
