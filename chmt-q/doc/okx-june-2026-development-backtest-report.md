# CHMT-Q BTC-USDT June 2026 development backtest

**Report date:** 2026-07-26  
**Study ID:** `086950d1edf9d23ac0ba1a0bb8bbda2a20adf53a3ca4e749afdb2a7395e3f6c1`  
**Status:** Local, non-promotable development research  
**Instrument:** OKX BTC-USDT spot coordinate  
**Model:** CHMT-Q v1, seed 17, fold-1 checkpoint step 2,010  
**Primary test window:** 2026-06-21 through 2026-06-24 UTC  
**Capital illustration:** 100,000 USDT  
**Outer period:** 2026-06-25 through 2026-06-30 was unused by this historical study, but a later
target-bearing projection physically read those rows; it is process-accessed and retired from
confirmatory use. No values were retained or analyzed.

## Executive conclusion

The current model shows a small probabilistic-forecast improvement, but this study does **not**
support a profitable-trading claim.

- On the only untouched forward window available to this checkpoint, normalized weighted pinball
  loss was **0.251391**, versus **0.257193** for the training-only empirical-marginal baseline: a
  **2.26% relative improvement**. A paired UTC-day block bootstrap gave an empirical-minus-model
  effect of 0.005802 with a nominal 95% interval of **[0.000470, 0.012607]**. There are only four
  distinct forward days, so this interval is suggestive rather than confirmatory.
- CHMT-Q does not predict OHLC candles. Its five-minute total-POC median was only a weak proxy for
  cutoff-to-close direction: **52.67% accuracy**, **52.36% balanced accuracy**, **MCC 0.0475**, and
  **Spearman correlation 0.0780**. An always-down majority rule would score 52.39% accuracy on the
  same rows, so the model added only 0.28 percentage points of raw accuracy.
- The preregistered uncertainty-aware long/flat policy made **zero trades**. Every five-minute q10/q90
  total-POC interval straddled zero on the forward window. Its result was therefore 0 USDT, equal to
  cash, not evidence of trading profit.
- Aggressive median-only policies were positive before costs: +2.16% compounded for long/flat and
  +4.99% for hypothetical long/short over four days. Their break-even costs were only **0.223** and
  **0.215 basis points per side**. Under the preregistered base assumption of 12 bps per side, the
  compounded results became **-68.07%** and **-93.34%**. The apparent frictionless edge is far too
  small for the tested turnover.
- The six newest days, 8,586 examples, were not scored or used by this run. They were later
  process-accessed by a target-bearing schema/count projection and are now retired. A truthful
  full-month out-of-sample claim therefore cannot be made from this run.

### Decision card

| Gate | Result | Decision |
|---|---:|---|
| Direct probabilistic forecast vs empirical baseline | 2.26% lower forward weighted pinball | Promising but small |
| Median total-POC point error | Essentially tied at five minutes; worse at 1–3 minutes | No clear point-forecast edge |
| Candle-direction proxy | 52.67%, MCC 0.0475 at five minutes | Too weak for a directional claim |
| Uncertainty-aware primary strategy | 0/1,148 scheduled trades | Safe abstention; no profit evidence |
| Aggressive strategy after base costs | -68.07% / -93.34% compounded | Reject |
| Statistical depth | One seed, four forward days | Insufficient |
| Data provenance/readiness | Explicit research exception; incomplete | Non-promotable |
| Live trading | Not tested or authorized | Prohibited |

The appropriate portfolio decision is **do not deploy and do not allocate capital**. Continue only as
a research program, using a direct return/candle target, materially longer pre-June training history,
multiple seeds, a frozen policy, and an executable bid/ask replay.

## 1. What was tested

### 1.1 Model contract

CHMT-Q consumes:

- 64 one-minute bar tokens with 16 features each;
- 240 half-second L1 tokens with 16 features each;
- 240 half-second L2 snapshots, two sides by ten levels by 12 features;
- 16 data-quality features.

For each of the next five one-minute intervals it predicts marginal q10, q50, and q90 values for:

1. total footprint POC offset from the causal anchor, in 0.10-USDT ticks;
2. buy POC offset;
3. sell POC offset;
4. log-volume residual;
5. delta fraction;

plus two side-presence probabilities. These are five independent horizon marginals. They are not a
joint price path, a return distribution, an OHLC forecast, a probability that price will rise, or an
order instruction. Buy/sell POC presence means that an eligible side POC exists; it does not mean
buy/sell market direction.

### 1.2 Frozen checkpoint

| Item | Value |
|---|---:|
| Trainable parameters | 2,089,554 |
| Buffer elements | 164 |
| Total state elements | 2,089,718 |
| Weight file | 8,463,430 bytes / 8.071 MiB |
| Full resumable checkpoint | 25,463,382 bytes / 24.284 MiB |
| Seed | 17 |
| Precision | deterministic FP32 |
| Training microbatch / accumulation / effective batch | 32 / 4 / 128 |
| Best epoch / checkpoint step | 10 / 2,010 |
| Completed epochs / optimizer steps | 18 / 3,618 |
| Stop condition | early stop after eight non-improving validations |
| Aggregate recorded epoch time | 7,316.5 seconds / 2.03 hours |
| Mean training throughput | 63.37 examples/second |
| Peak CUDA memory | 1.241 GiB |

The run used Python 3.12.6, NumPy 2.1.2, Torch 2.5.1+cu121, CUDA 12.1, cuDNN 90100, and an NVIDIA
GeForce GTX 1650 on Windows 11. No exported or promoted package was created.

## 2. Data, provenance, and evidence classes

### 2.1 Corpus

The compiled corpus covers all 30 UTC days of June 2026 and contains 42,930 examples, 1,431 per day.
Cutoffs run from 00:05 through 23:55 UTC. The compiler used genuine OKX trades for labels and official
400-level archive updates for L1/L2 inputs. The full compiler audit observed 16,918,630 trade rows and
145,978,684 L2 rows. This study revalidated the raw source inventory and loaded 13,967,248 trades into
36,000 exact UTC-minute summaries for June 1–24. No trade-ID gaps or timestamp reversals were found.

### 2.2 Chronological partition

| Evidence class | Dates UTC | Examples | Permitted interpretation |
|---|---|---:|---|
| Fold-1 training replay | Jun 1–18 | 25,758 | In-sample description only |
| Fold-1 selected validation | Jun 19–20 | 2,862 | Used for early stopping/checkpoint selection; optimistic |
| Untouched development forward | Jun 21–24 | 5,724 | Only genuinely unseen window for this checkpoint |
| Retired outer period | Jun 25–30 | 8,586 | Not used or scored by this study; later process-accessed; no result |

The generated tape contains 34,344 predictions. An explicit ID allow-list admitted only fold-1 train,
fold-1 validation, and fold-2/3 validation IDs; any outer-holdout intersection failed the run. The
historical manifest records `outer_holdout_accessed=false` for this run. It does not override the
later process-integrity disclosure or make the period eligible for confirmatory use.

### 2.3 Known source limitations

The dataset remains non-promotable despite the user's local research authorization:

- the archives do not contain an independent local receive/availability timestamp;
- historical L2 records lack native sequence/checksum evidence;
- the instrument contract was captured as current metadata rather than an effective-dated contract;
- the dataset capability record remains `training_ready=false`, `rights_confirmed=false`, and
  `provenance_complete=false`;
- the user authorization permits local training and validation while explicitly denying export,
  serving, promotion, and outer-holdout access.

The absence of an independent receive clock means that causality is best-effort at equal timestamps.
The economic replay adds a one-minute entry delay, but that cannot repair source-time provenance.

## 3. Study design and leakage controls

The study protocol was written and hashed before inference. Its SHA-256 is
`885fcd0fee23115a4a5480ec94d95f436e251ec0caf796f9686867ed7e72f5d0`.

Primary endpoints were fixed as:

- forecast: normalized weighted pinball difference against the training-only empirical marginal;
- economics: five-minute interval long/flat policy, base 12-bps one-way cost, untouched Jun 21–24;
- capital scale: 100,000 USDT.

All five policy variants and all four cost scenarios are reported. None was selected after seeing the
forward result. The empirical-marginal and zero-residual forecast baselines were fit only on fold-1
training data. No June 25–30 data was used for policy selection, calibration, scoring, or pricing.

The remaining selection risks are material:

- the checkpoint itself was selected on Jun 19–20;
- there is only one seed and one trained fold;
- the architecture and targets were designed before this report but not against a separate pre-June
  corpus;
- several exploratory policy variants are shown, so only the preregistered primary policy should be
  read as the primary economic endpoint;
- the four-day test is too short for regime, tail, or annualized-risk inference.

## 4. Probabilistic forecast results

### 4.1 Aggregate normalized score

Lower weighted pinball is better. Horizon weights are `1/sqrt(h)` and target weights are
`[1, 1, 1, 0.5, 0.5]`.

| Evidence class | Model | Empirical | Zero residual | Relative model improvement | Paired effect 95% day-block interval |
|---|---:|---:|---:|---:|---:|
| Training replay | 0.310221 | 0.320658 | 0.468357 | 3.25% | [0.005063, 0.017130] |
| Selected validation | 0.210765 | 0.222446 | 0.306795 | 5.25% | [0.006488, 0.016873] |
| Untouched development forward | 0.251391 | 0.257193 | 0.370622 | 2.26% | [0.000470, 0.012607] |

The reduction survives all four forward days in the paired bootstrap, but four blocks are not enough
to estimate robust sampling uncertainty. The drop from 5.25% on selected validation to 2.26% on the
untouched window is consistent with selection optimism and/or regime drift.

### 4.2 Forward physical scorecard

POC cells are in 0.10-USDT ticks. Log-volume residual and delta fraction are dimensionless. Coverage
is the raw q10–q90 central interval; nominal coverage is 80%.

| H | Target | Model MAE | Empirical MAE | Model pinball | Empirical pinball | Coverage | Width | Bias |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Total POC | 176.563 | 174.177 | 60.264 | 64.372 | 82.22% | 595.549 | -4.032 |
| 1 | Buy POC | 181.771 | 179.679 | 62.420 | 66.493 | 82.44% | 613.730 | -0.771 |
| 1 | Sell POC | 181.655 | 176.244 | 61.754 | 65.209 | 82.69% | 607.138 | 20.842 |
| 1 | Log volume | 0.510 | 0.506 | 0.161 | 0.164 | 77.83% | 1.685 | 0.063 |
| 1 | Delta fraction | 0.549 | 0.513 | 0.155 | 0.146 | 94.76% | 1.864 | 0.277 |
| 2 | Total POC | 316.781 | 315.865 | 104.690 | 111.262 | 83.58% | 1,098.978 | -11.360 |
| 2 | Buy POC | 319.913 | 319.313 | 105.789 | 112.979 | 85.05% | 1,138.233 | -19.583 |
| 2 | Sell POC | 324.574 | 316.878 | 105.928 | 111.721 | 82.84% | 1,088.662 | 57.081 |
| 2 | Log volume | 0.553 | 0.520 | 0.175 | 0.169 | 76.12% | 1.823 | 0.139 |
| 2 | Delta fraction | 0.585 | 0.513 | 0.162 | 0.146 | 97.61% | 1.929 | 0.388 |
| 3 | Total POC | 403.533 | 402.359 | 134.697 | 141.718 | 83.05% | 1,425.945 | -15.320 |
| 3 | Buy POC | 405.762 | 405.182 | 134.840 | 143.024 | 85.36% | 1,482.480 | -28.426 |
| 3 | Sell POC | 415.888 | 404.961 | 135.686 | 142.452 | 82.11% | 1,392.740 | 85.079 |
| 3 | Log volume | 0.594 | 0.529 | 0.188 | 0.172 | 74.30% | 1.963 | 0.219 |
| 3 | Delta fraction | 0.632 | 0.513 | 0.171 | 0.146 | 99.00% | 1.964 | 0.502 |
| 4 | Total POC | 471.085 | 471.262 | 159.349 | 165.670 | 81.87% | 1,681.448 | -13.267 |
| 4 | Buy POC | 474.065 | 474.639 | 158.311 | 166.819 | 87.00% | 1,793.519 | -27.538 |
| 4 | Sell POC | 485.876 | 471.906 | 158.898 | 166.015 | 83.56% | 1,640.636 | 111.043 |
| 4 | Log volume | 0.633 | 0.536 | 0.201 | 0.174 | 70.49% | 2.044 | 0.291 |
| 4 | Delta fraction | 0.686 | 0.513 | 0.180 | 0.146 | 99.67% | 1.983 | 0.602 |
| 5 | Total POC | 532.841 | 532.905 | 181.677 | 186.570 | 81.15% | 1,904.669 | -10.118 |
| 5 | Buy POC | 534.435 | 535.820 | 178.364 | 187.426 | 86.65% | 2,014.257 | -31.349 |
| 5 | Sell POC | 552.933 | 533.163 | 180.001 | 186.585 | 83.84% | 1,865.951 | 143.401 |
| 5 | Log volume | 0.674 | 0.542 | 0.216 | 0.176 | 67.40% | 2.175 | 0.368 |
| 5 | Delta fraction | 0.740 | 0.513 | 0.190 | 0.146 | 99.91% | 1.992 | 0.688 |

The model's POC pinball improves on the empirical baseline at every horizon, mainly through its
interval forecasts. Median point accuracy does not show the same pattern: total-POC MAE is worse at
one to three minutes, nearly tied at four minutes, and better by only 0.063 tick at five minutes.
Sell-POC, volume, and delta medians degrade materially with horizon.

At five minutes, total-POC median absolute error is 532.84 ticks, or 53.28 USDT. The error distribution
has a 36.76-USDT median, 121.81-USDT p90, and 260.07-USDT p99. Only 0.07% of predictions are within one
tick and 0.19% within two ticks. Exact-tick accuracy is not a useful success criterion for this target.

### 4.3 Calibration and sharpness

For total POC, empirical `P(y <= q10) / P(y <= q50) / P(y <= q90)` was:

| Horizon | q10 CDF | q50 CDF | q90 CDF | Central coverage | Mean width ticks / USDT |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.09% | 50.52% | 90.30% | 82.22% | 595.55 / 59.55 |
| 2 | 5.94% | 50.51% | 89.52% | 83.58% | 1,098.98 / 109.90 |
| 3 | 5.05% | 50.10% | 88.10% | 83.05% | 1,425.94 / 142.59 |
| 4 | 4.47% | 50.66% | 86.34% | 81.87% | 1,681.45 / 168.14 |
| 5 | 4.05% | 51.54% | 85.20% | 81.15% | 1,904.67 / 190.47 |

Central coverage is close to 80%, but the tails become asymmetric and the bands become very wide.
The delta-fraction intervals cover 94.76–99.91% while their widths expand from 1.864 to 1.992 on a
quantity bounded to [-1, 1]. They are therefore close to the entire legal range and provide little
selective information. The growing positive delta bias, 0.277 to 0.688, is a serious calibration flaw.

Buy/sell POC presence is nearly deterministic in this liquid one-minute corpus. Near-zero model log
loss is not a directional achievement; the labels are almost always present and the empirical
baseline is already near one.

No separate chronological conformal-calibration slice was available. These are raw model intervals.

## 5. Future-candle proxy analysis

The following analysis does **not** turn CHMT-Q into a candle model. It asks whether the predicted
total-POC median displacement is associated with three downstream signs:

- actual footprint total-POC displacement, the direct target;
- the future minute's candle body, `sign(close - open)`;
- future close relative to the causal anchor, `sign(close - anchor)`.

| H | Direct POC accuracy / MCC | Candle-body accuracy / MCC | Anchor-close accuracy / MCC | Anchor Pearson | Anchor Spearman |
|---:|---:|---:|---:|---:|---:|
| 1 | 53.20% / 0.0624 | 51.75% / 0.0456 | 52.52% / 0.0495 | 0.0476 | 0.0642 |
| 2 | 51.85% / 0.0335 | 49.83% / 0.0068 | 51.96% / 0.0346 | 0.0648 | 0.0670 |
| 3 | 52.34% / 0.0416 | 50.72% / 0.0239 | 52.15% / 0.0381 | 0.0711 | 0.0698 |
| 4 | 52.88% / 0.0528 | 50.79% / 0.0256 | 52.60% / 0.0470 | 0.0737 | 0.0752 |
| 5 | 52.76% / 0.0481 | 50.68% / 0.0239 | 52.67% / 0.0475 | 0.0797 | 0.0780 |

At five minutes the anchor-close confusion matrix, actual rows by predicted columns, is:

| Actual / predicted | Down | Flat | Up |
|---|---:|---:|---:|
| Down | 1,769 | 0 | 1,230 |
| Flat | 0 | 0 | 0 |
| Up | 1,479 | 0 | 1,246 |

The five-minute accuracy's nominal 95% UTC-day-block interval is [51.35%, 53.53%]. With only four
days it does not establish a persistent edge. The model never emitted a median-flat direction, and its
q10/q90 POC interval never sat wholly above or below zero. Confidence-filtered direction coverage was
therefore 0% at every horizon.

For an AI researcher, the result says that optimizing the current footprint pinball objective did not
produce a strong representation of candle returns. A direct log-return or OHLC-distribution head,
jointly evaluated but kept distinct from footprint outputs, is required for a genuine candle claim.

## 6. Trading policy and execution proxy

### 6.1 Causal sequence

`cutoff t -> h=5 forecast -> frozen rule -> enter at minute open t+1 -> exit at minute close t+4`

Decisions were sampled every five minutes, so trades could not overlap. Position size was one fixed
initial-capital notional, with no leverage, pyramiding, stop, take-profit, or forecast-dependent size.
The primary spot-compatible rule was:

> Long only if the five-minute total-POC q10 offset is above zero; otherwise hold cash.

Exploratory rules used the median or a q10/q90 long-short decision. Long-short results are hypothetical
margin/perpetual proxies; they are not directly executable on BTC-USDT spot and omit borrow, funding,
liquidation, and margin constraints.

### 6.2 Price and cost assumptions

Entry and exit prices are first/last genuine trade prices aggregated to a minute, not executable ask
and bid quotes. The base case approximates a 10-bps taker fee plus 2-bps slippage per side. This is an
assumption, not measured transaction-cost analysis.

| Scenario | One-way fee + slippage assumption | Round trip |
|---|---:|---:|
| Frictionless | 0 bps | 0 bps |
| Optimistic | 9 bps | 18 bps |
| Base | 12 bps | 24 bps |
| Stress | 20 bps | 40 bps |

Spread, queue position, partial fills, capacity, impact, adverse selection, routing latency below the
one-minute delay, financing, funding, and tax are not modeled. Real execution is likely worse than
the frictionless and minute-price scenarios.

## 7. Economic results

### 7.1 Primary policy

There were 1,148 scheduled non-overlapping decisions on Jun 21–24. The interval long/flat policy took
zero positions because every q10/q90 interval crossed zero. Its gross return, costs, net return,
drawdown, and P&L are all zero under every scenario. This is a valid abstention result.

Cash also returned zero. Buy-and-hold over the same outer price endpoints returned -4.97% before
costs and **-5.21%** at base costs, or -5,209 USDT per 100,000 USDT. The primary policy avoided that
loss but did not demonstrate forecast monetization.

### 7.2 Aggressive policies and gross-to-net waterfall

| Policy | Active trades | Gross return / P&L per 100k | Base modeled costs | Base fixed-notional net P&L | Frictionless compounded | Base compounded | Break-even one-way cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Median long/flat | 484 | +2.155% / +2,155 | 116,160 | -114,005 | +2.16% | -68.07% | 0.223 bps |
| Median long/short | 1,148 | +4.937% / +4,937 | 275,520 | -270,583 | +4.99% | -93.34% | 0.215 bps |

The fixed-notional calculation uses 100,000 USDT on every trade. A result below -100,000 USDT would
require replenishing capital and is not a feasible self-financing account path. The compounded column
shrinks notional after losses and is the better capital-survival illustration. Neither calculation
models margin rules.

### 7.3 Cost sensitivity

| Policy | Frictionless | Optimistic 9 bps/side | Base 12 bps/side | Stress 20 bps/side |
|---|---:|---:|---:|---:|
| Median long/flat compounded | +2.16% | -57.28% | -68.07% | -85.32% |
| Median long/short compounded | +4.99% | -86.73% | -93.34% | -98.95% |

For median long/flat, gross daily fixed-notional returns were +1.256%, +1.013%, -0.758%, and +0.644%.
After base modeled costs they became -34.02%, -25.39%, -28.60%, and -26.00%. For median long/short,
base daily results were between -65.98% and -69.80%. Annualized Sharpe and Calmar values from four
days are mathematically computable but economically meaningless, so they are not used for a decision.

The gross average edge was about 0.445 bps per long/flat round trip and 0.430 bps per long/short round
trip. Base round-trip cost was 24 bps—more than 50 times the observed gross edge. This is the central
economic result.

## 8. Statistical assessment

### 8.1 What the bootstrap does and does not show

The report resamples complete UTC days with replacement, 10,000 times, seed 20,260,726. Pairing model
and baseline scores within the same sampled day preserves their dependence. The positive pinball
interval indicates consistent relative scoring across the four observed days, not a general market
law.

Four blocks cannot estimate:

- monthly or annual regime variation;
- tail loss, drawdown, VaR, or CVaR with useful precision;
- execution-cost dispersion;
- performance decay after repeated retraining;
- model-selection or data-snooping bias;
- probability of backtest overfitting across policy families.

No p-value is presented. There was no correction for the 25 horizon-target cells because they are
reported descriptively, with one preregistered aggregate forecast endpoint. A future confirmatory
study should predeclare one primary hypothesis, use substantially more independent day/session
blocks, and apply family-wise or FDR control to secondary cells.

### 8.2 Stability and seed uncertainty

This is one seed and one trained fold. Neural initialization variance is unmeasured. A credible model
comparison requires at least three seeds per fold, reporting all outcomes and a paired fold/session
analysis rather than the best seed. The existing checkpoint selection also makes Jun 19–20 unsuitable
for final inference.

## 9. Interpretation by role

### AI researcher

- The probabilistic objective learned some POC interval information beyond a static marginal, but
  point accuracy and downstream direction remain weak.
- Delta forecasts are biased and nearly maximally wide. Volume forecasts degrade with horizon.
- Presence is a degenerate task on these one-minute samples.
- Add direct return/candle targets only with a clearly separate output contract; do not reinterpret
  POC quantiles as return probabilities.
- Run modality ablations, multiple seeds, conformal calibration, reliability analysis, and stronger
  linear/tree/TCN/DeepLOB-style baselines under identical splits.

### Quant researcher

- The only honest forward sample has four days. Treat every economic result as exploratory.
- The observed gross edge is roughly 0.43–0.45 bps per round trip, far below plausible taker costs.
- The uncertainty-aware policy correctly abstains; the median policy's turnover converts a weak
  signal into a large loss.
- Future testing needs executable bid/ask event replay, measured latency, spread and slippage,
  capacity/participation limits, funding for shorts, and prediction-time artifact replay.
- Register cash, buy-and-hold, simple momentum/mean-reversion, ridge/logistic, boosted-tree, and causal
  neural baselines before the next final holdout.

### Portfolio manager

- Allocate 0 USDT. There is no evidence of net profitability at plausible costs.
- Do not use the frictionless +2.16% or +4.99% as an expected return; both are erased by costs more
  than 50 times larger than the average edge.
- The primary policy's 0-P&L outcome is cash, not alpha.
- Risk numbers from four days must not be annualized for capital allocation, leverage, or limits.
- Require a longer external test, cost-qualified positive return, drawdown/tail evidence, and an
  operational risk review before reconsidering.

### Mathematician/statistician

- The direct loss is proper marginal quantile scoring, but the five marginals do not specify a joint
  stochastic process.
- Day-block resampling respects within-day overlap, yet four blocks give a highly discrete and fragile
  empirical distribution.
- Direction accuracy is dominated by weak class imbalance; MCC near 0.05 is a more informative signal
  of limited association.
- The q10/q90 bands achieve aggregate coverage partly by becoming broad; coverage must always be read
  with width/sharpness.
- Model selection, single-seed uncertainty, multiple exploratory policies, and nonstationarity remain
  outside the reported interval.

## 10. Mathematical definitions

For target `y`, quantile `q_tau`, and residual `u = y - q_tau`, pinball loss is

`rho_tau(u) = u * (tau - 1[u < 0])`.

The aggregate score weights horizon `h` by `1/sqrt(h)` and targets by 1 for the three POCs and 0.5 for
volume and delta. Median MAE, RMSE, bias, central coverage, interval width, presence log loss, and Brier
score use only cells whose explicit supervision masks are true.

For a completed fixed-horizon observation with direction `d` in `{-1, 0, +1}`:

`gross_return = d * (P_exit / P_entry - 1)`

`net_return = gross_return - 2 * one_way_cost`.

Fixed-notional P&L is `C0 * sum(net_return_i)`. The reported compounded path is
`product(1 + net_return_i) - 1`. Turnover counts two notional sides per active round trip. The cost
model is deliberately simple and is not an order-book fill model.

Balanced accuracy is the mean class recall. Multiclass MCC is reported because it incorporates all
confusion cells and remains informative under imbalance. Pearson measures linear association;
Spearman is Pearson correlation of average ranks.

## 11. Engineering and operational assessment

The study verified the exact source tree, dependency lock, dataset, authorization, split, scaler, run
directory, selected checkpoint, and every checkpoint file before inference. Predictions were emitted
chronologically to a compressed immutable row-level tape. The standalone harness lives outside the
frozen training source tree, so it did not alter checkpoint identity.

What is still missing for terminal use:

- a signed, promoted model package and reviewed export path;
- a scaler-wrapped inference artifact with PyTorch/ONNX parity evidence;
- a causal live compiler with independent receive timestamps;
- measured model, IPC, routing, and broker latency distributions;
- timeout, stale-data, abstention, fallback, drift, and rollback behavior;
- an offline signal adapter into the terminal backtester with executable quote replay;
- risk limits and a paper-trading stage;
- explicit authorization for export, serving, promotion, and any later live use.

This study intentionally adds no live order path to DaxAlgo Terminal.

## 12. Reproducibility ledger

| Artifact | SHA-256 or identity |
|---|---|
| Dataset | `565c1d8e912ee426a1a92572d5be481ec5168d54d1ae53bd77fdd70d888e4df6` |
| Research authorization | `85a6589d87b2346a05ed32acd875026c44d642bef31c246cecc5584b02a853f4` |
| Split manifest | `7f5254f62a3d70a6f02e41b1991f25d3bd8338e7aa9b25c5fa19737f70a7bb8e` |
| Outer holdout | `284e69e35c110f7e3cb3dc41dbe57da52c0a35650962d5d8f23a4e04236b5cff` |
| Fold-1 scaler | `29a733fac9986eb7e879555b6813f8d35b06c98dee9041057e6fc4287bd50880` |
| Experiment/run ID | `25be2f446a75ad37120bba956ea86e46c0ecefa004ab3343d6745cdbee102fb3` |
| Frozen source tree | `7bd3153371f1e09d0aa33d924c5487111efd36589b6b4a498fe78175a48670d3` |
| Dependency lock | `48792cf2789a576553fc29b1b727e97342d28fed1fdbeb91c31135a84bfff898` |
| Checkpoint weights | `e6cd2195e332010192316e05220ef7eb85e0c3761cc3f021f059d8a393fff2ad` |
| Study protocol | `885fcd0fee23115a4a5480ec94d95f436e251ec0caf796f9686867ed7e72f5d0` |
| Prediction tape | `27d9ffbe41b18868b8a45bf9c48c936908de7394dd6ecfec7062cc531cc7e1f0` |
| Aggregate results | `083da170146585cf1a16dde56e2da02697194640ed7b053d3298d8e3775bee6a` |

Local result directory:

`C:\DaxAlgoData\okx\BTC-USDT\2026-06-chmtq-research-v1\backtest-fold1-step2010-v1`

It contains:

- `study-protocol.json`: frozen endpoints, policies, costs, and artifact identities;
- `predictions.jsonl.gz`: 34,344 row-level physical forecasts, labels, masks, and timestamps;
- `results.json`: complete forecast, proxy, economic, daily, risk, and bootstrap tables;
- `manifest.json`: file sizes, hashes, study ID, and holdout-use status.

## 13. Required next experiment

To answer the original “one month” question honestly:

1. Acquire enough provenance-improved history before June—preferably several months spanning multiple
   volatility regimes.
2. Train expanding walk-forward folds and at least three seeds without using June as training or model
   selection data.
3. Add and preregister a direct return/candle distribution head if candle prediction is the objective.
4. Fit calibration and action thresholds only on pre-June development data.
5. Freeze one candidate, one policy, one primary baseline, costs, latency, capacity, and risk rules.
6. Replay all of June once as an external month using timestamped executable bid/ask quotes.
7. Report all seeds/folds, the no-trade baseline, cost and capacity stress, day/session block uncertainty,
   and every attempted policy.
8. Acquire a new access-controlled holdout; Jun 25–30 is retired and cannot be restored to sealed
   status by policy or relabeling.

Until those steps are complete, CHMT-Q should remain a local forecast-research artifact, not a trading
system.
