---
title: Do Ragged Volume-Footprint Transformers Beat Aggregate Linear Models? A Negative BTCUSDT Development Study
short_title: Negative BTCUSDT footprint study
author: Anonymous author(s)
affiliation: Anonymous review manuscript
date: 29 July 2026
keywords: volume footprint; order flow; quantile forecasting; transformers; set attention; market microstructure; reproducibility
---

# Do Ragged Volume-Footprint Transformers Beat Aggregate Linear Models? A Negative BTCUSDT Development Study

## Abstract

Executed-volume footprints preserve buy and sell volume at every traded price within a bar, creating
ragged sets that fixed aggregate features do not represent. We ask whether a compact hierarchical
transformer that consumes all accepted rows improves direct multi-horizon forecasting over aggregate
controls. FDT-N uses four learned row queries, gated fusion with 23 causal bar features, three masked
temporal blocks, and eight horizon queries to predict q10, q50, and q90 marginals for seven future
footprint variables. Its output parameterization prevents quantile crossing and orders the low,
point-of-control, and high marginals.

We evaluate three declared seeds on 14 contiguous days of public Binance BTCUSDT individual trades:
20,160 one-minute bars, 357,676 encoded rows, and 20,089 valid 64+8 windows. A purged chronological
split leaves 2,942 test windows. Across seeds, pooled normalized q50-forecast mean absolute error is
0.23643 +/- 0.00072. The validation-selected run records 0.23587, 7.25% below zero-change persistence
but 2.68% above a train-only direct ridge model. Ridge has lower error in 55 of 56 horizon-target
cells, and no paired four-hour block-resampling interval favors FDT-N over ridge. Pooled central-80%
marginal coverage is 78.98%, but cell values range from 74.75% to 85.66% and no separate calibration
set exists.

These findings do not identify a benefit from full footprint rows and do not support deployment or
trading value. The defensible contribution is a causal ragged-set formulation, a physically
constrained marginal-output contract, a hash-bound artifact trail, and an openly reported negative
development result. Independent periods, task-matched probabilistic baselines, and controlled
ablations are required.

## 1. Introduction

A conventional open-high-low-close-volume bar records the boundary and total activity of a trading
interval. A volume-footprint bar retains a more detailed object: executed buy-initiated and
sell-initiated volume at each traded price row, together with row delta, imbalance flags, point of
control (POC), value area, side-specific POCs, centroids, total volume, cumulative delta, feed quality,
and time. The object is attractive for market-microstructure research because it represents the
realized distribution of trading activity within a bar. It is also awkward for standard forecasting:
the number of price rows changes from bar to bar, the rows must remain causally sealed, and the
forecast should respect both probabilistic ordering and elementary candle geometry.

Limit-order-book studies have shown that local price-volume structure can be useful input to neural
models [11-14]. Their usual object, however, is displayed liquidity at fixed book levels, often paired
with a future mid-price-direction label. A footprint is executed flow on a variable price support.
Treating those modalities as interchangeable would obscure both the data-generating process and the
meaning of a forecast.

This paper studies the following narrower question: can a compact transformer consume the complete
accepted row set of sealed footprint bars and issue direct multi-horizon marginal quantiles for future
footprint-derived variables? We introduce FDT-N (Footprint Distribution Transformer for next-N bars) and make
four contributions.

1. We define an exact-row, causal forecast task with explicit coordinate, continuity, clipping, label,
   and rejection rules.
2. We combine set-style learned-query pooling [2,3] with a short causal temporal transformer [1],
   avoiding quadratic row-to-row attention and autoregressive forecast feedback.
3. We constrain the q10/q50/q90 outputs so quantiles do not cross and predicted price marginals obey
   low <= POCs <= high at each quantile.
4. We bind data, splits, runs, checkpoint, ONNX graph, runtime parity, and the manuscript to hashes,
   while reporting a negative comparison with a stronger linear control.

The fourth contribution is central to interpretation. FDT-N reduces pooled q50-forecast MAE relative
to persistence, principally through delta and relative-volume targets, but it does not outperform the
train-only direct ridge model. No experiment isolates the value of full rows, learned queries, gated
fusion, or the temporal transformer. The results therefore support implementation feasibility and a
reproducible experimental object, not a claim that architectural complexity improved forecasting.

## 2. Related work

### 2.1 Set-valued and ragged inputs

Deep Sets characterizes permutation-invariant functions through elementwise encoding and symmetric
aggregation [2]. Set Transformer replaces fixed pooling with learned attention blocks and inducing or
seed vectors [3], while Perceiver uses cross-attention into a learned latent bottleneck [23]. FDT-N
adopts the same broad latent-query principle for a different statistical object: each bar is a set of
executed price rows. Four learned queries attend once to the row embeddings, so attention-score count
grows linearly with the number of rows. The model does not assert that the learned queries have
human-interpretable footprint roles; attention weights alone are insufficient explanation [22].

### 2.2 Multi-horizon time-series forecasting

Temporal Fusion Transformer combines recurrent processing, attention, gating, and interpretable
multi-horizon outputs [6]. DLinear demonstrates that simple linear models can be difficult to beat in
long-horizon benchmarks [7], a warning directly reflected in our ridge result. PatchTST tokenizes
univariate patches [8], iTransformer attends across variables [9], and Moirai trains a universal
probabilistic forecaster across frequencies and variate counts [10]. These models motivate strong
comparators, but none natively defines a bar as a variable-size set of executed price rows. We do not
claim superiority over them; no matched adaptation was trained in this study.

### 2.3 Market-microstructure models

DeepLOB uses convolutional structure followed by temporal modeling on fixed-depth limit-order-book
states [11]. Transformer-based LOB models likewise operate on displayed levels or engineered book
features [12]. The FI-2010 benchmark formalized fixed-level mid-price prediction [13], while deep
order-flow-imbalance research has emphasized the strength of comparatively simple structural signals
at multiple horizons [14]. These results motivate spatial-before-temporal processing and strong linear
controls. They do not validate forecasts of executed-volume footprints, whose support, targets, and
measurement process differ.

### 2.4 Probabilistic evaluation under dependence

Quantile regression estimates conditional quantiles using the asymmetric pinball loss [15]. Proper
scoring rules reward honest probabilistic forecasts in expectation [16], while conformalized quantile
regression can adjust interval coverage when a suitable calibration design and assumptions are
available [17]. Financial forecast errors are serially dependent and overlapping horizons share
realizations. We therefore report paired block-resampling sensitivity summaries inspired by dependent
bootstrap methods [18], but we do not label them formal confidence intervals: only 13 blocks are
available, adjacent boundary windows overlap, and no multiplicity correction was applied. A later
confirmatory study should predeclare a dependence model and may use loss-differential procedures such
as Diebold-Mariano with explicit horizon and multiplicity handling [19].

## 3. Forecasting task and data

### 3.1 Sealed footprint bars

For bar t and price row j, let p[t,j] be the snapped price, v+[t,j] buy-initiated volume, and
v-[t,j] sell-initiated volume. Row volume and delta are

$$
v_{t,j}=v^+_{t,j}+v^-_{t,j},\qquad \Delta_{t,j}=v^+_{t,j}-v^-_{t,j}.
$$

A bar is sealed only after its interval ends. All row collections, aggregate features, references,
and quality fields used at cutoff t are functions of bars at or before t. A training window is valid
only when every timestamp difference across the 64 context bars and eight label bars equals exactly
one minute. Missing or irregular intervals are rejected rather than treated as adjacent market time.

Given lookback L=64 and horizon H=8, the model estimates

$$
\widehat{\mathbf{Y}}_{t+1:t+H}=F_\theta(\mathcal{B}_{t-L+1:t}),
$$

where each bar B contains its ragged row set and the causal bar vector. The output tensor has shape
[batch, 8, 7, 3]: eight horizons, seven target families, and three marginal quantiles.

### 3.2 Inputs

Each accepted row contributes 11 features: price offset from the bar POC; buy, sell, and total shares;
delta share; log buy and sell ratios; bid and ask imbalance flags; and zero-bid and zero-ask flags.
Price offsets are expressed in configured footprint-row increments. The encoded series stores all
rows in one flat array plus bar offsets. Windows and batches pad only to their widest member, and masks
exclude padding from attention.

Each bar contributes 23 causal features: total POC relative to the cutoff anchor; buy and sell POC
offsets; total, buy, and sell centroids; low, high, and value-area gaps; relative log volume; buy and
sell fractions; delta fraction; context-relative cumulative-delta change; stacked-buy and stacked-sell
fractions; row-count fraction; feed quality; and sine/cosine encodings of minute-of-day and weekday.
The price anchor is the last context POC. The volume reference is the median of log(1+volume) over at
most the final 16 context bars.

Rows are neither truncated nor compacted for accepted bars. A bar with more than 1,024 rows is
rejected. The observed maximum is 605, so this corpus has no row-bound rejection. This is an
exact-row-under-a-bound policy, not an unbounded-input claim.

### 3.3 Targets

Five price targets - total POC, low, high, buy POC, and sell POC - are direct offsets from the last
context POC, normalized by 64 footprint-row increments and clipped to [-2,2]. Volume is the future
log-volume difference from the context-only reference, scaled by 4 and clipped to [-2,2]. Delta is
the future buy-minus-sell fraction in [-1,1]. Side-POC cells are masked when the corresponding side
has no volume. Direct anchor-relative targets avoid reconstructing an absolute forecast by adding two
unrelated predicted marginals.

### 3.4 Frozen development corpus

The data source is the public Binance Data Vision BTCUSDT spot individual-trades archive [20]. Fourteen
daily archives from 1-14 January 2024 were aggregated into one-minute bars using a 2.5 USDT footprint
row. Quantities follow the terminal ingestion coordinate: quantity times 1,000, nearest-integer
ties-to-even rounding, and zero-size removal. The manifest records each archive name, original byte
count, and SHA-256. Raw archives are publicly reacquirable; local copies were removed after encoding.

| Corpus property | Value |
|---|---:|
| Instrument and venue | BTCUSDT spot, Binance |
| Period | 2024-01-01 00:00 to 2024-01-14 23:59 UTC |
| Bar interval / footprint row | 60 seconds / 2.5 USDT |
| Bars / encoded price rows | 20,160 / 357,676 |
| Rows per bar, p50 / p95 / p99 / maximum | 13 / 47 / 94.41 / 605 |
| Valid 64+8 windows | 20,089 |
| Train / validation / test windows | 13,990 / 2,869 / 2,942 |
| Purge at each split boundary | 72 bars |

The split is chronological: 70% train, 15% validation, and 15% test before two-sided purges equal to
lookback plus horizon. The test contexts run from 12 January 2024 22:50 UTC through 14 January 2024
23:51 UTC. There is no separate calibration partition or untouched outer period.

{{FIGURE:data_split|Figure 1. Frozen corpus and purged chronological split. Counts refer to valid 64-bar-context plus 8-bar-label windows after continuity screening.}}

## 4. FDT-N architecture

{{FIGURE:architecture|Figure 2. FDT-N architecture. Learned row queries summarize each variable-size executed-volume row set; gated bar tokens pass through causal temporal blocks, and parallel horizon queries decode all eight futures without feeding predictions back into the model.}}

### 4.1 Row encoder and learned-query pooling

Each 11-dimensional row vector x[t,j] is independently projected to model width d=64:

$$
e_{t,j}=\mathrm{RMSNorm}\!\left(W_2\,\mathrm{GELU}(W_1x_{t,j}+b_1)+b_2\right).
$$

RMS normalization follows Zhang and Sennrich [4]. Four learned query vectors attend to the row
embeddings. For query k and one attention head,

$$
\alpha_{t,k,j}=\mathrm{softmax}_j\!\left(\frac{(W_Qq_k)^\top(W_Ke_{t,j})}{\sqrt{d_h}}+m_{t,j}\right),\quad
a_{t,k}=\sum_j\alpha_{t,k,j}W_Ve_{t,j},
$$

where m masks padded rows. With K=4 queries and n[t] rows, the attention-score term is O(K n[t])
rather than O(n[t]^2). The four outputs are concatenated and projected to one row-set vector. The
queries are learned latent slots; intended labels such as total, buy, sell, and imbalance are not
treated as empirical interpretations.

### 4.2 Gated row/aggregate fusion

Let a[t] be the projected row-set summary and b[t] the projected 23-feature bar vector. Fusion is

$$
g_t=\sigma(W_g[a_t;b_t]+b_g),\qquad
u_t=g_t\odot a_t+(1-g_t)\odot b_t+W_r[a_t;b_t].
$$

The gate allows coordinate-wise weighting while the residual projection avoids forcing a convex
mixture. Whether this is better than concatenation, addition, or either branch alone is untested.

### 4.3 Causal temporal encoder and parallel horizon decoder

A learned position embedding is added to 64 fused bar tokens. Three pre-normalized blocks apply
four-head masked self-attention, SwiGLU feed-forward layers [5], dropout 0.10, and LayerScale initialized
at 0.01 [21]. A triangular mask prevents position i from attending to later positions even though all
64 tokens are historical at final inference; this preserves an online interpretation for intermediate
states.

Eight learned horizon queries then pass through two decoder blocks. The queries may attend to one
another because none contains a future realization or previous prediction. Each query cross-attends
to all encoded history states. The decoder therefore estimates all horizons directly:

$$
(\widehat{Y}_1,\ldots,\widehat{Y}_8)=F_\theta(X_{t-63:t}),\qquad
\widehat{Y}_h\not\leftarrow\widehat{Y}_{1:h-1}.
$$

This removes forecast-as-input error propagation; it does not make horizon errors independent.

| Architecture item | Value |
|---|---:|
| Model width / attention heads | 64 / 4 |
| Learned row queries | 4 |
| Temporal blocks / decoder blocks | 3 / 2 |
| Feed-forward multiplier / dropout | 3 / 0.10 |
| Lookback / forecast horizons | 64 / 8 bars |
| Trainable parameters | 371,861 |

### 4.4 Ordered marginal quantiles and physical constraints

For target k and horizon h, a head emits a center c and two positive increments produced by softplus.
The ordered quantiles are

$$
\widehat Q_{0.1}=c-\mathrm{softplus}(s^-),\quad
\widehat Q_{0.5}=c,\quad
\widehat Q_{0.9}=c+\mathrm{softplus}(s^+).
$$

For every quantile separately, low is the lower anchor, high is low plus a positive range, and each
POC is mapped through a sigmoid into that interval. This guarantees low <= total/buy/sell POC <= high
in normalized target space, a hard-parameterization alternative to post-hoc noncrossing adjustments
[24]. It does not define a coherent joint candle distribution: the outputs are seven constrained
marginal distributions, not sampled paths with learned dependence.

## 5. Training and evaluation

### 5.1 Objective and optimization

For quantile tau, prediction q, and target y, pinball loss is

$$
\rho_\tau(y-q)=\max\{\tau(y-q),(\tau-1)(y-q)\}.
$$

Loss is averaged over valid target cells and q10/q50/q90, with horizon weight proportional to
1/sqrt(h). Training uses AdamW, peak learning rate 3e-4, weight decay 1e-3, gradient clipping,
warm-up followed by cosine decay, deterministic algorithms, and early stopping with patience eight.
The batch item cap is 128, but row-token budgeting reduces item count around unusually wide windows
without deleting rows. Three declared seeds - 17, 29, and 43 - use identical data, configuration,
and parameter count.

The checkpoint with minimum validation weighted pinball loss is saved within each seed. The primary
run is the seed with minimum validation loss, tie-broken by seed number. No test score, plot, baseline
comparison, or runtime measure enters selection.

### 5.2 Baselines

Persistence predicts zero change from the context anchor for the five price offsets and uses causal
context references for the two flow targets. It has no fitted parameters. Direct ridge uses the latest,
context-mean, and context-trend bar aggregates to fit an independent point forecast for every
horizon-target cell. Ridge penalty alpha=1 is fixed and fitting uses training windows only. Both
baselines are evaluated on byte-identical test indices.

The baselines answer a limited question: whether FDT-N's q50 is better than simple causal point
predictions. They do not evaluate q10/q90 or proper distributional skill. No task-matched quantile
tree, aggregate-only neural model, sequence model, or adapted time-series foundation model was
trained. Consequently the paper does not claim probabilistic superiority.

### 5.3 Metrics and dependence treatment

We report normalized pinball loss at q10/q50/q90, MAE of the q50 forecast, signed q50 bias,
central-80% marginal coverage, mean q90-q10 width, and POC-offset sign agreement. The historical JSON
field `median_mae_normalized` means mean absolute error of the median (q50) forecast; it is not the
median of absolute errors.

For each seed and each of 56 horizon-target cells, we resample paired model-minus-baseline absolute
errors using 2,000 draws over 13 contiguous blocks of 240 test examples. A cell is described as
favoring FDT-N only when the complete 95% resampling interval is below zero. These are sensitivity
intervals, not independent-sample confidence intervals or significance tests. Adjacent windows at
block boundaries overlap; the block count is small; and no family-wise multiplicity correction is
applied. No pooled effect interval is computed.

## 6. Results

### 6.1 Seed selection and stability

Seed 17 is selected solely by validation loss. Pooled q50-forecast MAE varies by 0.00137 across seeds,
whereas POC-offset sign agreement ranges from 49.44% to 54.90%. The latter is a sign diagnostic for
normalized POC displacement, not candle direction, return direction, or trading performance.

| Seed | Best epoch | Validation loss | Test loss | Pooled q50 MAE | Central-80% coverage | POC sign agreement |
|---:|---:|---:|---:|---:|---:|---:|
| 17 | 21 | 0.112634 | 0.070392 | 0.235870 | 78.98% | 54.90% |
| 29 | 13 | 0.114391 | 0.071542 | 0.237238 | 78.96% | 49.44% |
| 43 | 20 | 0.113713 | 0.070793 | 0.236171 | 80.23% | 50.71% |
| Mean +/- sample SD | - | 0.113579 +/- 0.000886 | 0.070909 +/- 0.000584 | 0.236426 +/- 0.000719 | 79.39% +/- 0.73 pp | 51.69% +/- 2.86 pp |

The selected run's pooled q10/q50/q90 pinball losses are 0.055493, 0.117935, and 0.051387. Mean
normalized interval width is 0.720113 and pooled q50 bias is +0.034807.

### 6.2 Point-baseline comparison by target

{{FIGURE:target_mae|Figure 3. Selected-run q50-forecast MAE by target. Lower is better. FDT-N improves on persistence mainly for delta fraction and relative log volume, but direct ridge is lower for every target aggregate.}}

| Target | FDT-N | Persistence | Ridge | FDT - persistence | FDT - ridge | 80% coverage |
|---|---:|---:|---:|---:|---:|---:|
| POC offset | 0.2414 | 0.2432 | 0.2357 | -0.0017 | +0.0057 | 78.59% |
| Low offset | 0.2552 | 0.2423 | 0.2371 | +0.0129 | +0.0181 | 77.00% |
| High offset | 0.2272 | 0.2221 | 0.2225 | +0.0051 | +0.0047 | 78.87% |
| Buy-POC offset | 0.2431 | 0.2442 | 0.2374 | -0.0011 | +0.0058 | 78.31% |
| Sell-POC offset | 0.2415 | 0.2426 | 0.2359 | -0.0011 | +0.0056 | 80.15% |
| Relative log volume | 0.1219 | 0.1536 | 0.1198 | -0.0317 | +0.0021 | 80.70% |
| Delta fraction | 0.3207 | 0.4321 | 0.3197 | -0.1114 | +0.0011 | 79.23% |
| Pooled | 0.235870 | 0.254298 | 0.229719 | -0.018429 | +0.006151 | 78.98% |

Pooled FDT-N q50-forecast MAE is 7.25% lower than persistence and 2.68% higher than ridge. Excluding
delta and relative-volume targets, FDT-N is worse than persistence by 0.002826 averaged across the
remaining 40 horizon-target cells. The apparent pooled persistence gain is therefore not broad across
price-structure targets. At cell level, ridge has lower MAE in 55 of the 56 horizon-target cells.

### 6.3 Point-baseline comparison by horizon

{{FIGURE:horizon_mae|Figure 4. Q50-forecast MAE averaged over seven targets at each horizon. FDT-N is below persistence and above ridge at every horizon aggregate.}}

| Horizon | FDT-N | Persistence | Ridge | FDT - persistence | FDT - ridge | 80% coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1388 | 0.1577 | 0.1315 | -0.0189 | +0.0074 | 83.41% |
| 2 | 0.1843 | 0.2041 | 0.1782 | -0.0198 | +0.0061 | 79.45% |
| 3 | 0.2146 | 0.2344 | 0.2084 | -0.0198 | +0.0062 | 78.55% |
| 4 | 0.2383 | 0.2560 | 0.2310 | -0.0177 | +0.0073 | 78.80% |
| 5 | 0.2559 | 0.2731 | 0.2496 | -0.0173 | +0.0063 | 77.44% |
| 6 | 0.2732 | 0.2906 | 0.2663 | -0.0174 | +0.0070 | 77.64% |
| 7 | 0.2846 | 0.3038 | 0.2802 | -0.0192 | +0.0044 | 78.07% |
| 8 | 0.2973 | 0.3147 | 0.2926 | -0.0174 | +0.0046 | 78.45% |

{{PAGEBREAK}}

### 6.4 Marginal coverage and resampling summaries

{{FIGURE:coverage|Figure 5. Empirical central-80% marginal coverage by target and horizon for seed 17. Pooled coverage is near 80%, but cellwise variation and the absence of a separate calibration set preclude a general calibration claim.}}

Selected-run pooled central-80% marginal coverage is 78.98%; individual horizon-target cells range
from 74.75% to 85.66%. The artifacts do not record separate empirical cumulative calibration at q10,
q50, and q90. Accordingly, the result is reported as central-interval coverage near nominal in
aggregate, not as calibrated quantiles.

{{FIGURE:bootstrap|Figure 6. Counts of 56 horizon-target descriptive intervals by comparator and seed. No interval favors FDT-N over direct ridge. These dependent-block resampling summaries are descriptive, not formal confidence intervals.}}

| Seed | Comparator | Favors FDT-N | Spans zero | Favors comparator |
|---:|---|---:|---:|---:|
| 17 | Persistence | 21 | 21 | 14 |
| 29 | Persistence | 16 | 19 | 21 |
| 43 | Persistence | 18 | 21 | 17 |
| 17 | Direct ridge | 0 | 20 | 36 |
| 29 | Direct ridge | 0 | 25 | 31 |
| 43 | Direct ridge | 0 | 25 | 31 |

### 6.5 Clipping and runtime evidence

Price-target clipping affects approximately 1.04%-1.17% of observations, depending on target and
bound. Encoded row-price offset boundary saturation is 3.51%; the frozen dataset does not retain the
unclipped row coordinate needed to reconstruct exact pre-clip row loss. The reported metrics therefore
apply to the clipped normalized target space.

The validation-selected checkpoint is exported to ONNX. On one representative batch-one window with
187 rows at the widest bar, ONNX Runtime CPU output differs from PyTorch by 2.38e-7 maximum and
5.17e-8 mean absolute error, below the declared 1e-4 tolerance. Fifty isolated CPU runs yield
11.410/14.641/17.240 ms p50/p95/p99 latency. This is artifact timing, not end-to-end chart latency,
throughput, peak memory, operational safety, or production qualification.

## 7. Discussion

### 7.1 What the experiment supports

FDT-N shows that a variable-size executed-volume footprint can be represented without silent row
compaction, processed with bounded learned-query attention, and mapped to ordered multi-horizon
marginal quantiles in a compact 371,861-parameter model. Training is reproducible enough for three
seeds to produce tightly grouped test pinball loss and q50-forecast MAE. The exported graph has strong
cross-runtime numerical parity, and pooled central-80% coverage lies close to its nominal level.

The experiment does not show that the architecture improves forecasting over a strong simple model.
Direct ridge has lower q50-forecast MAE in every target aggregate, every horizon aggregate, and 55 of
56 individual horizon-target cells; no selected-run descriptive interval favors FDT-N over ridge.
This result is consistent with evidence
that linear controls can be difficult to beat on structured time-series tasks [7]. It also suggests
that the corpus is small relative to the model's flexibility, aggregate recency/trend features carry
much of the point-prediction signal, the current objective allocates capacity to interval prediction
rather than q50 alone, or the proposed row representation lacks useful incremental information.
These explanations are hypotheses, not conclusions.

### 7.2 Why the negative ridge result matters

Reporting only the persistence comparison would produce a misleading success narrative. The
persistence gain is concentrated in delta and relative volume; price extremes are worse, and total
and side POC improvements are small. Ridge narrows this gap with causal aggregate features and a
fraction of the model complexity. A useful next paper must therefore test whether full rows improve
proper distributional scores under matched budgets, rather than assuming that richer input is
beneficial.

### 7.3 Required next experiments

The highest-value next experiment is a preregistered factorial comparison on a later untouched
period: aggregate-only versus full-row input; fixed pooling versus learned queries; ridge/quantile
trees/MLP versus temporal convolution, recurrent, and transformer encoders; and point versus
distributional objectives. Hyperparameters and model-selection budgets should be matched on training
and validation periods. Evaluation should include a separate calibration interval, multiple market
regimes, additional instruments and venues, and a dependence unit chosen before the final test.

Economic value is a separate study. Any trading experiment would require a frozen decision rule,
latency model, fees, spread, slippage, fill uncertainty, position/risk constraints, and abstention
policy. Forecast error, sign agreement, ONNX latency, or central-interval coverage alone cannot support
a profitability or capital-allocation claim.

## 8. Threats to validity

**External validity.** The corpus contains one instrument, venue, row size, interval, and 14-day
period. The test portion covers approximately two days. Results do not establish cross-market,
cross-regime, seasonal, or long-run performance.

**Construct validity.** Footprint rows depend on trade classification, row snapping, quantity scaling,
and bar sealing. Buyer-is-maker flags, inferred aggressor rules, and synthetic pseudo-prints have
different error profiles. This study uses exchange-provided individual trades and does not validate
synthetic flow.

**Statistical validity.** Forecast windows and horizons overlap. The block summaries use only 13
blocks, retain overlap across block boundaries, and lack multiplicity correction. They are descriptive
sensitivity analyses. No untouched outer test or repeated chronological folds are available.

**Comparator validity.** Persistence and ridge provide point forecasts only. There is no matched
distributional baseline, aggregate-only ablation, or architecture ablation. The study cannot attribute
performance to full rows, learned queries, gated fusion, attention, or parallel horizon decoding.

**Probabilistic validity.** The q10/q90 outputs define marginal central intervals. Near-nominal pooled
coverage does not imply conditional calibration, and constrained marginals do not form a joint candle
distribution. A separate calibration set and task-matched proper-score baselines are absent.

**Preprocessing validity.** Target clipping and row-coordinate saturation discard tail information.
Exact pre-clip row-coordinate loss cannot be reconstructed from the frozen encoded series.

**Operational validity.** Numerical ONNX parity and isolated CPU timing do not establish production
latency, memory use, concurrency behavior, drift monitoring, or fail-safe system operation.

## 9. Reproducibility and artifact availability

The raw source archives are publicly available from Binance Data Vision [20]. The frozen manifest
records all 14 archive identities and hashes. The processed dataset, split files, results, checkpoint,
ONNX graph, metadata, and parity fixture are bound by SHA-256 in the accompanying research package.
Public distribution of private implementation code and processed artifacts is not asserted by this
anonymous manuscript; a submission requiring open artifacts must resolve that policy before
acceptance.

| Artifact | SHA-256 or identity |
|---|---|
| Source manifest | b88d89565febb97dbe56151fe447e3952fa844b8c433ecb86cb747e11e4ce412 |
| Encoded dataset | d83f78894af08b65123d2bf9f4fbef8ca28e901a552d3af2c61f6e37513bf6cc |
| Seed-17 / seed-29 / seed-43 results | 746403a05790753943b8c22f615f097c8dc397e413be49f35ea553b82e0cbd0a / 1b3e19b476b65e27becb30aaf600ea0384477b64b13f4d0d3eb2abddf3d0e0cc / 4d28b1e41c6f93e23f6c86261d7710a9974a58c5985aab7c029415f2a4e605ad |
| Selected checkpoint | 9d2cf525971f81fc93ca9ed97103cffb29203d7fd59edea5bd92d0e1480e124a |
| Selected ONNX graph | adce8cd39312bf3ad1389a23ed5ca24d6f47a7f58dbb26d76ae7d54af1011af7 |
| Selected metadata | 9f998ac6b56b008ac02aaa0ed5c4f08447b1735f33027294773d9e1aa0867763 |

Recorded software includes Python 3.12.6, PyTorch 2.5.1, NumPy 2.1.2, ONNX 1.22.0, and ONNX Runtime
1.28.0. Seeds, configuration, parameter count, best epoch, elapsed time, and device class are present
in the result bundle. The exact GPU model, original command lines, and a clean repository revision
snapshot were not frozen; these are reproducibility limitations.

## 10. Ethics, availability, and declarations

The study uses public market-event archives and contains no personal data or human participants. FDT-N
is a research forecaster. The selected artifact is marked shadow-only and does not create orders,
positions, or execution instructions. No result in this paper supports deployment, financial advice,
or capital allocation.

Raw-data availability and the current artifact policy are stated in Section 9. Author identity,
affiliation, ORCID, funding, competing interests, and author-contribution statements are withheld for
anonymous review rather than inferred. They must be supplied, together with any venue-specific data
and code statement, before non-blind publication.

## 11. Conclusion

FDT-N is a compact hierarchical transformer for exact-row-under-a-bound volume-footprint forecasting.
It combines learned-query set pooling, gated aggregate fusion, causal temporal attention, parallel
direct horizons, ordered marginal quantiles, and simple physical constraints. On a frozen 14-day
BTCUSDT development corpus, its pooled q50-forecast MAE is lower than persistence but consistently
higher than train-only direct ridge. Central-80% marginal coverage is near nominal in aggregate, while
cellwise variation, missing calibration data, and absent distributional baselines prevent a calibration
claim. The appropriate contribution is a reproducible formulation and honest negative learned-baseline
result. Establishing incremental value from full footprint rows requires preregistered ablations,
strong probabilistic controls, and independent multi-market replication.

## References

[1] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need. Advances in Neural Information Processing Systems. 2017;30. https://arxiv.org/abs/1706.03762

[2] Zaheer M, Kottur S, Ravanbakhsh S, Poczos B, Salakhutdinov R, Smola AJ. Deep Sets. Advances in Neural Information Processing Systems. 2017;30. https://arxiv.org/abs/1703.06114

[3] Lee J, Lee Y, Kim J, Kosiorek A, Choi S, Teh YW. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. Proceedings of Machine Learning Research. 2019;97:3744-3753. https://proceedings.mlr.press/v97/lee19d.html

[4] Zhang B, Sennrich R. Root Mean Square Layer Normalization. Advances in Neural Information Processing Systems. 2019;32. https://arxiv.org/abs/1910.07467

[5] Shazeer N. GLU Variants Improve Transformer. 2020. https://arxiv.org/abs/2002.05202

[6] Lim B, Arik SO, Loeff N, Pfister T. Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting. International Journal of Forecasting. 2021;37(4):1748-1764. https://doi.org/10.1016/j.ijforecast.2021.03.012

[7] Zeng A, Chen M, Zhang L, Xu Q. Are Transformers Effective for Time Series Forecasting? Proceedings of the AAAI Conference on Artificial Intelligence. 2023;37(9):11121-11128. https://doi.org/10.1609/aaai.v37i9.26317

[8] Nie Y, Nguyen NH, Sinthong P, Kalagnanam J. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. International Conference on Learning Representations. 2023. https://arxiv.org/abs/2211.14730

[9] Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. International Conference on Learning Representations. 2024. https://arxiv.org/abs/2310.06625

[10] Woo G, Liu C, Kumar A, Xiong C, Savarese S, Sahoo D. Unified Training of Universal Time Series Forecasting Transformers. International Conference on Machine Learning. 2024. https://arxiv.org/abs/2402.02592

[11] Zhang Z, Zohren S, Roberts S. DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. IEEE Transactions on Signal Processing. 2019;67(11):3001-3012. https://doi.org/10.1109/TSP.2019.2907260

[12] Wallbridge J. Transformers for Limit Order Books. 2020. https://arxiv.org/abs/2003.00130

[13] Ntakaris A, Magris M, Kanniainen J, Gabbouj M, Iosifidis A. Benchmark Dataset for Mid-price Forecasting of Limit Order Book Data with Machine Learning Methods. Journal of Forecasting. 2018;37(8):852-866. https://doi.org/10.1002/for.2543

[14] Kolm PN, Turiel J, Westray N. Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book. Mathematical Finance. 2023;33(4):1044-1081. https://doi.org/10.1111/mafi.12413

[15] Koenker R, Bassett G. Regression Quantiles. Econometrica. 1978;46(1):33-50. https://doi.org/10.2307/1913643

[16] Gneiting T, Raftery AE. Strictly Proper Scoring Rules, Prediction, and Estimation. Journal of the American Statistical Association. 2007;102(477):359-378. https://doi.org/10.1198/016214506000001437

[17] Romano Y, Patterson E, Candes E. Conformalized Quantile Regression. Advances in Neural Information Processing Systems. 2019;32. https://arxiv.org/abs/1905.03222

[18] Kunsch HR. The Jackknife and the Bootstrap for General Stationary Observations. The Annals of Statistics. 1989;17(3):1217-1241. https://doi.org/10.1214/aos/1176347265

[19] Diebold FX, Mariano RS. Comparing Predictive Accuracy. Journal of Business and Economic Statistics. 1995;13(3):253-263. https://doi.org/10.1080/07350015.1995.10524599

[20] Binance Data Vision. BTCUSDT spot daily individual-trades archive, 1-14 January 2024. https://data.binance.vision/?prefix=data/spot/daily/trades/BTCUSDT/

[21] Touvron H, Cord M, Sablayrolles A, Synnaeve G, Jegou H. Going Deeper with Image Transformers. IEEE/CVF International Conference on Computer Vision. 2021:32-42. https://doi.org/10.1109/ICCV48922.2021.00043

[22] Jain S, Wallace BC. Attention is not Explanation. Proceedings of NAACL-HLT. 2019:3543-3556. https://doi.org/10.18653/v1/N19-1357

[23] Jaegle A, Gimeno F, Brock A, Vinyals O, Zisserman A, Carreira J. Perceiver: General Perception with Iterative Attention. Proceedings of Machine Learning Research. 2021;139:4651-4664. https://proceedings.mlr.press/v139/jaegle21a.html

[24] Bondell HD, Reich BJ, Wang H. Noncrossing Quantile Regression Curve Estimation. Biometrika. 2010;97(4):825-838. https://doi.org/10.1093/biomet/asq048
