# When Better Proper Scores Do Not Imply Tradable Edge

## A Leakage-Aware Failure Analysis of a Causal Hierarchical Market Transformer with Quantile Outputs

**Anonymous manuscript for peer review**  
Author, affiliation, ORCID, and corresponding-author details withheld pending submission format.  
Manuscript version 1.0 — 29 July 2026

{PAGEBREAK}

## Abstract

Forecasting studies in electronic markets often report predictive loss without establishing whether the learned conditional distribution contains an executable signal. We document this gap through a bounded negative-result study of CHMT-Q, a 2,089,554-parameter causal hierarchical market Transformer with quantile outputs. CHMT-Q consumes 64 minutes of bars, 120 seconds of level-1 and ten-level level-2 state, and 16 data-quality features. It emits five one-minute-ahead marginal distributions for three point-of-control offsets, residual log volume, and delta fraction. The model was trained once on 18 UTC sessions of OKX BTC-USDT spot data from June 2026, selected on two sessions, and evaluated on four development-forward sessions. The remaining six sessions cannot serve as a confirmatory holdout because a later schema/count projection physically read their target fields, although no values were retained or analyzed.

On 5,724 development-forward examples, CHMT-Q reduced weighted normalized pinball loss from 0.257193 for an empirical marginal baseline to 0.251391, a 2.26% relative improvement. That aggregate result did not transfer to the intended economic use. At the five-minute horizon, total point-of-control median mean absolute error improved by only 0.012%; Pearson and Spearman associations were 0.077 and 0.075; the predicted-to-realized standard-deviation ratio was 0.151; and every nominal 10%–90% interval crossed zero. The preregistered uncertainty-aware policy therefore made zero trades. A post hoc median-sign long-flat policy earned 2.16% before costs but lost 68.07% under a 12-basis-point one-way cost assumption across 484 trades. Its measured gross edge was 0.445 basis points per round trip, 53.9 times smaller than modeled costs.

The audit identifies task–decision misalignment, cost dominance, weak and shrunk conditional medians, severe dependence from overlapping windows, narrow validation, target redundancy, saturated delta tails, persistent gradient clipping, and weak baselines. These observations support a methodological conclusion rather than a profitability claim: a better proper score is insufficient evidence of tradable edge. We provide a falsifiable corrective protocol and treat a cost-aware Transformer–Double-DQN architecture only as future work.

**Keywords:** limit order book; probabilistic forecasting; quantile regression; Transformer; market microstructure; backtest overfitting; negative results; offline reinforcement learning

{FIGURE:evidence_map.png|Figure 1. Evidence and claim boundary. Only frozen development artifacts support the reported findings; the retired holdout and proposed reinforcement-learning study support no performance claim.|Flow diagram separating observed data, implemented CHMT-Q, development evaluation, bounded findings, retired holdout, and proposed future work.}

## 1. Introduction

Modern limit-order-book models can optimize a statistically valid forecasting objective while remaining economically unusable. The distinction matters because a strictly proper scoring rule rewards a forecast distribution for representing the evaluated target distribution [16], whereas a trading decision depends on a different object: an executable, conditional return distribution net of spread, fees, slippage, latency, queue effects, and risk. A lower forecasting loss is therefore neither a theorem nor reliable empirical evidence of positive net profit.

This paper studies that distinction with CHMT-Q, the Causal Hierarchical Market Transformer with Quantiles. Here, “causal” means that every attention path and feature obeys temporal availability; it does not denote causal inference or an identified treatment effect. CHMT-Q was designed to preserve the geometry of a ten-level order book, combine it with level-1 and bar histories, and emit noncrossing quantiles at five horizons. Its architecture draws on sequence attention [1,2], time-series patching and multi-horizon forecasting [5–8], and hierarchical treatment of limit-order-book state [9–15,18].

The experiment appears favorable if summarized by one number: on the development-forward partition, aggregate weighted normalized pinball loss was 2.26% lower than an empirical marginal baseline. A deeper audit reverses the practical interpretation. The conditional medians were strongly shrunk, the forecast target was not executable return, uncertainty intervals admitted no trades, and aggressive median-sign policies turned over far too often for their small gross edge. Moreover, the nominal row count substantially overstated independent information because consecutive examples shared 63 of 64 bar observations and four of five labels at the longest horizon.

We ask four research questions:

1. **RQ1 — Distributional score:** Does CHMT-Q improve its registered probabilistic forecasting metric on a temporally later development partition?
2. **RQ2 — Conditional signal:** Does the improvement correspond to useful median, rank, directional, or uncertainty information at the intended five-minute horizon?
3. **RQ3 — Economic feasibility:** Can the forecasts support a decision rule whose measured gross edge is plausibly larger than transaction costs?
4. **RQ4 — Failure mechanism:** Which data, target, optimization, validation, and execution factors explain the gap without making causal claims unsupported by ablations?

The contributions are deliberately narrow:

- a fully specified multimodal quantile Transformer and immutable experiment ledger;
- a separation of proper-score, conditional-signal, and economic claims;
- an audit showing how a positive aggregate score can coexist with zero uncertainty-authorized trades and negative cost-adjusted returns;
- an integrity disclosure that retires a nominal holdout after process-level access; and
- a registered corrective design whose reinforcement-learning component remains unimplemented and unvalidated.

The study is not a live-trading test, not a comparison against production-grade execution, not evidence that Transformers are generally unsuitable for order-book forecasting, and not evidence that reinforcement learning will repair the observed failure.

## 2. Related Work

### 2.1 Order-book representation and multi-horizon prediction

DeepLOB established a strong convolutional–recurrent baseline for extracting spatial and temporal structure from limit-order-book tensors [9]. Subsequent work used attention and Transformers for order books [10,11], standardized prediction benchmarks [12], linked order-flow imbalance to multi-horizon return predictability [13], and emphasized that market, label, horizon, and evaluation choices materially change apparent performance [14,15]. Universal price-formation work further suggests that some representations transfer across instruments, but it does not establish that a particular target or execution rule transfers [18].

CHMT-Q follows the representational lesson—book rank, side, and time should not be flattened indiscriminately—but differs in objective. It forecasts footprint primitives rather than a direct mid-price class or executable return. That choice is central to the result: representation quality cannot compensate for a target that is only weakly coupled to the downstream decision.

### 2.2 Probabilistic forecasting

Quantile loss is a proper score for conditional quantiles, and proper scoring rules provide a coherent basis for evaluating predictive distributions [16]. Noncrossing parameterizations can impose ordered quantiles, while conformal methods can calibrate intervals under stated exchangeability or adaptation assumptions [17]. CHMT-Q uses an ordered quantile head but does not claim distribution-free coverage. The reported 10%–90% interval coverage is descriptive, and no conformal correction was fit on the four-day forward partition.

Time-series Transformers use multiple strategies for separating variables, patching time, and decoding horizons [5–8]. The mixed record of complex models against linear baselines [8] makes strong simple baselines especially important. In the present run, only empirical-marginal and zero-residual baselines were registered. The absence of ridge, boosted-tree, convolutional, and recurrent competitors prevents an architectural superiority claim.

### 2.3 Backtest validity and reinforcement learning

Financial model selection is vulnerable to repeated testing, dependence, nonstationarity, and transaction costs. Reality-check, multiple-testing, and backtest-overfitting methods address only parts of this problem [19,21,29], while stationary block resampling preserves local dependence more faithfully than row-wise resampling [20]. Gap-separated validation is also relevant when observations share features or labels [30]. Four development-forward sessions are too few for stable daily-block inference; accordingly, we report descriptive estimates and do not treat bootstrap intervals as confirmatory.

Deep Q-learning [22], Double DQN [23], dueling value–advantage decomposition [24], distributional Q-learning [25], and conservative offline objectives [26] provide relevant tools for a future decision model. Yet deep reinforcement learning is known to exhibit high variance and fragile reproducibility [28], and offline reinforcement learning introduces extrapolation error when learned policies choose poorly supported actions [27]. These methods cannot manufacture information absent from the state or overcome an unfavorable cost-to-edge ratio by optimization alone.

## 3. Claim Taxonomy and Study Design

We distinguish three claim levels. A **forecast claim** concerns a frozen score on a specified target. A **conditional-signal claim** concerns associations, median errors, directional discrimination, or interval selectivity. An **economic claim** concerns a fixed decision rule evaluated with an explicit execution and cost model. Evidence at one level does not automatically support the next.

| Claim level | Registered question | Evidence available | Defensible conclusion |
|---|---|---|---|
| Forecast distribution | Is weighted normalized pinball lower than the empirical marginal baseline? | 5,724 development-forward examples | Yes, by 2.26% on this bounded partition |
| Conditional signal | Are H5 medians and intervals informative enough to select actions? | MAE, correlations, sign metrics, dispersion, interval width | Weak median/rank signal; intervals never exclude zero |
| Economic value | Does a fixed policy retain positive return after costs? | Frictionless and fixed-bps simulations | No under the tested assumptions |
| Generalization | Does performance persist across regimes, seeds, venues, or instruments? | One month, one instrument, one venue, one seed, one fold | Not established |
| Deployment | Is the model executable in a live terminal? | Offline Python model and replay artifacts | Not established; no live integration |

{FIGURE:study_timeline.png|Figure 2. Chronological evidence partition for June 2026. The final six sessions are retired from confirmatory use because a later projection physically read target fields.|Horizontal timeline showing 18 training sessions, 2 selection sessions, 4 development-forward sessions, and 6 retired holdout sessions.}

### 3.1 Data source and provenance

The source corpus comprises official OKX historical archives for BTC-USDT spot, June 2026 UTC: 31 level-2 archive files and 31 trade archive files containing 149,390,769 raw level-2 records and 16,918,630 trades. Compilation produced 42,930 examples across 30 UTC sessions in 168 Arrow shards (1.612 GiB). The dataset identifier is `565c1d8e912ee426a1a92572d5be481ec5168d54d1ae53bd77fdd70d888e4df6`.

The corpus has material provenance limitations. It lacks local receive timestamps, native sequence numbers, and checksums; historical instrument metadata was not verified as June-effective; and the original license/provenance review was incomplete. The data owner explicitly authorized local, non-promotable research with those limitations. We therefore make no claim about live receive-time causality, book reconstruction under packet loss, or redistribution rights.

### 3.2 Temporal partitions and holdout integrity

The frozen run used 25,758 training examples from 18 sessions, 2,862 selected-validation examples from two sessions, and 5,724 development-forward examples from four sessions. The 8,586 examples in the final six sessions were initially designated an outer holdout and were not used in the historical study. Later, however, a schema/count projection traversed all 42,930 Arrow rows and physically read projected target fields in those shards. No sealed values were printed, retained, analyzed, or used, but the physical read violates a strict process-level holdout policy. We therefore retire the period from confirmatory use and report no result from it.

This distinction is not semantic. A holdout is an operational control, not merely a promise not to optimize after seeing a metric. Future confirmatory evidence requires a newly acquired, access-controlled period with audit logs that prohibit any target-bearing read before preregistration is locked.

### 3.3 Evaluation protocol

The model checkpoint was selected by validation loss at epoch 10, optimizer step 2,010. Predictions were made once per minute. Economic decisions were sampled every five minutes. The execution proxy entered at the next minute open and exited at the close four minutes later. Notional was fixed; leverage, pyramiding, and endogenous market impact were excluded. Cost scenarios applied 0, 9, 12, or 20 basis points per side. The 12-basis-point one-way scenario was designated the base case.

The primary forecasting endpoint was weighted normalized pinball loss relative to an empirical marginal forecast fit only on permitted data. The primary economic rule was an interval-based long-flat policy: trade only when the predicted uncertainty interval supported one sign. Median-sign long-flat and long-short rules were diagnostic, not confirmatory. Daily-block bootstrap output used 10,000 replicates, but with only four forward days it is not interpreted as reliable inferential evidence.

## 4. CHMT-Q Architecture

{FIGURE:architecture.png|Figure 3. Implemented CHMT-Q v1 architecture. Book rank is encoded before time; modality memories are fused with quality context and decoded by five horizon queries.|Architecture diagram showing bar, L1, L2, and quality inputs; hierarchical encoders; fusion; horizon decoder; ordered quantile and presence heads.}

### 4.1 Input and output contract

Each example has four numeric inputs. Bars have shape `[64,16]`; level-1 observations have shape `[240,16]`; level-2 observations have shape `[240,2,10,12]`, representing time, side, rank, and feature dimensions; and quality context has shape `[16]`. The level-1 and level-2 streams span 120 seconds at nominal half-second resolution. Four-frame temporal patches reduce the modality sequence length. After modality encoders and fusion, the decoder attends with five learned horizon queries.

The raw output has shape `[5,17]`. For each horizon, 15 values define three quantiles for five continuous targets and two logits define buy/sell presence. The targets are total point-of-control offset, buy point-of-control offset, sell point-of-control offset, log-volume residual, and delta fraction. They are footprint descriptors, not OHLC values, direct returns, or trade actions.

| Component | Tensor or setting | Implemented value |
|---|---|---|
| Bar history | steps × features | 64 × 16 |
| L1 history | steps × features | 240 × 16 |
| L2 history | steps × sides × levels × features | 240 × 2 × 10 × 12 |
| Quality vector | features | 16 |
| Shared model width | `d_model` | 128 |
| Attention / feed-forward | heads / width | 4 / 256 |
| L2 level encoder | width / heads / feed-forward | 32 / 4 / 64 |
| Encoder depth | bar / L1 / L2 temporal | 3 / 3 / 3 blocks |
| Decoder | horizon queries / blocks | 5 / 2 |
| Quantiles | levels | 0.10, 0.50, 0.90 |
| Parameters | trainable total | 2,089,554 |

### 4.2 Causal hierarchical encoders

Bars and level-1 inputs are projected to width 128, augmented with learned modality and temporal encodings, patched over four frames, and processed by three pre-normalized causal attention blocks. Each block uses four attention heads, RMS normalization [3], residual connections, and a gated feed-forward sublayer [4]. Causal masks prevent a token from attending to a later token.

The level-2 branch first processes the 20 side–rank positions within each snapshot at width 32. Side and rank embeddings preserve bid/ask identity and distance from the top of book. A learned summary then maps each snapshot to width 128. The resulting temporal sequence is patched and processed by three causal blocks. This order—rank before time—prevents the book’s price ladder from being treated as an arbitrary flat feature vector.

The encoded bar, L1, and L2 memories contain 184 tokens in total. A quality-conditioned fusion stage modulates their contribution rather than treating missingness or degraded snapshots as ordinary market information. Two decoder blocks use five learned horizon queries to attend to the fused memory directly, avoiding autoregressive exposure bias across the five output horizons.

### 4.3 Ordered quantile head

For each horizon and target, the network predicts a center `c` and two unconstrained gaps `g−` and `g+`. The ordered outputs are

{EQUATION:q_0.50 = c,    q_0.10 = c − softplus(g−),    q_0.90 = c + softplus(g+).}

The construction prevents quantile crossing by design. Delta-fraction outputs are subsequently mapped through `tanh` to the interval [−1,1]. The audit shows that applying the bounded transform after gap construction saturates the tails: the lower quantile was near −1 for all forward cases and the upper quantile was near +1 for 78.0%. This is an observed pathology; its causal effect on aggregate performance was not isolated.

### 4.4 Objective

For quantile level `τ`, outcome `y`, and forecast `q`, pinball loss is

{EQUATION:ρ_τ(y − q) = (y − q)[τ − 1{y < q}].}

The primary loss is a horizon- and target-weighted sum of normalized pinball terms. Point-of-control targets receive unit weight; log volume and delta receive weight 0.5. The training objective adds a median Huber term with coefficient 0.25, a buy/sell presence binary-cross-entropy term with coefficient 0.20, and an auxiliary decoder loss scheduled from 0.25 toward 0.05 after 60% of planned optimizer steps.

{EQUATION:L = L_pinball + 0.25 L_Huber(q_0.50) + 0.20 L_presence + λ_aux(s)L_aux.}

Because early stopping occurred after only 36% of the planned schedule, auxiliary decay never began. Every recorded pre-clipping gradient norm exceeded the configured threshold of 1.0. These facts motivate ablations but do not, on their own, prove that the optimizer caused the economic failure.

## 5. Training and Reproducibility

The single run used seed 17, AdamW with learning rate `3×10−4`, minimum learning rate `3×10−5`, betas 0.9 and 0.95, epsilon `10−8`, weight decay 0.05, 5% warm-up, and global gradient clipping at 1.0. A microbatch of 32 with four-step accumulation gave an effective batch of 128. Training used deterministic FP32 execution on one CUDA device. The run was configured for 50 epochs with patience eight, stopped at epoch 18 after 3,618 optimizer steps, and selected epoch 10 at step 2,010.

| Item | Value |
|---|---|
| Training sessions / examples | 18 / 25,758 |
| Selection sessions / examples | 2 / 2,862 |
| Development-forward sessions / examples | 4 / 5,724 |
| Retired outer sessions / examples | 6 / 8,586 |
| Seed / folds | 1 / 1 |
| Selected checkpoint | epoch 10, step 2,010 |
| Final training step | 3,618 |
| Selected validation loss | 0.210765 |
| Empirical validation baseline | 0.222446 |
| Weight artifact | 8.071 MiB |
| Resumable checkpoint | 24.284 MiB |

The selected checkpoint improved the validation metric by 5.25% relative to the empirical baseline. This is selection evidence, not an independent performance estimate. No hyperparameter search beyond the documented configuration is represented in the reported statistics.

## 6. Results

### 6.1 Aggregate probabilistic score

On the development-forward partition, weighted normalized pinball loss was 0.251391 for CHMT-Q and 0.257193 for the empirical marginal baseline. The relative improvement was 2.2558%, answering RQ1 positively within the bounded development design.

The aggregate conceals substantial target heterogeneity. All three point-of-control targets improved in pinball loss across all five horizons, although the magnitude declined with horizon. Log-volume residual improved only at horizon one and deteriorated increasingly thereafter. Delta-fraction pinball loss was worse than baseline at every horizon. At horizon five, the per-target improvements were +2.62%, +4.84%, +3.53%, −22.43%, and −29.57%, respectively.

{FIGURE:score_heatmap.png|Figure 4. Development-forward pinball-loss improvement relative to the empirical marginal baseline. Positive values favor CHMT-Q; aggregate weighting masks poor log-volume and delta cells.|Five-by-five heatmap of relative pinball improvement by forecast horizon and target.}

| Development-forward endpoint | CHMT-Q | Baseline | Relative change |
|---|---:|---:|---:|
| Weighted normalized pinball | 0.251391 | 0.257193 | +2.2558% improvement |
| H5 total-POC median MAE (ticks) | 532.841 | 532.905 | +0.0119% improvement |
| H5 log-volume median MAE | — | — | 24.44% worse |
| H5 delta median MAE | — | — | 44.18% worse |

The proper-score improvement is real as an arithmetic comparison on this partition. It is not evidence that the Transformer is necessary, because the registered benchmark set omitted strong simple conditional models. It is also not evidence of calibration under distribution shift or statistical significance across regimes.

### 6.2 Five-minute conditional signal

The horizon-five total point-of-control median illustrates the gap between distributional and decision-relevant quality. Its median MAE was effectively tied with the marginal baseline. Pearson correlation with the realized value was 0.0774 and Spearman correlation was 0.0753. Direct sign classification produced Matthews correlation 0.0481. The separate anchor-to-future-close direction proxy had 52.67% accuracy, 52.36% balanced accuracy, Matthews correlation 0.0475, and Spearman association 0.0780.

Predicted medians were markedly under-dispersed: their standard deviation was only 15.13% of the realized standard deviation. At the same time, the average 10%–90% interval spanned 1,904.7 ticks, or approximately USDT 190.47 at the dataset tick size, and empirical interval coverage was 81.15%. Every interval crossed zero. The coexistence of a shrunk median and a broad interval is consistent with learning marginal uncertainty more successfully than conditional location.

{FIGURE:h5_diagnostics.png|Figure 5. Horizon-five signal diagnostics. Median association and dispersion are weak even though nominal interval coverage is close to 80%.|Grouped diagnostic bars for correlation, sign MCC, dispersion ratio, interval coverage, and fraction of intervals crossing zero.}

These findings answer RQ2 negatively for the intended decision. The model contains a weak association, but it does not separate positive from negative opportunity with useful uncertainty selectivity.

### 6.3 Economic feasibility

The registered interval policy made zero trades because all intervals crossed zero. This is not a software failure: an uncertainty-aware rule abstained because the forecast did not justify a directional position.

The post hoc median-sign long-flat rule made 484 trades. Its frictionless compounded return was +2.1607%, but its return under the base 12-basis-point one-way cost was −68.0682%. Average measured gross edge was 0.4453 basis points per round trip, implying a break-even cost of 0.2227 basis points per side. The base cost was 53.89 times that break-even level. A median-sign long-short rule made 1,148 trades, returned +4.989% without costs, and −93.344% at base costs; its gross round-trip edge was 0.4301 basis points.

{FIGURE:cost_feasibility.png|Figure 6. Cost feasibility of the diagnostic median-sign policies. The observed gross edge is far below the base round-trip cost of 24 basis points.|Chart comparing gross round-trip edge, break-even one-way cost, and tested one-way cost assumptions for long-flat and long-short policies.}

| Policy | Trades | Frictionless compounded return | Base-cost compounded return | Gross edge / round trip |
|---|---:|---:|---:|---:|
| Registered interval long-flat | 0 | 0.00% | 0.00% | Not estimable |
| Diagnostic median long-flat | 484 | +2.1607% | −68.0682% | 0.4453 bps |
| Diagnostic median long-short | 1,148 | +4.989% | −93.344% | 0.4301 bps |

The simulator uses minute trade prices and fixed basis-point deductions. It does not measure bid–ask spread at the executable quote, queue position, partial fills, market impact, financing, outages, or capacity. Therefore, it cannot validate live profitability. Since even this simplified simulation is dominated by costs, adding realistic execution frictions would not rescue the tested policies. RQ3 is negative under the stated assumptions.

## 7. Failure-Factor Audit

{FIGURE:failure_map.png|Figure 7. Ranked failure factors. Solid labels denote observed decisive or major factors; hatched labels denote plausible contributors whose causal effect requires ablation.|Ranked horizontal bars for twelve audited failure factors with evidence-strength categories.}

### 7.1 Decisive factors

**Task–decision mismatch.** The model optimized five footprint primitives, not executable returns, price changes, or actions. Only 3.46% of the full horizon-weighted objective was attributable to horizon-five total-POC quantiles. A lower total loss can therefore arise while the economically relevant proxy remains unchanged. This is the clearest explanation for why the proper score did not transfer.

**Cost dominance.** The diagnostic long-flat edge of 0.4453 basis points per round trip is structurally incompatible with a 24-basis-point base round trip. Threshold search cannot alter this arithmetic without also changing trade selection and requiring fresh validation. The 53.89× cost-to-break-even ratio is more informative than the frictionless return.

**Uncertainty without selectivity.** Every horizon-five total-POC interval crossed zero. The primary rule correctly abstained. Forcing a trade from the median discards the uncertainty information that the model was explicitly trained to provide.

### 7.2 Major data and validation limitations

**Dependent examples.** Adjacent examples share 98.44% of the 64-minute bar context and 80% of the horizon-five labels. For the realized H5 target, an autocorrelation-based heuristic estimated an effective fraction of approximately 22.3%, and predicted medians were even more persistent. The 25,758 training rows are not independent observations; they come from 18 market sessions. With 2.09 million parameters, the model has approximately 81 parameters per nominal training row and 116,086 per training session.

**Narrow selection and evaluation.** The study uses one seed, one fold, two selection days, and four development-forward days. It cannot estimate across-seed instability, regime robustness, venue transfer, or tail behavior. The retired outer period removes the only planned confirmatory window.

**Weak benchmark set.** The empirical marginal baseline tests whether the model learns conditional structure beyond a stationary distribution. It does not test whether attention, hierarchy, or 2.09 million parameters are warranted. Ridge, logistic, boosted-tree, temporal-convolutional, DeepLOB-like, and persistence/seasonal variants are required before an architecture claim.

### 7.3 Target and output pathologies

**Redundant point-of-control heads.** The three POC targets were correlated at roughly 0.93–0.97, and one side nearly always coincided with total POC. Together they consumed 75% of the target weighting, allowing aggregate improvement to be concentrated in redundant outputs.

**Degenerate presence labels.** Buy and sell presence were positive for all relevant forward examples. The auxiliary classifier therefore learned an almost constant outcome and provided no decision selectivity.

**Delta saturation.** The delta target’s bounded transform pinned the lower quantile near −1 in 100% of cases and the upper quantile near +1 in 78.0%. Its median predictions were also badly biased: the realized median was −0.0345, while the predicted-median distribution had median 0.6695 and bias 0.6881. Pearson association was −0.0107.

### 7.4 Optimization and feature contributors requiring ablation

All 3,618 recorded pre-clipping gradient norms exceeded 1.0; their median was 19.58, the 90th percentile 38.30, and the maximum 109.07. Because early stopping occurred before auxiliary decay, the auxiliary weight remained at its initial value. These facts show that training operated in a persistently clipped regime, but only controlled retraining can establish effect size.

Robust scaling also collapsed sparse level-1 changes. Several zero-median-absolute-deviation features received the `10−6` minimum scale, so any nonzero change mapped to the ±8 clipping boundary. Approximately 22.84% of selected mid/bid/ask changes were clipped, retaining mostly sign rather than magnitude. Again, the representation loss is observed; its metric impact remains unisolated.

| Rank | Factor | Evidence status | Implication |
|---:|---|---|---|
| 1 | Target is not trading return | Decisive, observed | Forecast loss is not a PnL objective |
| 2 | Gross edge ≪ costs | Decisive, observed | Tested policies are economically infeasible |
| 3 | Every interval crosses zero | Major, observed | Primary policy abstains |
| 4 | Weak, shrunk H5 medians | Major, observed | Aggregate score masks weak location signal |
| 5 | Heavy window overlap | Design limitation | Nominal sample size is misleading |
| 6 | One seed/fold; four forward days | Validation limitation | Robustness is unknown |
| 7 | Degenerate presence labels | Wasted objective capacity | Auxiliary classifier adds no selectivity |
| 8 | Saturated delta tails | Output pathology | Bounded uncertainty head is uninformative |
| 9 | Persistent gradient clipping | Plausible contributor | Requires optimizer ablation |
| 10 | Sparse-feature clipping | Representation loss | Requires scaler ablation |
| 11 | Weak baselines | Benchmark limitation | Transformer necessity is untested |
| 12 | Simplified execution | Fidelity limitation | Live performance is unestablished |

RQ4 is answered by a hierarchy, not a single root cause. Task mismatch and cost dominance are decisive for the tested trading interpretation. Other factors constrain generalization or plausibly degrade learning but require prospective ablation before causal attribution.

## 8. Threats to Validity

### 8.1 Internal validity

The checkpoint and empirical baseline were frozen before the reported development-forward evaluation, but the diagnostic policies and several post hoc baseline analyses were developed after inspecting results. They are explicitly labeled diagnostic. The single run prevents separation of architecture effects from random initialization. The external dataset compiler and target scaler were hash-identified, but the absence of exchange sequence/checksum data limits reconstruction assurance.

### 8.2 Statistical conclusion validity

Rows are highly dependent, labels overlap, and only four calendar days contribute to the forward estimate. Row-wise confidence intervals would be anti-conservative. Daily-block resampling has too few blocks to support a stable tail probability. We therefore do not report a significance claim, Sharpe ratio inference, or probability of profitability. Relative improvements are descriptive arithmetic on the frozen partition.

### 8.3 Construct validity

Footprint POC and volume targets are not executable returns. Minute-open and minute-close trade prices are not guaranteed fills. Fixed basis-point costs do not reproduce a changing spread, latency, queue priority, impact, or partial execution. The separate candle-direction proxy is an approximation derived after the fact, not a target on which the model was trained.

### 8.4 External validity

The evidence covers BTC-USDT spot on OKX during one month. It does not establish transfer across assets, derivative contracts, venues, volatility regimes, fee tiers, market structures, or live infrastructure. The historical archives omit receive-time information, so causal masks establish temporal ordering only relative to archive event order and feature construction.

### 8.5 Researcher degrees of freedom

The study contains one registered forecast endpoint and one registered interval policy, but the audit necessarily examines many cells and diagnostics. These analyses generate hypotheses; they do not upgrade the original result. Future work must freeze targets, baselines, thresholds, costs, folds, seeds, and stopping rules before accessing a new confirmatory period.

## 9. Corrective Research Protocol

The next study should be treated as a new experiment, not a continuation that reuses the retired holdout. Figure 8 summarizes the gates.

{FIGURE:next_study.png|Figure 8. Registered next-study protocol. Forecast, execution, and reinforcement-learning claims require separate gates and a newly sealed confirmatory period.|Stage-gate diagram from data acquisition through baselines, target ablations, walk-forward testing, DQN eligibility, sealed holdout, and shadow deployment.}

### 9.1 Data and validation gates

- Acquire at least 240 UTC sessions with exchange event time and local receive time where feasible, sequence/checksum integrity, historical instrument metadata, and a documented license review.
- Use at least five anchored rolling folds. Each fold must contain at least 120 training sessions, 20 selection sessions, and 20 disjoint test sessions, with purge and embargo equal to the maximum feature/label reach.
- Evaluate at least five Transformer seeds. Finalist decision models require at least ten seeds.
- Reserve a new access-controlled holdout of at least 60 sessions. Lock code, data hashes, thresholds, costs, and primary endpoints before target-bearing access.
- Require at least 60 subsequent shadow-trading days before any capital-allocation decision.

### 9.2 Target and model ablations

The forecast study should add direct, executable return targets and retain footprint targets only if their incremental value survives ablation. Total/buy/sell POC redundancy should be reduced through target selection or orthogonalization. Presence labels should be removed unless both classes occur with adequate frequency. Delta quantiles should use a distribution compatible with bounded support or transform the target before ordered-gap construction.

Every complex model must be compared with frozen simple baselines: zero/persistence, seasonal or exponentially weighted averages, ridge/logistic models, gradient-boosted trees, temporal convolution, and a DeepLOB-like encoder. Ablations should test modality removal, hierarchical versus flattened L2, quality gating, loss weights, scaler behavior, gradient clipping, auxiliary scheduling, and direct-return versus footprint supervision.

### 9.3 Economic evaluation

The primary endpoint should be net return after a preregistered cost model, with turnover, drawdown, exposure, and capacity reported jointly. Cost assumptions should be derived from executable bid/ask quotes and include fee tier, latency, queue/partial-fill rules, impact, and financing where relevant. Sensitivity grids are supplementary; the primary cost and position sizing rule must be frozen.

The key feasibility gate is not a positive frictionless return. The lower uncertainty bound of gross edge must exceed the all-in cost estimate by a preregistered safety margin across folds and seeds. No threshold selected on test or holdout data may be carried into a performance claim.

## 10. Proposed Transformer–Double-DQN Study

The reinforcement-learning design is a preregistered future-work architecture. It has not been implemented, trained, backtested, or integrated into the terminal. No performance number in this paper comes from a DQN.

### 10.1 State, action, and transition

The state would combine frozen or jointly trained CHMT-Q embeddings with current target position, realized volatility, spread/liquidity proxies, time-of-day, and risk-budget variables. Actions would specify the next target position `a ∈ {−1,0,+1}` rather than ambiguous buy/sell/hold events. This representation makes turnover and costs functions of position change.

Transitions would come from a fixed-small-notional, exogenous counterfactual simulator. The environment must reject transitions whose required book state or executable quotes are unavailable. Since historical actions do not affect future observations under this assumption, conclusions would apply only to strategies whose size is too small to move the market.

### 10.2 Learning objective

The initial learner would use Double DQN to reduce max-operator bias [23] and a dueling decomposition to separate state value from action advantage [24]. A slowly updated target network and replay buffer would stabilize training [22]. The training reward may include bounded risk shaping, but the evaluation endpoint must remain raw net PnL under the frozen simulator:

{EQUATION:r_t = p_t · Δm_{t+1} − C(|p_t − p_{t−1}|, spread_t, liquidity_t) − λ_risk R_t.}

Shaped reward is a training device, not the reported business metric. Quantile-distributional DQN [25] and conservative Q-learning [26] are planned ablations, particularly because offline action support is limited. Behavior-cloning and no-change policies are required controls. Negative controls should randomize rewards, shift features, and permute action labels; any apparent edge that survives these tests indicates leakage or simulator error.

### 10.3 Eligibility and stopping gates

The DQN study may begin only after the supervised state representation beats strong simple baselines on direct executable-return targets and the simulator passes accounting, no-lookahead, and cost tests. Training must stop if action support is inadequate, Q-values diverge, negative controls profit, or seed dispersion exceeds the registered tolerance. Model selection uses only training and selection folds; test folds and the new holdout remain single-access evaluations.

| Gate | Minimum condition | Failure consequence |
|---|---|---|
| Data integrity | Sequence/receive-time audit and no-lookahead tests pass | Stop |
| Forecast utility | Direct-return model beats frozen simple baselines across folds | Do not train DQN |
| Cost feasibility | Gross edge clears all-in costs with safety margin | Do not train DQN |
| Offline support | Every candidate action has adequate state support | Restrict action set or stop |
| Robustness | ≥5 encoder seeds and ≥10 finalist DQN seeds | No promotion |
| Confirmatory evidence | New ≥60-session sealed holdout | No performance claim |
| Operations | ≥60 live shadow days with drift/latency monitoring | No capital allocation |

A DQN can optimize the timing and persistence of positions under a specified reward. It cannot repair missing causal information, make transaction costs disappear, or validate itself on reused data. Profitability remains an empirical question behind every gate.

## 11. Reproducibility, Data Availability, and Ethics

### 11.1 Artifact ledger

| Artifact | SHA-256 or identifier | Role |
|---|---|---|
| Compiled dataset | `565c1d8e912ee426a1a92572d5be481ec5168d54d1ae53bd77fdd70d888e4df6` | Dataset identity |
| Dataset manifest | `68ec9eca21d2a30546435738e8e07d565a64edbf01825246db659a4c1bb98046` | Source/shard manifest |
| Compiler report | `cc3f9208e2410b41ca9836ef762c15cfd91d3b9c704c2d198c438425f4c9e8ba` | Compilation audit |
| Dependency lock | `48792cf2789a576553fc29b1b727e97342d28fed1fdbeb91c31135a84bfff898` | Environment identity |
| Source split | `7f5254f62a3d70a6f02e41b1991f25d3bd8338e7aa9b25c5fa19737f70a7bb8e` | Partition identity |
| Input/target scaler | `29a733fac9986eb7e879555b6813f8d35b06c98dee9041057e6fc4287bd50880` | Scaling state |
| Training run | `25be2f446a75ad37120bba956ea86e46c0ecefa004ab3343d6745cdbee102fb3` | Experiment identity |
| Study protocol | `885fcd0fee23115a4a5480ec94d95f436e251ec0caf796f9686867ed7e72f5d0` | Frozen evaluation protocol |
| Prediction tape | `27d9ffbe41b18868b8a45bf9c48c936908de7394dd6ecfec7062cc531cc7e1f0` | Model outputs |
| Study results | `083da170146585cf1a16dde56e2da02697194640ed7b053d3298d8e3775bee6a` | Forecast/economic results |
| Failure diagnostics | `ece5b18f8a8455f7edac8491b6d23362fa14f618129b1193b1febf14bc89505e` | Audit results |
| DQN revision plan | `d5e81f203619c55754c940df7e92ec5995e4db41b3a7d6f3db4137318f7a502c` | Future-work specification |

The implementation and local artifacts are private research materials. Exchange archives and derived data are not redistributed by this manuscript. Reproduction therefore requires independent acquisition under applicable OKX terms and regeneration using the disclosed tensor, split, training, and evaluation contracts. The hash ledger supports local identity checking; it does not substitute for public data availability.

### 11.2 Computational and market-risk ethics

The reported training used one GPU run; no claim is made that the architecture is compute-optimal. More importantly, the model is unsuitable for capital allocation. Publishing a frictionless return without the cost result would create a material risk of misuse, so both are reported together. Any later live test should use shadow mode, hard position and loss limits, kill switches, human oversight, and monitoring for data drift, latency, execution quality, and anomalous turnover.

### 11.3 Declarations

**Funding:** [To be completed by the authors.]  
**Competing interests:** [To be completed by the authors.]  
**Author contributions:** [To be completed using the target venue taxonomy.]  
**Use of generative tools:** [To be completed according to venue policy; all quantitative claims require author verification.]  
**Data and code availability:** Private local research artifacts; no redistribution claim. Reproduction contract and hashes are provided above.  
**Human or animal participants:** Not applicable.

## 12. Conclusion

CHMT-Q improved a registered aggregate probabilistic score by 2.26% on a four-session development-forward partition. That result did not imply a useful trading signal. Horizon-five medians were weak and under-dispersed, every uncertainty interval crossed zero, the primary policy made no trades, and diagnostic policies lost heavily once modest fixed costs were applied. The observed long-flat gross edge was 53.9 times too small for the base cost assumption.

The strongest contribution is therefore methodological. Proper scores, conditional signal, and executable economics are separate evidentiary levels. A model should not advance because it wins at the first while failing or remaining untested at the others. The current checkpoint remains research-only and non-promotable. A larger, access-controlled, multi-fold, multi-seed study with direct return targets, strong baselines, realistic execution, and a new sealed holdout is required before reconsideration. The proposed Transformer–Double-DQN design is one testable option within that protocol, not a remedy already demonstrated.

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems*, 30. https://arxiv.org/abs/1706.03762
2. Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. *NAACL-HLT*, 464–468. https://doi.org/10.18653/v1/N18-2074
3. Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1910.07467
4. Shazeer, N. (2020). GLU Variants Improve Transformer. Technical report. https://arxiv.org/abs/2002.05202
5. Nie, Y., et al. (2023). A Time Series Is Worth 64 Words: Long-Term Forecasting with Transformers. *ICLR*. https://arxiv.org/abs/2211.14730
6. Liu, Y., et al. (2024). iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. *ICLR*. https://arxiv.org/abs/2310.06625
7. Lim, B., Arik, S. O., Loeff, N., & Pfister, T. (2021). Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting. *International Journal of Forecasting*, 37(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012
8. Zeng, A., et al. (2023). Are Transformers Effective for Time Series Forecasting? *AAAI*, 37(9), 11121–11128. https://doi.org/10.1609/aaai.v37i9.26317
9. Zhang, Z., Zohren, S., & Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. *IEEE Transactions on Signal Processing*, 67(11), 3001–3012. https://doi.org/10.1109/TSP.2019.2907260
10. Wallbridge, J. (2020). Transformers for Limit Order Books. Technical report. https://arxiv.org/abs/2003.00130
11. Zhang, Z., & Zohren, S. (2021). Multi-Horizon Forecasting for Limit Order Books: Novel Deep Learning Approaches and Hardware Acceleration Using Intelligent Processing Units. Technical report. https://arxiv.org/abs/2105.10430
12. Ntakaris, A., et al. (2018). Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods. *Journal of Forecasting*, 37(8), 852–866. https://doi.org/10.1002/for.2543
13. Kolm, P. N., Turiel, J., & Westray, N. (2023). Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the Limit Order Book. *Mathematical Finance*, 33(4), 1044–1081. https://doi.org/10.1111/mafi.12413
14. Lucchese, L., Pakkanen, M. S., & Veraart, A. E. D. (2024). The Short-Term Predictability of Returns in Order Book Markets: A Deep Learning Perspective. *International Journal of Forecasting*, 40(4), 1607–1626. https://doi.org/10.1016/j.ijforecast.2024.02.001
15. Briola, A., Bartolucci, S., & Aste, T. (2025). Deep Limit Order Book Forecasting: A Microstructural Guide. *Quantitative Finance*. https://doi.org/10.1080/14697688.2025.2522911
16. Gneiting, T., & Raftery, A. E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *Journal of the American Statistical Association*, 102(477), 359–378. https://doi.org/10.1198/016214506000001437
17. Romano, Y., Patterson, E., & Candès, E. J. (2019). Conformalized Quantile Regression. *Advances in Neural Information Processing Systems*, 32. https://arxiv.org/abs/1905.03222
18. Sirignano, J., & Cont, R. (2019). Universal Features of Price Formation in Financial Markets: Perspectives from Deep Learning. *Quantitative Finance*, 19(9), 1449–1459. https://doi.org/10.1080/14697688.2019.1622295
19. White, H. (2000). A Reality Check for Data Snooping. *Econometrica*, 68(5), 1097–1126. https://doi.org/10.1111/1468-0262.00152
20. Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313. https://doi.org/10.1080/01621459.1994.10476870
21. Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the Cross-Section of Expected Returns. *Review of Financial Studies*, 29(1), 5–68. https://doi.org/10.1093/rfs/hhv059
22. Mnih, V., et al. (2015). Human-Level Control through Deep Reinforcement Learning. *Nature*, 518, 529–533. https://doi.org/10.1038/nature14236
23. van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. *AAAI*, 30(1), 2094–2100. https://doi.org/10.1609/aaai.v30i1.10295
24. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). Dueling Network Architectures for Deep Reinforcement Learning. *Proceedings of Machine Learning Research*, 48, 1995–2003. https://proceedings.mlr.press/v48/wangf16.html
25. Dabney, W., Rowland, M., Bellemare, M. G., & Munos, R. (2018). Distributional Reinforcement Learning with Quantile Regression. *AAAI*, 32(1). https://doi.org/10.1609/aaai.v32i1.11791
26. Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). Conservative Q-Learning for Offline Reinforcement Learning. *Advances in Neural Information Processing Systems*, 33, 1179–1191. https://arxiv.org/abs/2006.04779
27. Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. Technical report. https://arxiv.org/abs/2005.01643
28. Henderson, P., et al. (2018). Deep Reinforcement Learning That Matters. *AAAI*, 32(1). https://doi.org/10.1609/aaai.v32i1.11694
29. Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*, 20(4), 39–69. https://doi.org/10.21314/JCF.2016.322
30. Racine, J. (2000). Consistent Cross-Validatory Model-Selection for Dependent Data: hv-Block Cross-Validation. *Journal of Econometrics*, 99(1), 39–61. https://doi.org/10.1016/S0304-4076(00)00030-0

## Appendix A. Tensor and Parameter Ledger

| Subsystem | Function | Output width or tokens |
|---|---|---|
| Bar projection and patcher | 64 × 16 history to causal memory | 16 patched tokens, width 128 |
| L1 projection and patcher | 240 × 16 microstructure history | 60 patched tokens, width 128 |
| L2 level hierarchy | side/rank encoding per snapshot | width 32 internal |
| L2 temporal encoder | 240 hierarchical snapshots | 60 patched tokens, width 128 |
| Cross-modality memory | encoded memories plus learned summaries | 184 tokens total |
| Quality conditioner | gates modality reliability | width 128 context |
| Horizon decoder | five learned queries, two blocks | 5 × 128 |
| Output head | 5 targets × 3 quantiles + 2 presence logits | 5 × 17 |

The exact implemented trainable parameter count is 2,089,554. This count includes 36 learned query/key temperature parameters omitted from an earlier draft ledger. The model artifact size is not a parameter count: the selected weight file is 8.071 MiB because it stores approximately 2.09 million FP32 values plus serialization metadata; the 24.284 MiB resumable checkpoint additionally stores optimizer and training state.

## Appendix B. Registered and Post Hoc Analyses

| Analysis | Status | Interpretation |
|---|---|---|
| Weighted normalized pinball vs empirical marginal | Registered primary forecast endpoint | Development evidence |
| Interval long-flat policy at base costs | Registered primary economic endpoint | Zero trades; development evidence |
| Median-sign long-flat and long-short policies | Diagnostic post hoc | Cost-feasibility diagnosis only |
| Anchor-close candle direction proxy | Diagnostic post hoc | Approximate downstream association |
| Per-target/per-horizon heatmaps | Diagnostic decomposition | Explains aggregate heterogeneity |
| Recent-three-day and hour baselines | Post hoc, not in main comparison | Hypothesis generation only |
| DQN architecture and validation plan | Proposed future work | No performance evidence |

## Appendix C. Submission Checklist

- Replace anonymous author, affiliation, correspondence, ORCID, funding, competing-interest, and contribution placeholders as required by the selected venue.
- Reformat references and headings to the venue style without changing quantitative claims.
- Confirm whether the venue permits private code/data with a regeneration contract; otherwise select an archive or venue with an appropriate artifact policy.
- Preserve the holdout-integrity disclosure. Do not relabel the retired June 25–30 period as sealed or untouched.
- Recompute and independently verify every table and figure from the immutable artifacts before submission.
- Declare generative-tool assistance according to venue policy and retain human responsibility for all text, equations, and results.
