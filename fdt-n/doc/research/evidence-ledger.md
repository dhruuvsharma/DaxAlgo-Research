# FDT-N evidence and claim ledger

**Status:** active source ledger; completed measurements are artifact-bound in the final PDF, while
unmeasured hypotheses remain explicitly open  
**Companion paper:** [research-paper.md](research-paper.md)  
**Purpose:** keep local implementation facts, literature transfers, hypotheses, and empirical claims
separate and auditable.

## 1. Evidence classes

| Code | Class | Permitted wording |
|---|---|---|
| O | Observed contract/data fact | “The shared schema contains…” |
| I | Implemented mechanism | “The current code performs…” |
| H | Design hypothesis | “We hypothesize…” or “This motivates testing…” |
| R | Empirical result | Only a frozen measurement with dataset/split/artifact identity. |
| X | Rejected/deferred transfer | “Not used in v1 because…” |

An implementation test can support an I-claim. It cannot promote an H-claim into an R-claim.

## 2. Local implementation evidence

| ID | Claim | Local evidence | Status / boundary |
|---|---|---|---|
| O01 | A shared row retains price, buy/sell volume, bid/ask imbalance, and zero flags; total and delta are derived. | [`FootprintFeatures.cs`](../../../public/src/windows/Core/TradingTerminal.Core/MarketData/FootprintFeatures.cs), `FootprintFeatureRow` | Observed shared contract. |
| O02 | A shared completed bar retains every row plus POC, three centroids, side/total volume, delta, CVD, stacked runs, quality, and UTC bounds. | Same file, `FootprintBar` | Side POCs, range, and value area are derivable from rows rather than stored fields. |
| O03 | Real tape, synthetic L1, and no-feed quality states are distinct. | Same file, `FeedQuality` and `Multiplier` | Synthetic L1 is not equivalent to real tape. |
| O04 | Forecast requests bind instrument, source, timeframe, and row size and defensively snapshot complete rows. | [`FootprintForecastProvider.cs`](../../../public/src/windows/Core/TradingTerminal.Core/Ml/FootprintForecastProvider.cs), `FootprintForecastCoordinate`, `FootprintForecastRequest` | Shared seam is model-agnostic. |
| O05 | Shared requests enforce UTC, exact bar duration, exact bar-to-bar contiguity, bounded history/horizon, and cutoff equal to the final history end. | Same file, request validation | Gapped or overlapping history is rejected by the generic seam. |
| O06 | Shared available results require finite ordered quantiles for total POC, low, high, buy POC, sell POC, volume, and delta fraction, plus physical range/volume/delta invariants. | Same file, `FootprintForecastQuantiles`, `FootprintHorizonForecast` | Side-POC quantiles are required. These checks do not prove calibration or a joint distribution. |
| O07 | A null provider returns an unavailable result rather than fabricating a forecast. | Same file, `NullFootprintForecastProvider` | Supports fail-closed integration. |
| I01 | Default encoding uses 64 lookback bars, 8 horizons, and a 1,024-row safety bound. | [`config.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/config.py), `EncodingConfig` | Bars above the bound are rejected. |
| I02 | Row, bar, and target feature orders are explicit and serialized. The target order is `poc_offset`, `low_offset`, `high_offset`, `buy_poc_offset`, `sell_poc_offset`, `log_volume_relative`, `delta_fraction`. | Same file plus [`data.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/data.py) manifest writer | Feature order is part of model compatibility. |
| I03 | Accepted bars retain every row in flat ragged storage with offsets. | `data.py`, `EncodedFootprintSeries.from_bars` | Active path does not compact rows. |
| I04 | Windows and batches pad only to observed local width with masks; the row-token sampler caps both item count and padded width-items, so wide-candle windows form smaller batches without dropping rows. | `data.py`, `make_window`; [`training.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/training.py), `collate_windows`, `RowTokenBatchSampler` | Final defaults cap 128 items and 8,192 padded width-items at reference width 64. Padding is not an observed row. |
| I05 | Continuous context-plus-label intervals are required. | `data.py`, `valid_window_indices` and `make_window` | Missing/irregular windows are excluded. |
| I06 | Context-only POC and recent median log-volume references normalize features and labels. | `data.py`, `make_window`; [`report.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/report.py), `_compute_clipping_audit` | Final PDF reports raw bar/target pre-clip fractions and encoded row-bound saturation; exact row pre-clip loss is explicitly unimplemented because the frozen NPZ retains only the clipped row coordinate. |
| I07 | A per-row MLP feeds four learned cross-attention queries. | [`model.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/model.py), `row_projection`, `auction_queries`, `row_pool` | Query semantic labels are hypotheses. |
| I08 | Row and bar projections are combined by a sigmoid feature gate plus residual projection. | `model.py`, `fusion_gate`, `fusion_projection` | Effectiveness needs ablation. |
| I09 | Three default causal temporal blocks use pre-RMSNorm, multi-head attention, SwiGLU, and LayerScale. | `config.py`; `model.py`, `CausalTemporalBlock` | Mechanical architecture fact. |
| I10 | Eight learned horizon queries decode in parallel through self-attention and history cross-attention. | `model.py`, `horizon_queries`, `HorizonDecoderBlock` | No target or earlier forecast is supplied. |
| I11 | The output shape is `[batch,horizon,7,3]`; five price targets are direct last-context-POC-relative marginals; q10/q50/q90 are ordered; quantilewise low is `min(raw_low, POC)`, high is `max(raw_high, POC)`, and side POCs are clamped into that range; delta fraction is bounded. | `model.py`, `_ordered_quantiles`; [`test_model.py`](../../../tools/volume-footprint-transformer/tests/test_model.py) | Order and physical constraints are not calibration or a joint distribution. |
| I12 | Training uses weighted masked pinball loss with (1/\sqrt h) horizon weighting. | [`loss.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/loss.py) | Final report must also show unweighted cell scores. |
| I13 | Default training uses AdamW, warm-up/cosine schedule, deterministic algorithms, clipping, early stopping, and a purged 70/15/15 chronology. | `training.py` | Current protocol is preliminary, not final qualification. |
| I14 | Current evaluator reports per-horizon/per-target q10/q50/q90 pinball, q50-forecast error, coverage, width, and POC direction; it compares model q50 forecasts with persistence and a direct ridge fitted only on training windows, using paired four-hour contiguous-block resampling. | `training.py`, `distribution_metrics`, `fit_direct_ridge_baseline`, `paired_block_bootstrap_mae_difference`, `_evaluation_block_ids` | The legacy JSON name `median_mae_normalized` means MAE of the q50 forecast, not median absolute error. Adjacent boundary windows still overlap; outputs are dependence-aware descriptive sensitivity intervals, not independent-block confidence intervals or formal generalization bounds. Point baselines do not provide quantile calibration. |
| I15 | ONNX export has dynamic batch/row axes and emits config, feature order, hashes, a reference fixture, PyTorch/ORT parity, representative sample width, and 50-run CPU latency. | [`export.py`](../../../tools/volume-footprint-transformer/daxalgo_footprint_transformer/export.py) | Final PDF accepts these facts only when selected metadata binds the validation-selected result and checkpoint hashes. |
| I16 | Export metadata marks the model research-only and asserts it creates no order signals. | `export.py` | Product safety boundary, not predictive evidence. |

### 2.1 Exact-row wording gate

Approved wording:

> The active encoded-series path retains every observed price row for accepted bars up to an explicit
> 1,024-row safety bound, using ragged storage and mask-only local padding. Bars beyond the bound are
> rejected.

Disallowed wording:

- “unlimited rows”;
- “all rows are always accepted”;
- “overflow compaction is lossless information preservation”;
- “padding rows are market observations.”

No `compact_rows` helper exists in the current research package. If an overflow approximation is
introduced later, the paper must change from exact-row wording to an explicit approximation and
report its implementation, affected frequency, conservation checks, and ablation.

### 2.2 Frozen development corpus

These are observed data facts, not model-result claims.

| ID | Observed fact | Evidence / identity | Boundary |
|---|---|---|---|
| O08 | Fourteen official Binance Data Vision BTCUSDT spot daily individual-`trades` archives cover 2024-01-01 through 2024-01-14 and match the live connector's `@trade` granularity. | [Official archive directory](https://data.binance.vision/?prefix=data/spot/daily/trades/BTCUSDT/); `tmp/volume-footprint-transformer/btcusdt-1m-fourteen-days.manifest.json` records all 14 original byte lengths and SHA-256 values. | Original total is 374,843,570 bytes. Local ZIP copies were truncated after encoding; raw bytes are externally recoverable, not locally retained. |
| O09 | One-minute encoding at 2.5 USDT per footprint row yields 20,160 contiguous bars and 357,676 exact rows; row-count p50/p95/p99/max is 13/47/94.41/605. | Frozen manifest and encoded NPZ | Maximum 605 is below the 1,024 reject bound; no row-bound reject occurred in this corpus. |
| O10 | Quantity conversion matches terminal `SizeScale=1000`: scale, ties-to-even integer rounding, then drop non-positive scaled sizes. | Manifest `encoding.quantity_scale`; `data.py`, `iter_binance_trades` | This preprocessing choice is part of dataset identity. |
| O11 | There are 20,089 valid 64+8 windows; purged chronological counts are train 13,990, validation 2,869, test 2,942, with 72-bar purges. | `valid_window_indices`; `chronological_split` applied to frozen NPZ | No separate calibration partition exists. |
| O12 | Manifest SHA-256 is `b88d89565febb97dbe56151fe447e3952fa844b8c433ecb86cb747e11e4ce412`; the 9,963,548-byte NPZ SHA-256 is `d83f78894af08b65123d2bf9f4fbef8ca28e901a552d3af2c61f6e37513bf6cc`. | Direct SHA-256 over the frozen local files | Hashes establish manifest/encoded-dataset byte identity, not locally retained raw bytes, label correctness, or predictive value. |

## 3. Primary-source ledger

The “transfer” column records only a design lesson. The “not established” column prevents citation
laundering from another domain into FDT-N performance.

| ID | Primary source | Mechanism or evidence used | FDT-N decision | Not established by the source |
|---|---|---|---|---|
| S01 | Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), NeurIPS 2017 | Scaled dot-product and multi-head attention | Use standard dense attention | Footprint predictability or market calibration |
| S02 | Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), NeurIPS 2019 | RMS-only normalization | Use RMSNorm in row, attention, and temporal blocks | Superior FDT-N optimization |
| S03 | Shazeer, [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), 2020 | Gated feed-forward variants including SwiGLU | Use SwiGLU after attention | Market-specific benefit |
| S04 | Shaw, Uszkoreit, and Vaswani, [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155), NAACL 2018 | Relative-position principle | Relative-age encoding remains an ablation; current implementation uses learned 64-position embeddings | Best positional scheme for footprint bars |
| S05 | Zeng et al., [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504), [DOI](https://doi.org/10.1609/aaai.v37i9.26317), AAAI 2023 | Simple linear baselines can be strong | Require linear/seasonal baselines | That linear models will win or lose here |
| S06 | Kim et al., [Reversible Instance Normalization](https://openreview.net/forum?id=cGDAkQo1C0p), ICLR 2022 | Past-window normalization under distribution shift | Robust normalization is an ablation; current scheme uses physical tick/share scales | Leakage-safe benefit on footprint data |
| S07 | Lim et al., [Temporal Fusion Transformers](https://arxiv.org/abs/1912.09363), [DOI](https://doi.org/10.1016/j.ijforecast.2021.03.012), IJF 2021 | Gating, known/observed covariate separation, direct quantiles | Transfer gating and information-contract principles | Non-crossing quantiles or FDT-N accuracy |
| S08 | Nie et al., [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730), ICLR 2023 | Temporal patches and channel independence | Treat a sealed bar as a domain patch; require PatchTST baseline; reject strict channel independence | Validity for interacting footprint rows |
| S09 | Liu et al., [iTransformer](https://arxiv.org/abs/2310.06625), ICLR 2024 | Variate tokens and cross-variate attention | Comparator and partial motivation for structured feature interaction | Natural ordering or geometry for ragged price rows |
| S10 | Zhang and Yan, [Crossformer](https://openreview.net/forum?id=vSVLM2j9eie), ICLR 2023 | Dimension-segment-wise embedding and staged cross-time/cross-dimension attention | Adapted comparator | Transfer to irregular row-set inputs |
| S11 | Wang et al., [TimeXer](https://arxiv.org/abs/2402.19072), NeurIPS 2024 | Patch-wise endogenous and variate-wise exogenous interaction | Defer external branch until causal availability is proven | That any external series improves FDT-N |
| S12 | Woo et al., [Unified Training of Universal Time Series Forecasting Transformers](https://arxiv.org/abs/2402.02592), ICML 2024 | Moirai universal probabilistic forecasting across frequencies/variates | Candidate zero-shot/fine-tuned baseline with frozen adapter | Native support for footprint row sets or market edge |
| S13 | Zhang, Zohren, and Roberts, [DeepLOB](https://arxiv.org/abs/1808.03668), [DOI](https://doi.org/10.1109/TSP.2019.2907260), IEEE TSP 2019 | Local LOB structure before temporal modeling | Adopt hierarchy principle; require DeepLOB-like baseline | Executed-footprint quantiles or profitability |
| S14 | Wallbridge, [Transformers for Limit Order Books](https://arxiv.org/abs/2003.00130), 2020 | Causal dilated convolutions with transformer modeling | Causal structural-temporal comparator | Generality beyond its LOB tasks |
| S15 | Berti and Kasneci, [TLOB](https://arxiv.org/abs/2502.15757), 2025 | Separate temporal and feature/spatial attention | Supports dual-axis ablation | FDT-N query semantics or target predictability |
| S16 | Dai et al., [DeepSeekMoE](https://arxiv.org/abs/2401.06066), 2024 | Shared and routed expert specialization | Reject in v1; no capacity/routing bottleneck shown | Benefit for a sub-million-parameter forecaster |
| S17 | DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434), 2024 | Multi-head latent attention and giant MoE | Reject MLA/MoE; no long autoregressive KV cache | Dense 64-token attention inadequacy |
| S18 | DeepSeek-AI, [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), 2024/2025 | Auxiliary-loss-free routing and multi-token prediction | Multi-horizon supervision is a conceptual question only; FDT-N does not implement DeepSeek MTP | Numeric multi-horizon accuracy |
| S19 | Yuan et al., [Native Sparse Attention](https://arxiv.org/abs/2502.11089), 2025 | Hardware-aligned sparse attention for long contexts | Reject at 64 temporal tokens | Benefit when dense attention is already small |
| S20 | Yang et al., [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671), 2024 | Large dense/MoE language transformers using mature gated/norm components | Component precedent only | Financial forecasting or need for GQA/MoE |
| S21 | Qwen Team, [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), 2025 | Dense/MoE models, training stability, reasoning modes | Component/stability precedent only | Value of reasoning modes for supervised numeric forecasts |
| S22 | Sun, Cetin, and Tang, [Learning to Predict Structural Adaptations of Large Language Models](https://arxiv.org/abs/2501.06252), 2025 | Transformer-Squared task-conditioned singular-value scaling | Reject runtime adaptation; possible offline research only | Stable market-regime adaptation or auditability |
| S23 | Akiba et al., [Evolutionary Optimization of Model Merging Recipes](https://arxiv.org/abs/2403.13187), [DOI](https://doi.org/10.1038/s42256-024-00975-8), Nature Machine Intelligence 2025 | Evolutionary task-vector/data-flow merging | Defer until qualified same-base experts exist; never runtime Frankenmerging | Improvement without valid experts and held-out tests |
| S24 | Kimi Team, [Kimi Linear](https://arxiv.org/abs/2510.26692), 2025 | Kimi Delta Attention with periodic full attention for very long contexts | Defer; dense exact attention and explicit reset semantics fit 64 bars | Short-context improvement |
| S25 | Kimi Team, [Kimi K2](https://arxiv.org/abs/2507.20534), 2025/2026 | Large MoE/agentic training and stability methods | Optimizer diagnostics only if independently justified; reject agentic/MoE transfer | Supervised footprint forecasting |
| S26 | Romano, Patterson, and Candès, [Conformalized Quantile Regression](https://arxiv.org/abs/1905.03222), NeurIPS 2019 | Split conformal correction for quantile intervals | Optional chronological calibration experiment | Universal coverage under dependent, nonstationary markets |
| S27 | Ntakaris et al., [Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data](https://arxiv.org/abs/1705.03233), [DOI](https://doi.org/10.1002/for.2543), Journal of Forecasting 2018 | FI-2010 benchmark conventions | Context for LOB evaluation differences | A footprint benchmark or trading profitability |
| S28 | Kolm, Turiel, and Westray, [Deep Order Flow Imbalance](https://doi.org/10.1111/mafi.12413), Mathematical Finance 2023 | Stationary order-flow representations and multiple horizons | Motivates strong flow baselines and rolling evaluation | FDT-N targets, intervals, fills, or P&L |

## 4. Transfer decisions for giant-model mechanisms

| Mechanism | Original problem | FDT-N condition | Decision |
|---|---|---|---|
| Multi-head latent attention | Reduce autoregressive KV-cache width | Encoder-style batch forecast; 64 history tokens; no long generated stream | Reject v1 |
| Mixture of experts | Increase capacity while activating a subset | 371,861 parameters in the final trained configuration; no demonstrated capacity bottleneck or expert corpus | Reject v1 |
| Native sparse attention | Reduce long-context attention work | 4,096 temporal pairs per layer/head | Reject v1 |
| Kimi Delta Attention | Maintain recurrent memory over very long sequences | Fixed 64-bar window; gap/reset correctness is more important than asymptotic length | Defer benchmark |
| Grouped-query attention | Reduce decoder cache and bandwidth | No autoregressive token generation | Reject v1 |
| Multi-token prediction | Auxiliary language-token training | FDT-N directly supervises eight numeric horizons without future-token embeddings | Do not claim implementation transfer |
| Transformer-Squared | Mutate singular-value scales by task at inference | Frozen hash and deterministic export required | Reject runtime use |
| Evolutionary merging | Search combinations of existing experts | No qualified same-base experts exist | Defer offline-only study |
| LLM reasoning/agent mode | Allocate generation compute to language reasoning/tool use | Fixed supervised tensor mapping | Out of scope |

## 5. Prior negative evidence: CHMT-Q boundary

The local
[CHMT-Q June 2026 development report](../../chmt-q/okx-june-2026-development-backtest-report.md)
is an observed prior study, not a directly comparable FDT-N baseline.

| ID | Observed prior evidence | Transfer to FDT-N | Boundary |
|---|---|---|---|
| N01 | CHMT-Q consumed aggregate bar, half-second L1, fixed-depth L2, and quality tokens; it did not consume exact ragged footprint rows. | Different input design is cautionary context only. | Do not compare scores as if modalities matched. |
| N02 | Total-POC q50-forecast MAE was worse than its empirical baseline at horizons 1–3 and effectively tied at 4–5. | Proper-score gains need not imply better medians. | Different targets, five-horizon task, venue, period, and baseline. |
| N03 | Its five-minute anchor-close direction proxy was 52.67% with MCC 0.0475. | Do not reinterpret POC marginals as a return or candle model. | Secondary proxy, not FDT-N directional evidence. |
| N04 | The preregistered uncertainty policy made zero trades; the report concluded no deployment and no capital allocation. | Abstention and forecast utility must be reported separately from economic value. | No FDT-N trading claim follows. |
| N05 | The 2026-06-25 through 2026-06-30 period was unused in the historical study, but a later projection physically read target-bearing fields; it is now process-accessed and retired from confirmatory use. No values were retained or analyzed. | Preserve process-level holdout discipline. | CHMT-Q's restricted data must not be copied or reused for FDT-N. |

## 6. Hypothesis ledger

| ID | Hypothesis | Required decisive comparison | Status |
|---|---|---|---|
| H01 | Full rows add predictive information beyond aggregates. | FDT-N full input versus aggregate-only model under matched splits/budget. | **[PENDING]** |
| H02 | Four learned queries outperform fixed pooling. | Four queries versus one query, mean, max, and DeepLOB-like stem. | **[PENDING]** |
| H03 | The four queries specialize stably. | Seed stability, feature interventions, query swaps, and held-out attention probes. | **[PENDING]** |
| H04 | Gated fusion improves robustness. | Gate versus concatenation/addition/row-only/bar-only. | **[PENDING]** |
| H05 | Causal temporal attention improves proper scores. | Matched causal TCN, GRU/LSTM, and linear controls. | **[PENDING]** |
| H06 | Parallel horizon queries reduce long-horizon degradation. | Independent heads and recurrent/autoregressive diagnostic baseline. | **[PENDING]** |
| H07 | Ordered heads yield useful uncertainty. | Per-cell pinball, interval score, calibration, width, and conditional slices. | Partially measured: quantile calibration, coverage, and width reported; interval score and conditional slices unimplemented. |
| H08 | Feed-quality input supports safe degradation. | Quality removal plus genuine/synthetic/missing-feed slices; abstention analysis. | **[PENDING]** |
| H09 | Exact rows matter relative to a disclosed overflow approximation. | Exact-row model versus a separately implemented, audited approximation; report affected frequency and conservation limits. | **[PENDING]** |
| H10 | FDT-N meets interactive chart latency. | Frozen runtime, batch-one CPU p50/p95/p99, peak memory, and parity. | Partially measured: representative CPU p50/p95/p99 and parity passed; end-to-end chart latency and peak memory unmeasured. |

## 7. Required result ledger

Completed development measurements are populated only by the fail-closed PDF generator from frozen
run/export artifacts. Registered ablations and broader transfer claims remain unmeasured.

| ID | Result claim | Dataset/split identity | Metric and uncertainty | Artifact evidence | Status |
|---|---|---|---|---|---|
| R01 | Final label, clipping, and acceptance audit | Frozen corpus in O08–O12 | raw per-bound bar/target clipping and row-bound saturation | Final PDF input/target clipping pages | Measured except exact row pre-clip and upstream-exclusion audit, which are explicitly unimplemented |
| R02 | Held-out model median versus persistence | frozen byte-identical test indices for every seed | per-horizon/target MAE difference and paired four-hour descriptive sensitivity interval | Final PDF result bundle and hashes | Measured development evidence |
| R03 | Held-out model median versus train-only direct ridge | frozen byte-identical test indices for every seed | per-horizon/target MAE difference and paired four-hour descriptive sensitivity interval | Final PDF result bundle and hashes | Measured development evidence |
| R04 | Held-out proper score versus a task-matched distributional baseline | **[PENDING]** | per-horizon/target/quantile pinball and dependence-aware interval | **[PENDING]** | Comparator pending |
| R05 | Full-row ablation effect | **[PENDING]** | paired score difference and CI | **[PENDING]** | Not measured |
| R06 | Auction-query ablation and specialization | **[PENDING]** | score/stability/intervention evidence | **[PENDING]** | Not measured |
| R07 | q10/q50/q90 calibration | frozen held-out result cells for every declared seed | empirical coverage and width by horizon/target; interval score remains unimplemented | Final PDF per-seed cell pages | Measured development evidence within stated metric scope |
| R08 | Regime/instrument/feed robustness | **[PENDING]** | predeclared slices and counts | **[PENDING]** | Not measured |
| R09 | Runtime latency and memory | selected validation-only checkpoint/export | representative batch-one CPU p50/p95/p99 over 50 runs; peak bytes unimplemented | Selected `metadata.json` bound in final PDF | Latency measured; peak memory not measured |
| R10 | PyTorch-to-ONNX runtime parity | selected validation-only checkpoint/export | max/mean absolute error and declared tolerance | Selected `metadata.json`, checkpoint, ONNX, and result hashes | Passed: max 2.3842e-7 and mean 5.1666e-8 versus tolerance 1e-4 |

### 7.1 Completed headline evidence

| Measurement | Seed 17, selected by validation only | Cross-seed qualification |
|---|---|---|
| Validation / test pinball loss | 0.1126337 / 0.0703921 | All three runs completed; validation losses 0.1126337–0.1143907 |
| Pooled normalized q50-forecast MAE | 0.2358698 | 0.2358698–0.2372377 |
| Versus persistence pooled MAE 0.2542983 | 7.2468% lower | 6.7089%–7.2468% lower for every seed |
| Versus train-only ridge pooled MAE 0.2297187 | 2.6777% higher | 2.6777%–3.2731% higher for every seed |
| Horizon-target point cells versus persistence | 27 wins / 29 losses | Mixed; improvements concentrated in delta and relative volume |
| Paired descriptive cells versus persistence | 21 favorable / 21 crossing zero / 14 unfavorable | Descriptive only; boundary-adjacent windows overlap |
| Horizon-target point cells versus direct ridge | 1 win / 55 losses | Ridge is the stronger point control |
| Paired descriptive cells versus direct ridge | 0 favorable / 20 crossing zero / 36 unfavorable | No cell's interval favored FDT-N |
| Central 80% coverage / POC direction hit rate | 0.789775 / 0.549031 | Coverage 0.789641–0.802303; direction 0.494392–0.549031 |
| Selected CPU latency (50 runs) | p50 11.4099 ms / p95 14.6406 ms / p99 17.2398 ms | Artifact timing only, not end-to-end chart latency |

The direct ridge result prevents a superiority claim. These comparisons are point-forecast evidence;
no task-matched distributional baseline was implemented.

## 8. Baseline and ablation registration

| ID | Comparator | Inputs | Outputs | Selection budget | Status |
|---|---|---|---|---|---|
| B01 | Zero-change persistence | same causal context anchor | same seven targets | none | Implemented point baseline |
| B02 | Rolling median / seasonal naive | same past only | same targets | frozen windows | **[PENDING]** |
| B03 | Existing online linear footprint predictor | its documented feature set | map common targets only | existing settings | **[PENDING]** |
| B04 | Direct ridge | latest, context-mean, and context-trend bar aggregates | independent point output for every horizon/target | alpha fixed at 1.0; training windows only | Implemented point baseline |
| B05 | Quantile boosted trees | aggregate features | q10/q50/q90 | validation only | **[PENDING]** |
| B06 | Aggregate MLP | bar features | direct horizons | matched tuning | **[PENDING]** |
| B07 | Causal TCN or GRU | full or matched structural tokens | direct horizons | matched parameters | **[PENDING]** |
| B08 | DeepLOB-like structural stem | row matrix | direct horizons | matched parameters | **[PENDING]** |
| B09 | PatchTST/iTransformer/Crossformer/TFT adaptation | declared fixed adapter | same targets | matched search cap | **[PENDING]** |
| B10 | Moirai | frozen causal adapter | same measurable targets | zero-shot and optional frozen fine-tune budget | **[PENDING]** |

CHMT-Q is deliberately absent from this table: Section 5 records why its inputs, targets, period,
venue, and restricted data make it prior negative evidence rather than a task-matched baseline.

## 9. Artifact and provenance ledger

| Artifact | Path or identity | SHA-256 | Status |
|---|---|---|---|
| Raw market archives | 14 official Binance Data Vision BTCUSDT spot individual-trade ZIPs, 2024-01-01..14; original total 374,843,570 bytes | Individual SHA-256 values in source manifest | Provenance frozen; local copies truncated; raw bytes externally recoverable |
| Source-data manifest | `tmp/volume-footprint-transformer/btcusdt-1m-fourteen-days.manifest.json` | `b88d89565febb97dbe56151fe447e3952fa844b8c433ecb86cb747e11e4ce412` | Frozen for current dataset |
| Encoded dataset | `tmp/volume-footprint-transformer/btcusdt-1m-fourteen-days.npz` (9,963,548 bytes) | `d83f78894af08b65123d2bf9f4fbef8ca28e901a552d3af2c61f6e37513bf6cc` | Frozen for current run |
| Split indices | One `splits.npz` per seed; deterministic counts 13,990 / 2,869 / 2,942 with 72-bar purges | Byte hashes in final PDF | Arrays must be identical across seeds |
| Training configuration | Schema-v2 result JSON for every declared seed | Result hashes in final PDF | All fields except seed must match |
| Selected checkpoint | Validation-loss-selected seed-17 `best.pt` | `9d2cf525971f81fc93ca9ed97103cffb29203d7fd59edea5bd92d0e1480e124a` | Shadow-only research; best epoch 21 |
| ONNX model | Selected export `fdt-n.onnx` (1,834,205 bytes) | `adce8cd39312bf3ad1389a23ed5ca24d6f47a7f58dbb26d76ae7d54af1011af7` | Shadow-only research |
| Parity fixture / runtime evidence | Selected export directory and `metadata.json` | Metadata `9f998ac6b56b008ac02aaa0ed5c4f08447b1735f33027294773d9e1aa0867763` | Parity passed; 50-run CPU timing recorded |
| Results JSON | Seeds 17 / 29 / 43 | `746403a05790753943b8c22f615f097c8dc397e413be49f35ea553b82e0cbd0a` / `1b3e19b476b65e27becb30aaf600ea0384477b64b13f4d0d3eb2abddf3d0e0cc` / `4d28b1e41c6f93e23f6c86261d7710a9974a58c5985aab7c029415f2a4e605ad` | Primary selected by validation loss only; all held-out results retained |
| Environment record | Python/platform/PyTorch/NumPy plus ONNX/ORT versions in selected metadata | Metadata hash in final PDF | Artifact-specific environment disclosure |
| Research paper PDF | `output/pdf/fdt-n-volume-footprint-transformer-research-paper.pdf` | Generated file hash returned by reporter | Created only after all final gates pass |

## 10. Claim-writing rules for the final revision

Use:

- “On the frozen test period, FDT-N q50-forecast MAE minus point baseline **Y** was **X** for target **K**,
  horizon **H** (paired four-hour descriptive sensitivity interval **[A,B]**); adjacent boundary
  windows overlap, so this is not an independent-block confidence interval or generalization bound.”
- “Against distributional baseline **Y**, FDT-N changed qτ pinball loss by **X** for target **K**,
  horizon **H**, under the preregistered dependence procedure.”
- “Empirical q10/q90 coverage was **X/Y** for target **K**, horizon **H**, over **N** valid labels.”
- “The result is limited to instrument, venue, interval, row-size, and period **Z**.”

Do not use:

- “understands the auction”;
- “predicts the market”;
- “confidence interval” for a quantile interval without qualification;
- “calibrated” from pooled coverage alone;
- “generalizes” from one period or instrument;
- “profitable” without a separate prespecified execution study;
- “lossless” for any compaction of distinct price rows;
- “causal effect” for an observational forecast association;
- “production-ready” from unit tests, export success, or a development score.

## 11. Open gates

1. Preserve retrieval/terms evidence for the externally recoverable Binance archives and their 14 checksums.
2. Record event/receive/availability provenance beyond exchange timestamps, sufficient for causal sealing.
3. Complete the target-mask and upstream-exclusion audit; normalized bar/target clipping and encoded row-bound saturation are already reported.
4. Add stronger task-matched distributional baselines and registered ablations.
5. Reserve a distinct calibration period and untouched outer holdout.
6. Specify and validate delta-fraction-to-physical-delta marginal reconstruction.
7. Add conditional slices and later outer-period multi-seed comparisons.
8. Extend passed artifact-level ONNX parity/latency evidence with peak memory and live fail-closed tests.
9. Preserve trained-artifact hashes and result cells without replacing unmeasured work by estimates.
10. Regenerate and visually inspect the PDF after any source, result, or selected metadata change.
