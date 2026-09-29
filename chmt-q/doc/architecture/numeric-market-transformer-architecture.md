# Numeric Market Transformer — Architecture and Delivery Specification

- **Status:** CHMT-Q reference model and fail-closed development trainer implemented; causal .NET
  compiler, valid capture, live pipeline, and chart integration remain proposed
- **Last updated:** 2026-07-24
- **Scope:** Windows Professional overlay, Volume Footprint integration
- **Confidentiality:** Pro-private design. Do not copy architecture, feature contracts, weights, or
  training implementation into public/.
- **Source context:** [SESSION-CONTEXT.md](../SESSION-CONTEXT.md)
- **Visual guide:** [visual-guide.md](../visual-guide.md)

---

## Document map

- Sections 1–11: scope, boundaries, provenance, clocks, and example construction.
- Sections 12–17: tensor/target contracts and the custom transformer itself.
- Sections 18–29: capture, lake, compiler, training, evaluation, and promotion.
- Sections 30–36: ONNX packaging, isolated serving, chart runtime, scoring, and recovery.
- Sections 37–44: repository map, delivery phases, verification, operations, risks, and done criteria.

---

For the selected CHMT-Q v1 model, objective, normalization, and raw-output contracts, the later
[`research-spec.md`](../research-spec.md) and its executable Pro-private Python
contracts are authoritative. This architecture remains authoritative for system boundaries, capture,
the .NET compiler, serving, and product integration. Earlier point-forecast wording below must not be
used to override the selected probabilistic contract.

## 1. Purpose

This document specifies a new transformer built for numeric market data. It is not a wrapper around a
pretrained model and it does not tokenize text, images, symbols, news, or chart pixels. It starts from
random weights and learns only from versioned local tensors derived from:

- finalized OHLCV bars;
- canonical L1 quotes;
- canonical top-10 L2 order-book snapshots;
- real trade-tape footprint bars used only as supervised targets.

The first consumer is the Professional Volume Footprint chart. The transformer predicts the same five
numeric quantities the chart can already render as future ghost bars:

1. total point of control;
2. buy point of control;
3. sell point of control;
4. total volume;
5. delta.

The model produces direct forecasts for horizons 1 through 5. Training is offline. The desktop
application performs bounded inference and delayed scoring only.

This specification answers four questions:

1. What exactly enters and leaves the model?
2. What is the custom transformer architecture?
3. How is data captured, compiled, trained, evaluated, packaged, and promoted?
4. How does the model run safely inside Volume Footprint without blocking or leaking?

---

## 2. Decision summary

| Decision | V1 choice |
|---|---|
| Product scope | Professional Windows first |
| Public boundary | Neutral provenance fixes and an opaque optional provider seam only |
| Model ownership | Architecture, tensor builder, trainer, runtime, packages, and weights stay Pro-private |
| Model origin | Random initialization; no pretrained weights |
| Tensor engine | PyTorch for tensor operations, autograd, optimizer, and optional GPU training |
| Transformer implementation | Custom primitive Q/K/V attention and blocks; no prebuilt attention/transformer |
| Serving format | Fixed-shape ONNX, batch size 1 |
| Serving process | Long-running Pro Python/native worker hosting ONNX Runtime; loopback HTTP/JSON |
| Initial coordinate | One InstrumentId × one broker/source × one 1-minute timeframe |
| Input context | 64 fixed one-minute finalized-bar slots plus 120 seconds of L1/L2 at 500 ms |
| L2 depth | Top 10 bid and top 10 ask levels |
| Forecast horizons | 1, 2, 3, 4, and 5 footprint bars |
| Targets | POC, buy POC, sell POC, total volume, delta |
| Label eligibility | Genuine tape only; synthetic L1-derived prints are excluded |
| Rollout | Offline gate, then Pro shadow mode, then explicit opt-in, then possible default |
| Fallback | Existing online RLS and regression forecasts |
| Live execution | None; forecasts are informational |

Values such as sequence length, grid cadence, layer count, and learning rate are proposed V1 defaults.
They are frozen only for a specific dataset/model package and may change in a later schema version
after measured ablation and latency results.

---

## 3. Non-goals

V1 does not:

- use a language model, tokenizer, pretrained time-series checkpoint, or external embedding;
- consume text, news, images, strategy output, account state, orders, or positions;
- train or fine-tune inside the desktop application;
- perform online gradient updates from live chart data;
- share one model across brokers, instruments, or timeframes;
- support 15-second or 30-second footprint intervals;
- introduce live order submission or execution;
- publish Pro architecture or weights to public/;
- promise that a transformer will beat the existing linear baseline.

A failure to beat simple baselines on untouched data is a valid project result. Complexity is not a
promotion criterion.

---

## 4. Architectural invariants

The implementation must preserve all of these rules.

### 4.1 Causality

A forecast cut at availability time t may use only observations with AvailableTimeUtc less than or
equal to t. Exchange/event time is a feature and ordering signal; it is not evidence that data was
available to the application.

### 4.2 Provenance

Every raw observation used for a model example is attributable to:

- canonical InstrumentId;
- broker/source;
- event time;
- ingest or availability time;
- sequence or snapshot identity where available;
- approximation/quality flags;
- dataset file and content hash.

Cross-source rows never enter one V1 example.

### 4.3 Train-serving parity

One Pro WPF-free NumericMarketTensorBuilder owns feature ordering, sampling, masks, semantic domain
validation/bounds, and target transforms. The offline dataset compiler and live provider call the same
implementation. Python reads compiled tensors; it does not reimplement market feature engineering.
Fold-fitted robust scaling and its z-clamp live in the training/export wrapper, with final constants
embedded in ONNX.

### 4.4 Bounded runtime

Raw rings, request channels, pending predictions, logs, and score histories have hard capacities.
Model load and inference never execute in the WPF process. The host schedules requests away from the UI
thread, and a supervised Pro subprocess owns the Python/native ONNX Runtime session. It binds only to
127.0.0.1 and exposes a versioned HTTP/JSON seam, as required by ADR-0006. Restart and disposal cancel
all owned work and late results cannot update a new viewGeneration.

### 4.5 Fail closed, degrade gracefully

An invalid dataset, package, schema, checksum, tensor shape, NaN, timeout, or stale modality produces no
transformer forecast. It does not stop market ingest or the chart. Volume Footprint retains current
RLS/regression behavior and reports the fallback reason.

### 4.6 Reproducibility

Every candidate identifies the exact data, feature schema, source revision, configuration, random seed,
dependency versions, hardware description, training trace, evaluation folds, and exported bytes.

### 4.7 Open-core boundary

Public code may define source-correct market observations and an opaque provider/result contract.
Public code must not contain Pro feature names, tensor shapes, architecture, training commands, package
format details, weights, or model-selection policy.

---

## 5. System context

### 5.1 Existing reusable behavior

The shared Volume Footprint view model already:

- resolves InstrumentId and broker/source;
- joins the canonical ingest/hub pipeline;
- builds footprint bars from real or synthetic prints;
- seals bars at fixed chart intervals;
- invokes ML once per sealed bar;
- publishes MlPredictedBar values;
- renders violet forecast ghost bars;
- scores the current online learner against a regression baseline;
- uses bounded trade buffering, batched drain, coalesced render, and deterministic disposal.

The new provider must attach to those seams without replacing the chart's ingest or renderer.

### 5.2 Gaps that must be resolved first

The current source has data-contract gaps that make transformer training unsafe:

- finalized live bars are published but intentionally not persisted by PumpBarsAsync, so model capture
  needs a separate availability-stamped path rather than silently changing tick-primary storage;
- finalized bars expose open time but not explicit close/availability time;
- DepthSnapshot contains timestamp and levels but not InstrumentId, source, ingest time, sequence,
  approximation, or snapshot identity;
- historical depth reads can merge brokers and lose write-side provenance;
- archive depth export currently fabricates source and ingest time;
- the local Parquet lake omits L2;
- persistence can drop oldest queued records without a dataset-quality counter;
- the existing model registry is a small JSON/SQLite store for online linear weights, not ONNX blobs.
- current quantity fields are integer-oriented; a fractional-quantity venue needs an exact neutral
  fixed-point/decimal contract or must be excluded from the V1 coordinate.

No architecture or training work is considered production-valid until these gaps have executable
round-trip and causality tests.

---

## 6. Ownership and dependency boundary

### 6.1 Shared public responsibilities

Future public changes are limited to generally correct platform capabilities:

- a provenance-preserving depth observation/read seam;
- explicit finalized-bar availability semantics;
- source-scoped depth history;
- persistence drop/failure telemetry;
- provenance-correct raw depth export support that a Pro capture/compiler can consume;
- an opaque optional footprint forecast provider and Null implementation;
- a public result shape compatible with the chart;
- shared lifecycle/fallback tests.

These changes must make sense without knowing that the Pro transformer exists.

### 6.2 Professional responsibilities

The Pro overlay owns:

- exact tensor feature contract;
- causal tensor builder;
- dataset compiler;
- Python architecture and training package;
- experiment store and checkpoints;
- evaluation and promotion policy;
- ONNX package schema and binary registry;
- HTTP client/process supervision and an isolated Python/native ONNX Runtime worker;
- Pro dependency-injection registration;
- model picker/status UI additions that expose only safe metadata;
- Pro model/runtime tests;
- all weights and datasets.

### 6.3 Dependency direction

The intended dependency flow is:

~~~text
public Core neutral market + forecast contracts
        ^                         ^
        |                         |
public MarketData           public VolumeFootprint
provenance/store/export     view model/rendering
        ^                         ^
        |                         |
Pro NumericMl.Core          Pro NumericMl.Client
        ^                         ^
        |                         |
Pro dataset CLI             Pro shell composition
                                  |
                     loopback HTTP/JSON (127.0.0.1)
                                  |
                                  v
                    Pro Python/native ONNX worker
                                  |
                         model package registry
~~~

Public Core never references ONNX Runtime, Python, storage implementations, WPF, or a Pro assembly.
The C# build contains only the neutral seam and managed HTTP/process client. The worker is built and
packaged independently, so Python/native inference dependencies do not enter the C# project graph.

---

## 7. End-to-end flow

~~~text
Canonical broker ingest
  |
  +--> L1 quote observations -----------+
  +--> L2 depth observations -----------+--> raw local Parquet lake
  +--> finalized bar observations ------+
  +--> real trade prints (labels only) -+
                                           |
                                           v
                                 audit + source isolation
                                           |
                                           v
                              Pro causal tensor compiler
                                           |
                          +----------------+----------------+
                          |                                 |
                          v                                 v
                  immutable Arrow shards              dataset manifest
                          |                                 |
                          +----------------+----------------+
                                           |
                                           v
                              custom PyTorch transformer
                                           |
                             purged walk-forward evaluation
                                           |
                         reject <----- promotion gates -----> candidate
                                                               |
                                                               v
                                                  ONNX + verified hash set
                                                               |
                                                               v
                                                   atomic local registry
                                                               |
                                                               v
                                         Pro Volume Footprint provider session
                                                               |
                                  bounded latest-only worker request
                                                               |
                                                               v
                                      isolated loopback ONNX worker process
                                                               |
                                                               v
                                             ghost bars + delayed live scoring
~~~

---

## 8. Clock and ordering model

Market data carries several clocks. They must not be conflated.

| Clock | Meaning | Use |
|---|---|---|
| EventTimeUtc | Venue/broker event timestamp when supplied | Market age, lag, event ordering |
| IngestTimeUtc | When the terminal first received/normalized the event | Latency/quality diagnostics |
| AvailableTimeUtc | When the complete observation became downstream-usable | Sole feature causality cutoff |
| OpenTimeUtc | Beginning of an OHLC interval | Bar identity, never availability |
| FeatureCutoffUtc | Fixed interval boundary used to select inputs | Example boundary |
| DecisionIssuedUtc | Actual time the delayed scheduler/request ran | Operational latency only |
| TargetTimeUtc | Future footprint interval being predicted | Label identity only |

The deterministic raw replay order is:

1. AvailableTimeUtc;
2. EventTimeUtc;
3. FeedEpochId bytes;
4. stable stream rank;
5. StreamSequence;
6. ObservationId bytes.

The V1 stream rank is bar, quote, depth, trade-label. The chosen rank is not market truth; it is a
deterministic tie-breaker recorded in the feature schema. A sequence number from one stream is never
compared as though it were global across another stream.

Late observations are retained for quality reporting but do not mutate an already-emitted tensor
example. Dataset compilation and live inference follow the same rule.

---

## 9. Raw observation contracts

The names below describe the required information, not final public type names.

Every captured raw observation carries one provenance envelope. FeedEpochId and the terminal sequence
are stamped before shared fan-out; the Pro capture layer adds CaptureSessionId:

| Field | Type | Rule |
|---|---|---|
| CaptureSessionId | UUID | one explicit operator capture operation/raw manifest |
| FeedEpochId | UUID | changes on reconnect/resubscribe; multiple epochs may occur, IDs never repeat |
| ObservationId | SHA-256 | canonical identity described below |
| InstrumentId | canonical identity | required |
| Source | BrokerKind | required; never inferred on read |
| StreamKind | quote/trade/depth/bar | required |
| StreamSequence | int64 | required terminal-assigned monotonic value inside FeedEpochId + stream |
| SourceSequence | nullable int64/string | original source sequence when available |
| EventTimeUtc | UTC timestamp | source event time |
| IngestTimeUtc | UTC timestamp | first adapter receipt time |
| AvailableTimeUtc | UTC timestamp | stamped once before downstream fan-out; sole causality gate |
| EventTimeApproximate | bool | required |
| QualityFlags | versioned bit set | required |

ObservationId is SHA-256 over a canonical binary encoding of FeedEpochId, InstrumentId, Source,
StreamKind, and StreamSequence. The terminal sequence is assigned even when the venue supplies none;
SourceSequence remains evidence rather than identity. An exact duplicate ObservationId with identical
payload bytes is counted and deduplicated. Conflicting bytes under the same ID are fatal. Reconnect
creates a new FeedEpochId because current source sequences can restart, while the operator capture
session may continue and records every epoch transition.

### 9.1 QuoteObservation

| Field | Type | Required |
|---|---|---|
| Bid, Ask | exact decimal/fixed-point raw price | yes |
| BidSize, AskSize | exact decimal/fixed-point quantity | yes |

### 9.2 DepthObservation

| Field | Type | Required |
|---|---|---|
| SnapshotId | stable source-scoped identity | yes |
| Bids | descending price levels | yes |
| Asks | ascending price levels | yes |

Storage must group flattened rows by InstrumentId, Source, SnapshotId, and timestamp—not timestamp
alone.

### 9.3 FinalBarObservation

| Field | Type | Required |
|---|---|---|
| Size | BarSize/timeframe | yes |
| OpenTimeUtc | UTC interval open | yes |
| CloseTimeUtc | UTC interval close | yes |
| Open, High, Low, Close | exact decimal/fixed-point raw price | yes |
| Volume | exact decimal/fixed-point quantity | yes |
| IsFinal | bool | yes |
| Derived | bool | yes |

Only final bars are model inputs. Native and derived bars are different model coordinates unless a
future evaluation explicitly proves parity.

### 9.4 TradeLabelObservation

The payload adds exact decimal/fixed-point price and quantity plus aggressor classification to the
common envelope. It remains label-only. A training row must use a genuine native tape from the same
InstrumentId and source as the inputs. Synthetic quote-derived prints are rejected before label
construction.

---

## 10. V1 model coordinate

A model package is valid for exactly one coordinate:

~~~text
model_kind
instrument_id
broker_source
footprint_timeframe
bar_input_timeframe
feature_schema_hash
target_schema_hash
tensor_builder_version
exact_tick_size_and_price_scale
quantity_unit_and_scale
contract_multiplier_and_volume_semantics
session_calendar_version
~~~

The first implementation uses:

- footprint timeframe: 1 minute;
- bar input timeframe: 1 minute;
- lookback: 64 fixed one-minute finalized-bar slots, including explicit missing slots;
- microstructure lookback: 120 seconds;
- microstructure cadence: 500 ms;
- microstructure steps: 240;
- L2 depth: 10 levels per side;
- horizons: 1, 2, 3, 4, 5.

Raw Parquet uses Decimal128 or an equivalent checked fixed-point integer plus the coordinate scale;
fractional crypto quantities must not be truncated into integer contracts. Tick size is stored as an
exact decimal/rational. The compiler converts price to integer tick coordinates with the declared
AwayFromZero rule and rejects non-representable values rather than relying on binary floating-point
equality. OHLC volume is a usable input only when its unit/multiplier is known and convertible to the
coordinate quantity unit; otherwise that bar/coordinate is ineligible.

One of Interactive Brokers, Binance, or IronBeam should be selected only after the audit proves
simultaneous native tape, L1, L2, and finalized-bar coverage. The architecture does not choose the
venue in advance.

---

## 11. Decision-time example construction

### 11.1 Forecast cutoff

V1 uses a monotonic wall-clock interval state machine shared by offline replay and the Pro live
provider:

~~~text
Open(k) -> AwaitDecisionLag(k) -> Decided(k) -> LabelFinal(k) -> Scored(k)
~~~

For every one-minute interval k, the scheduler runs exactly once at interval close plus the versioned
decision_lag_ms. The logical feature cutoff remains the interval close; the delay gives the host time
to drain already-available work but does not admit observations stamped after the boundary. It does not
wait for the next print and never reopens an earlier interval. A host that misses a live boundary
records a missed decision instead of backfilling a forecast with later data. Let:

- t be FeatureCutoffUtc, equal to interval close;
- issued_at be the actual DecisionIssuedUtc after the configured operational lag;
- q be instrument tick size;
- p_ref be the latest valid L1 mid available at t, otherwise the latest valid finalized-bar close;
- ell_ref be the causal EMA of log(1 + finalized OHLC volume) converted to the coordinate quantity unit,
  using only bars available by t.

The quality vector records which anchor was used. If neither anchor or no valid ell_ref exists, the
example is ineligible. The builder records the reason and does not guess a reference price.
The decision record retains p_ref, q, and ell_ref for output reconstruction; they are not separate learned
ONNX inputs.

V1 uses EMA span 20 (alpha = 2 / 21), seeded from the first valid log-volume and requiring at least five
valid bars. For each chronological bar, log_volume_ema_residual uses the EMA state before that bar, then
the state updates; ell_ref at cutoff is the post-update state after all eligible bars. The state resets
at the same session/reconnect/gap boundaries as bar context. Span, seed, minimum history, and update
order are versioned contract fields.

### 11.2 Session boundaries

Context never carries across:

- a configured session boundary;
- broker reconnect;
- instrument/source change;
- a gap larger than the configured reset threshold;
- an explicit data-quality reset.

The builder may left-pad earlier context with masked rows. It may not fill an overnight or reconnect
gap by carrying the previous observation forward.

### 11.3 Fixed-grid sampling

The 240 microstructure points are:

~~~text
t - 119.5s, t - 119.0s, ..., t - 0.5s, t
~~~

At each grid point, the builder selects the newest observation whose availability time is not later
than that grid point. It then validates freshness:

- proposed L1 maximum age: 2 seconds;
- proposed L2 maximum age: 1 second.

These are schema values, not hard-coded constants. A stale observation produces a masked step and
contributes its staleness to the causal quality summary; its masked row is zeroed before attention. It
is never a silently valid carried value.

Every accepted L2 grid snapshot also requires a fresh, valid two-sided L1 midpoint at that grid time.
That midpoint is the causal level-price reference even when the depth snapshot itself is one-sided. If
no such L1 reference exists, l2_snapshot_mask is false and every level mask for that snapshot is false.

The finalized-bar fallback for the decision anchor has its own schema-bound freshness, proposed as at
most two completed one-minute slots. A bar older than that cannot rescue an otherwise ineligible cutoff.

### 11.4 Out-of-order data

An event that arrives after an example cutoff cannot revise the example. It increments late/out-of-
order counters and remains available only for later cutoffs if it passes source/session rules.
Negative event-to-available lag caused by clock skew is retained within clipping bounds and flagged; it
is never silently rewritten to zero.

The next five targets are the five consecutive wall-clock intervals after t. Target labels are sealed
at their close plus label_grace_ms, using only genuine prints whose event time belongs to that interval
and whose AvailableTimeUtc is within the grace. A later print is counted but cannot rewrite a label.

---

## 12. Tensor contract

Training may use batches larger than one. Exported inference uses fixed batch size one.
The graph accepts only float32 numeric tensors and boolean validity masks and returns one float32 tensor;
no string, enum name, tokenizer output, object, or text token crosses the model boundary.

ONNX inputs:

| Name | Training shape | ONNX shape | Dtype |
|---|---:|---:|---|
| bars | [B, 64, 16] | [1, 64, 16] | float32 |
| bars_mask | [B, 64] | [1, 64] | bool |
| l1 | [B, 240, 16] | [1, 240, 16] | float32 |
| l1_mask | [B, 240] | [1, 240] | bool |
| l2 | [B, 240, 2, 10, 12] | [1, 240, 2, 10, 12] | float32 |
| l2_level_mask | [B, 240, 2, 10] | [1, 240, 2, 10] | bool |
| l2_snapshot_mask | [B, 240] | [1, 240] | bool |
| quality | [B, 16] | [1, 16] | float32 |

Training-only labels:

| Name | Shape | Dtype |
|---|---:|---|
| targets | [B, 5, 5] | float32 |
| target_mask | [B, 5, 5] | bool |
| presence_labels | [B, 5, 2] | float32 |
| presence_mask | [B, 5, 2] | bool |

ONNX output:

| Name | Shape | Dtype |
|---|---:|---|
| raw_output | [1, 5, 17] | float32 |

The reconstructed median view remains `[1,5,5]`, but it is a host result derived from ordered
quantiles and is not the raw model output.

Masks are separate tensors. Invalid feature rows are zero after transformation, but zero never implies
validity.

### 12.1 Bar fields

The bar axis is the 64 consecutive one-minute clock slots ending at the feature cutoff, oldest first;
it is not the last 64 non-empty bars. A slot is valid only when its final observation was available by
the cutoff. For each valid bar, values are relative to the previous valid close c_prev:

| Index | Field | Definition |
|---:|---|---|
| 0 | open_offset_ticks | (open - c_prev) / q |
| 1 | high_offset_ticks | (high - c_prev) / q |
| 2 | low_offset_ticks | (low - c_prev) / q |
| 3 | close_offset_ticks | (close - c_prev) / q |
| 4 | log_return | log(close / c_prev), when prices are positive |
| 5 | range_ticks | (high - low) / q |
| 6 | body_ticks | (close - open) / q |
| 7 | upper_wick_ticks | (high - max(open, close)) / q |
| 8 | lower_wick_ticks | (min(open, close) - low) / q |
| 9 | close_location | (close - low) / max(high - low, q) |
| 10 | log_volume | log(1 + volume) |
| 11 | log_volume_ema_residual | log_volume minus the causal trailing EMA |
| 12 | availability_age_s | max(0, t - AvailableTimeUtc) in seconds |
| 13 | preceding_missing_slots | `log(1+min(g,63))/log(64)` for preceding missing-slot count `g` |
| 14 | session_sin | sine of normalized exchange-session progress |
| 15 | session_cos | cosine of normalized exchange-session progress |

The builder retains at most one eligible close immediately before the 64-slot window solely to compute
c_prev for the oldest token; it is not a 65th model token. If that predecessor is unavailable, the
oldest slot is masked rather than given a fabricated return.

### 12.2 L1 fields

All price offsets use the cutoff reference price p_ref:

| Index | Field | Definition |
|---:|---|---|
| 0 | mid_offset_ticks | (mid - p_ref) / q |
| 1 | mid_change_ticks | change from the previous valid grid midpoint divided by q |
| 2 | spread_ticks | (ask - bid) / q |
| 3 | log_bid_size | log(1 + max(bid_size, 0)) |
| 4 | log_ask_size | log(1 + max(ask_size, 0)) |
| 5 | size_imbalance | (bid_size - ask_size) / (bid_size + ask_size) when sum > 0; else row invalid |
| 6 | microprice_offset_ticks | (microprice - mid) / q |
| 7 | bid_change_ticks | change from the previous valid bid divided by q |
| 8 | ask_change_ticks | change from the previous valid ask divided by q |
| 9 | bid_log_size_change | change from the previous valid log bid size |
| 10 | ask_log_size_change | change from the previous valid log ask size |
| 11 | observation_age_s | grid_time - AvailableTimeUtc |
| 12 | event_lag_s | AvailableTimeUtc - EventTimeUtc |
| 13 | sequence_gap | clipped/log-transformed gap from prior quote |
| 14 | carried_forward_flag | 1 when this grid row reuses an earlier observation |
| 15 | approximate_time_flag | 1 when event time is approximate |

Crossed or non-positive books are masked and counted.

### 12.3 L2 level fields

The tensor keeps bid/ask as an explicit axis and rank 0–9 as an explicit axis. Side and rank are
represented by learned internal embeddings; they are not encoded as arbitrary numeric ordinals.

| Index | Field | Definition |
|---:|---|---|
| 0 | signed_distance_ticks | (level_price - fresh grid-time L1 mid) / q |
| 1 | better_level_gap_ticks | distance from the next-better same-side level, zero at rank 0 |
| 2 | log_size | log(1 + max(size, 0)) |
| 3 | log_cumulative_size | log(1 + cumulative same-side size through this rank) |
| 4 | side_size_share | size / total top-10 same-side size when total > 0; else side invalid |
| 5 | cumulative_side_share | cumulative size / same-side total when total > 0; else side invalid |
| 6 | paired_rank_imbalance | bid/ask size imbalance at the same rank |
| 7 | snapshot_age_s | grid_time - AvailableTimeUtc |
| 8 | event_lag_s | AvailableTimeUtc - EventTimeUtc |
| 9 | sequence_gap | clipped/log-transformed gap from the prior depth snapshot |
| 10 | carried_forward_flag | 1 when this grid row reuses an earlier snapshot |
| 11 | approximate_time_flag | 1 when event time is approximate |

Missing levels are masked. A one-sided book may retain valid levels on one side while masking the
other side. paired_rank_imbalance is zero with the one-sided quality flag when its paired rank is
missing; zero is not treated as proof of balance because masks/quality remain explicit.
For every grid row, l2_snapshot_mask must equal accepted_and_fresh AND any(l2_level_mask). A true
snapshot mask with all levels false is an invalid tensor, not an empty-book token; builder, worker, and
model tests enforce this nested-mask invariant.

### 12.4 Quality vector

The quality vector is computed causally at the cutoff:

| Index | Field |
|---:|---|
| 0 | valid_bar_fraction |
| 1 | valid_l1_fraction |
| 2 | valid_l2_snapshot_fraction |
| 3 | valid_l2_level_fraction |
| 4 | l1_carried_fraction |
| 5 | l2_carried_fraction |
| 6 | approximate_l1_fraction |
| 7 | approximate_l2_fraction |
| 8 | quote_sequence_gap_rate |
| 9 | depth_sequence_gap_rate |
| 10 | late_bar_availability_fraction |
| 11 | out_of_order_observation_rate |
| 12 | one_sided_book_rate |
| 13 | crossed_or_rejected_book_rate |
| 14 | anchor_from_l1_flag |
| 15 | session_progress, normalized 0..1 |

Rates use only observations available at the cutoff and a versioned trailing window.

---

## 13. Normalization and clipping

Feature engineering produces stationary values, but their scale still varies. The .NET tensor builder
emits ordered engineered float32 values plus masks; it does not fit or update live statistics. Each
fold fits robust normalization on its training partition only.

For every continuous field:

~~~text
center = training median
scale  = max(1.4826 * training MAD, minimum_scale)
z      = (x - center) / scale
z      = clamp(z, -8, +8)
~~~

Binary flags, masks, normalized fractions, side, and bounded imbalances bypass robust scaling.

Scaler layout is fixed:

- bars center/scale/bypass arrays have shape [16] and are shared across 64 slots;
- L1 arrays have shape [16] and are shared across 240 grid rows;
- L2 arrays have shape [12] and are shared across all 240 x 2 x 10 positions;
- quality arrays have shape [16];
- target arrays have shape [5, 5], fitted separately by horizon/head over valid labels.

Bar close_location/session sine/session cosine; L1 size imbalance and binary flags; L2 size shares,
paired imbalance, and binary flags; and every bounded quality fraction/flag use identity center 0/scale
1. Target delta fraction also uses identity scaling so tanh applies exactly once in the model head. All
other listed float fields use their fold-fitted robust center/scale. Masks are never scaled.

Input normalization constants are:

- stored per fold in the training run;
- refitted once on the frozen final-development partition and copied into the candidate manifest;
- embedded in the exported ONNX wrapper as a required operation;
- hashed under a distinct normalization hash bound by the package manifest;
- never recomputed from live data.

Invalid rows are zeroed after normalization inside the wrapper. Scaler statistics never include masked
values. The graph therefore accepts the same raw engineered tensor contract in training parity tests
and live serving.

Price-offset and log-volume-residual targets may be robust-standardized for optimization using training-
fold-only constants. The export wrapper performs their inverse transform, so the serving output is
always the five values defined in Section 14. Delta fraction remains bounded in [-1, 1].

---

## 14. Target contract

For each horizon h in 1..5, let the realized future genuine-tape footprint bar contain:

- P_h: total POC;
- B_h: buy POC;
- S_h: sell POC;
- V_h: total volume;
- D_h: delta.

Label construction freezes the current footprint semantics in the private label contract:

- interval membership is [TargetOpenTimeUtc, TargetCloseTimeUtc);
- price bucket is Round(price / tick_size, AwayFromZero);
- total POC is the price row with maximum total volume;
- buy/sell POC use maximum classified buy/sell volume respectively;
- a tie selects the higher price, matching the current descending-row/strict-comparison behavior;
- buy/sell POC is missing when that side has zero eligible classified volume;
- total volume and delta sum only eligible genuine tape from the exact source.

Using cutoff reference price p_ref, tick q, and causal trailing log-volume reference ell_ref:

~~~text
y[h, 0] = (P_h - p_ref) / q
y[h, 1] = (B_h - p_ref) / q
y[h, 2] = (S_h - p_ref) / q
y[h, 3] = log(1 + V_h) - ell_ref
y[h, 4] = clamp(D_h / V_h, -1, +1) when V_h > 0; otherwise 0 with delta mask false
~~~

This definition prevents raw price level and absolute volume scale from dominating learning.

### 14.1 Target masks

Each head has its own validity bit:

All five heads first require the source-specific TapeCompletenessPolicy to pass for the full target
interval plus label grace. The policy must declare that the adapter can prove native-tape continuity and
must reconcile active subscription/connection or heartbeat, FeedEpoch continuity, source/terminal
sequence gaps, capture/ring/writer drops, and interval/session coverage. If the source cannot prove that
no prints were missed, an empty interval is unknown and every target is masked rather than called zero.

- total POC requires a valid non-empty genuine-tape bar;
- buy POC requires positive classified buy volume;
- sell POC requires positive classified sell volume;
- volume requires a completed interval with proven tape completeness; it may validly be zero;
- delta requires genuine aggressor classification coverage above the schema threshold.

Missing buy/sell POC is masked, not replaced with total POC for training.

### 14.2 Runtime reconstruction

The provider maps model outputs back to chart values:

~~~text
offsets       = clamp(forecast[h, 0:3], manifest_min_tick_offset, manifest_max_tick_offset)
log_residual  = clamp(forecast[h, 3], manifest_min_log_residual, manifest_max_log_residual)
delta_fraction = clamp(forecast[h, 4], -1, +1)

poc, buy_poc, sell_poc = snap_to_tick(p_ref + q * offsets)
log_volume = clamp(ell_ref + log_residual, manifest_min_log_volume, manifest_max_log_volume)
volume = max(0, checked_expm1(log_volume))
delta = checked_multiply(volume, delta_fraction)
~~~

The exact order is: verify all model/reference values finite; apply transformed-space clamps; perform
checked price/exp/multiply reconstruction; snap prices; verify finite values and invariants including
volume >= 0 and |delta| <= volume. Any failure rejects the entire five-horizon result before it becomes
the MlPredictedBar-compatible projection.

---

## 15. Custom transformer architecture

### 15.1 Design objective

The architecture must learn at three different structures:

1. price-level structure inside one L2 snapshot;
2. short microstructure evolution across L1/L2 time;
3. slower OHLC regime context across bars.

A single flat sequence would either discard book structure or create thousands of expensive tokens.
V1 therefore uses a hierarchical depth encoder, causal microstructure patch pooling, three independent
temporal encoders, quality-aware fusion, and a small direct horizon-query decoder.

~~~text
bars [64,16] -> project/time -> 3 causal blocks ------------------> 64 tokens --+
                                                                              |
L1 [240,16] -> project/time -> 4-frame patch -> 3 causal blocks -> 60 tokens --+-->
                                                                              |   gated memory
L2 [240,2,10,12] -> level block/pool -> 4-frame patch                         |   [184,128]
                                  -> 3 causal blocks ------------> 60 tokens --+       |
                                                                                      v
quality [16] -> quality MLP -> hard-aware gates + global context -----> 5 horizon queries
                                                                                      |
                                                         2 direct decoder blocks ------+
                                                                                      |
                                                                                      v
                                                        price[3] + volume[1] + delta[1]
                                                                  = forecast [5,5]
~~~

### 15.2 Proposed dimensions

| Component | V1 value |
|---|---:|
| Main model width d_model | 128 |
| L2 level width d_level | 32 |
| Attention heads | 4 main / 4 level |
| Head width | 32 main / 8 level |
| Bar temporal blocks | 3 |
| L1 temporal blocks | 3 |
| L2 level blocks | 1 |
| L2 temporal blocks | 3 |
| Horizon decoder blocks | 2 |
| Main GEGLU hidden width | 256 |
| Level GEGLU hidden width | 64 |
| Microstructure patch width | 4 x 500 ms frames |
| Dropout | 0.10 |
| Horizon queries | 5 |
| Output values per query | 5 |

Expected capacity is approximately 2.12 million parameters, with a hard V1 ceiling of 2.25 million.
The implementation prints the exact count and a component breakdown; the estimate is not an
acceptance result.

### 15.3 Input projections

Each modality has an independent projection:

~~~text
bar_token = Linear(16, 128)(normalized_bar)
l1_token  = Linear(16, 128)(normalized_l1)
l2_level  = Linear(12, 32)(normalized_level)
quality   = MLP(16, 128, 128)(quality_vector)
~~~

Projection bias is allowed. Invalid tokens are zeroed after projection and excluded by masks.
QualityMLP is Linear(16,128) -> SiLU -> Linear(128,128). All other lines above are one biased Linear.

### 15.4 Numeric time encoding

The model uses numeric ages rather than a text/token position vocabulary. For an age a and configured
scales tau_k:

~~~text
time_features(a) =
    [a,
     log(1 + a),
     sin(a / tau_1), cos(a / tau_1),
     ...,
     sin(a / tau_k), cos(a / tau_k)]
~~~

A small MLP projects these features to the branch width. Fixed scales cover the relevant short and
long ranges. The encoding is added to the input token together with a learned modality vector.

Ages are deterministic graph values derived before feature normalization, so no extra ONNX time input
is needed:

- bar slot-close ages are [63*60, 62*60, ..., 0] seconds;
- L1 and L2 grid ages are [119.5, 119.0, ..., 0] seconds;
- each four-frame patch uses its final frame age [118, 116, ..., 0] seconds.

Bar availability lag remains feature 12 and carried-observation age remains in L1/L2 features; those
quality values do not redefine token time. V1 time scales are [60, 120, 240, 480, 960, 1920, 3840]
seconds for bars and [0.5, 1, 2, 4, 8, 16, 32, 64] seconds for microstructure/patches. The arrays are
non-trainable buffers included in the model/config hash and exported as ONNX constants.
Each branch TimeMLP is Linear(time_feature_count,32) -> SiLU -> Linear(32,branch_width). Raw age and
log-age inputs are divided by the branch context maximum before entering the MLP; sine/cosine terms are
already bounded.

The level encoder also adds learned parameter vectors for bid/ask side and rank 0–9. These are model
parameters indexed by fixed side/rank constant arrays, not values read from l2 and not token IDs
supplied by the market-data pipeline; every external input remains numeric.

### 15.5 Custom multi-head attention

Every attention layer is implemented from Linear, reshape, matrix multiplication, masking, softmax,
dropout, and output projection. V1 source must not use torch.nn.Transformer,
torch.nn.TransformerEncoder/Decoder, torch.nn.MultiheadAttention, or a third-party transformer block.
The reference path also does not call a fused scaled-dot-product-attention API. ONNX Runtime may fuse
equivalent exported primitive nodes internally after parity tests pass.

For input X with width d:

~~~text
Q = X W_Q
K = X W_K
V = X W_V

Q, K, V -> reshape [batch, heads, tokens, head_width]

scores = (Q K^T) / sqrt(head_width)
scores = scores + optional_relative_time_bias + optional_modality_bias + attention_mask
weights = softmax(scores, axis=keys)
weights = where(any_valid_key, weights, 0)
weights = dropout(weights)
context = weights V
output = concat_heads(context) W_O
~~~

Mask rules:

- invalid keys receive negative infinity before softmax;
- invalid queries are zeroed after the block;
- temporal self-attention receives a lower-triangular causal mask;
- horizon cross-attention can see every valid historical memory token;
- L2 levels within one snapshot have no causal relationship and use only the level-validity mask.

Masking uses an ONNX-safe large negative constant and an explicit any-valid-key guard. An all-masked
row returns exactly zero and never emits NaN. Invalid queries are zeroed again after residual paths.

### 15.6 Relative-time bias

Temporal self-attention receives a learned decay per head from clipped log token-time distance:

~~~text
delta_ij = max(0, key_age_at_cutoff - query_age_at_cutoff)
bias_ij_head = -softplus(gamma_head) * log(1 + delta_ij / modality_scale)
~~~

This lets the model distinguish adjacent tokens from tokens separated by a longer logical time gap
without relying only on ordinal positions. The temporal causal mask separately requires a key to be
valid and no later than its query under the deterministic tie order.
Relative-time bias applies only to bar/L1/L2 temporal self-attention. Decoder cross-attention uses only
the modality gate bias, while level attention and horizon-query self-attention use neither bias.
modality_scale is fixed at 60 seconds for bars and 2 seconds for patched L1/L2, stored in config.

### 15.7 Pre-norm transformer block

V1 uses pre-normalization for stable training:

~~~text
u = x + Dropout(CustomAttention(LayerNorm(x), mask, time))
z = u + Dropout(FFN(LayerNorm(u)))
z = z * query_valid_mask
~~~

The main feed-forward network is GEGLU:

~~~text
g = GELU(Linear_g(128, 256)(x))
u = Linear_u(128, 256)(x)
FFN(x) = Linear_out(256, 128)(Dropout(g * u))
~~~

The level-width variant uses 32 -> 64 -> 32. V1 avoids BatchNorm and recurrent state.

### 15.8 L2 snapshot encoder

For every one of the 240 grid points:

1. flatten the explicit side/rank axes into 20 rows;
2. project the 20 level rows to width 32;
3. add side, rank, and snapshot-age encodings;
4. run one masked non-causal level-attention block;
5. apply one learned book-pooling query over valid levels;
6. project the pooled book token from 32 to 128.

The operation is vectorized across batch and time by reshaping:

~~~text
[B, 240, 2, 10, 12] -> [B * 240, 20, 12] -> [B * 240, 20, 32]
~~~

The result is one depth frame per microstructure time:

~~~text
depth_tokens: [B, 240, 128]
~~~

An all-missing or rejected snapshot emits a false snapshot mask, not a learned empty-book value. The
level encoder captures spread shape, top-level imbalance, gaps, and cross-side structure without
turning all 4,800 levels into one temporal attention sequence.

### 15.9 Causal microstructure patch pooling

Before temporal self-attention, L1 and pooled L2 frames are reshaped into 60 non-overlapping patches of
four 500 ms frames:

~~~text
[B, 240, 128] -> [B, 60, 4, 128] -> masked learned pooling -> [B, 60, 128]
~~~

Each patch may use only its own four preceding frames, receives the fixed patch-end logical time, and is
invalid when all four frame masks are false. Patch pooling is independently parameterized for L1 and
L2. It reduces quadratic temporal work while preserving the full two-minute observation window.
The learned pool score is Linear(128,32) -> tanh -> Linear(32,1), masked softmax over only the four
frames, followed by a weighted sum. An all-masked patch returns zero/false. The output then receives the
patch-end logical-time encoding.

### 15.10 Temporal encoders

Three independent causal encoders process:

- bars: [B, 64, 128], three blocks;
- patched L1: [B, 60, 128], three blocks;
- patched L2: [B, 60, 128], three blocks.

The branches do not share weights. Their final token states remain sequences; they are not reduced to
one vector before decoding.

### 15.11 Modality quality gates and global context

For each branch, s_m is the latest valid encoded token. Because every branch is causally encoded, this
token summarizes only that modality's available history. The quality MLP produces c_q. A hard-
availability-aware gate is:

~~~text
g_m = valid_modality_m * sigmoid(MLP(concat(s_m, c_q)))
alpha_m = g_m / (sum(g) + epsilon)
c = LayerNorm(c_q + sum(alpha_m * s_m))
~~~

Each modality has its own gate MLP: Linear(256,64) -> SiLU -> Linear(64,1). Weights are not shared.

The gate weights the global summary and contributes log(alpha_m + epsilon) as its cross-attention
modality bias; it does not also multiply memory values. A missing branch receives gate zero and its
memory mask becomes entirely false. This prevents zero-filled L2 from looking like a real empty book.
If all modalities are invalid, inference abstains before decoding.

The quality gate is not permission to invent a reference price. Examples without either a fresh L1
anchor or an eligible finalized-bar fallback are rejected before the model.

### 15.12 Fused memory

The encoded sequences are concatenated:

~~~text
memory = concat(bar_memory, l1_memory, l2_memory)
shape  = [B, 64 + 60 + 60, 128] = [B, 184, 128]
~~~

Every token retains a learned modality vector and numeric time encoding. The memory validity mask and
per-token log gate bias are concatenated in the same order. Quality enters through c and the gates; it
is not exposed as an unrestricted memory token.

### 15.13 Horizon-query decoder

Five learned base queries represent horizons 1 through 5. Each query also receives numeric horizon
features such as horizon count and horizon seconds:

~~~text
q_h = learned_query_h + HorizonMLP([h / 5, log(1 + h), h * timeframe_seconds]) + W_c(c)
~~~

HorizonMLP is Linear(3,32) -> SiLU -> Linear(32,128); W_c is Linear(128,128). They are shared across
the five queries, while learned_query_h is a [5,128] parameter.

Each of two decoder blocks performs:

1. pre-norm self-attention across the five horizon queries;
2. pre-norm cross-attention from queries to fused historical memory;
3. pre-norm feed-forward network.

Query self-attention lets adjacent horizons learn a coherent forecast curve without generating one
horizon autoregressively from another. Every horizon is still a direct forecast from the same past.

### 15.14 Output heads

One shared raw head maps each decoded query:

~~~text
raw = Linear(128, 17)(query)

for each of five targets:
    median = center
    q10    = center - softplus(lower_gap)
    q90    = center + softplus(upper_gap)

delta quantiles = tanh(delta q10, q50, q90)
presence_logits = raw[15:17]
~~~

The raw output is `[B,5,17]`: three center/gap parameters for each of five targets followed by buy/sell
POC-presence logits. The ordered training distribution is `[B,5,5,3]` plus `[B,5,2]`. The current
review export is the normalized raw core; a production wrapper must freeze inverse target scaling,
calibration, reconstruction, finite checks, and manifest-defined safety clamps before promotion.

### 15.15 Forward pass pseudocode

~~~python
def forward(
    bars, bars_mask, l1, l1_mask,
    l2, l2_level_mask, l2_snapshot_mask, quality
):
    bar_ages = BAR_SLOT_CLOSE_AGES_SECONDS
    micro_ages = MICRO_GRID_AGES_SECONDS
    patch_ages = MICRO_PATCH_END_AGES_SECONDS

    bar_x = bar_projection(bars) + bar_time_encoding(bar_ages)
    l1_x = l1_projection(l1) + l1_time_encoding(micro_ages)

    level_x = level_projection(l2)
    level_x += side_embedding(STATIC_SIDE_INDICES)
    level_x += rank_embedding(STATIC_RANK_INDICES)
    level_x += depth_age_encoding(micro_ages)

    effective_depth_mask = l2_snapshot_mask & any(l2_level_mask, axis=(side, rank))

    level_x = reshape(level_x, batch * micro_steps, sides * levels, d_level)
    level_m = reshape(l2_level_mask, batch * micro_steps, sides * levels)
    level_x = depth_level_block(level_x, valid_mask=level_m, causal=False)

    depth_x = book_pool(level_x, level_m)
    depth_x = reshape(depth_x, batch, micro_steps, d_level)
    depth_frame = depth_to_model(depth_x)

    l1_x, l1_patch_mask = causal_patch_pool(l1_x, l1_mask, width=4)
    depth_x, depth_patch_mask = causal_patch_pool(
        depth_frame, effective_depth_mask, width=4
    )
    l1_x += l1_patch_time_encoding(patch_ages)
    depth_x += depth_patch_time_encoding(patch_ages)

    for block in bar_blocks:
        bar_x = block(bar_x, bars_mask, ages=bar_ages, causal=True)
    for block in l1_blocks:
        l1_x = block(l1_x, l1_patch_mask, ages=patch_ages, causal=True)
    for block in depth_time_blocks:
        depth_x = block(depth_x, depth_patch_mask, ages=patch_ages, causal=True)

    context, gates = calculate_quality_context_and_gates(
        bar_x, l1_x, depth_x, quality
    )

    memory, memory_mask = concatenate_memories(
        bar_x, l1_x, depth_x,
        bars_mask, l1_patch_mask, depth_patch_mask,
        gates,
    )
    modality_bias = memory_log_gate_bias(gates)

    queries = horizon_queries(context)
    for block in decoder_blocks:
        queries = block(queries, memory, memory_mask, modality_bias)

    raw = shared_output_head(queries)  # [B,5,17]
    return monotone_quantiles_and_presence(raw)
~~~

The actual signature uses the named masks from Section 12; abbreviated pseudocode does not redefine
the contract.

### 15.16 Capacity and inference budget

| Component | Approximate parameters |
|---|---:|
| Numeric/time/position projections | 45k |
| Hierarchical L2 and patch pooling | 30k |
| Nine temporal blocks | 1.49M |
| Quality gates and fusion | 70k |
| Two horizon decoder blocks | 464k |
| Queries, context projection, norms, and heads | 25k |
| Total | 2,089,554 in the executable literal ledger |

The FP32 weights are approximately 8.1 MiB. The packaged ONNX target is below 12 MiB and incremental
per-session worker memory target is below 64 MiB, both measured rather than assumed. The proposed
warmed batch-one worker-only p95 target is at most 25 ms on the declared target CPU over at least 500
calls with production thread/provider settings. End-to-end HTTP serialization latency is measured
separately. If the worker misses its budget, the first approved reductions are two temporal blocks per
branch or a larger causal patch; inference never moves into the UI process.

---

## 16. Loss function

The selected objective is the research specification's mask-normalized quantile pinball loss plus
`0.25` median Huber, `0.20` buy/sell presence BCE, and the scheduled block-1 auxiliary copy of that
objective. Horizon weights are `1/sqrt(h)` and target weights are `[1,1,1,0.5,0.5]`.

Every component carries additive numerators and denominators across the complete effective batch and
evaluation set. Ratios from differently masked micro-batches are never averaged. A component with no
supervised terms contributes zero; a batch/window with no supervision fails explicitly. Exact equations,
presence-mask semantics, and auxiliary scheduling are in Sections 12 and 14 of the research
specification and are asserted by the executable loss/trainer tests.

---

## 17. Initialization and numerical rules

The model has no pretrained state.

- Linear weights: Xavier uniform.
- Linear biases: zero.
- Residual output projections: additional 1 / sqrt(2L) gain for their stack depth.
- LayerNorm scale: one; bias: zero.
- Learned queries/embeddings: seeded truncated normal distribution with standard deviation 0.02.
- Final head: Xavier-uniform weight and zero bias, consistently with every Linear layer.
- Dropout: enabled only in training.
- Training dtype: float32 baseline; mixed precision is optional after parity tests.
- ONNX inference dtype: float32.
- Gradient clipping: global norm 1.0.
- Every batch checks finite inputs, loss, and gradients.
- Every inference checks finite output before postprocessing.

The run manifest records Python, PyTorch, CUDA/driver when used, OS, CPU/GPU, and deterministic-mode
settings. A deterministic CPU reference run remains the parity authority even when GPU training is
used.

---

## 18. Raw data capture

### 18.1 Capture principle

The transformer dataset must not be inferred from whatever happens to remain in the current stores.
The default L2 retention is short, write queues are best-effort, and live bars are hub-only. V1 uses an
explicit capture session with a declared instrument, source, interval, start time, and quality policy.

The terminal remains tick-primary. The design does not change PumpBarsAsync to persist every live bar
in the canonical store. Instead:

- public market contracts expose provenance-correct observations;
- a Pro NumericMl capture component joins ref-counted canonical feeds;
- it writes append-only raw training lake partitions with its own drop/failure counters;
- historical store/archive data may be imported only through the same audit and source rules.

Live capture is owned by an explicit Pro App hosted service because only the running terminal has the
resolved instrument, broker session, and credentials needed to join live feeds. A Pro-only capture
control starts/stops a declared session; capture is off by default and stops during host shutdown. The
dataset CLI audits, imports, compiles, and seals already captured data, but it does not attempt to open
credentialed broker feeds independently.

### 18.2 Capture session state

~~~text
Created
  -> ResolvingInstrument
  -> Subscribing
  -> Capturing
       <-> Degraded
  Capturing | Degraded
       -> Draining
  -> Sealed

Any non-terminal state
  -> Failed
  -> Cancelled
~~~

Sealed means every writer has flushed, every partition has a hash, and the session manifest is
atomically published. A directory with data files but no sealed manifest is incomplete and cannot be
compiled.

Capturing may move to Degraded after a drop, sequence gap, write retry, reconnect, clock regression, or
stale required stream. The incident is journaled immediately. Reconnect closes the current FeedEpochId
and starts a new one; it never silently resumes a sequence under the old epoch. A source change ends the
capture session because it changes the model coordinate.

### 18.3 Streams

One capture session subscribes to:

- L1 quote + L2 depth through the ref-counted canonical ingest handle;
- finalized 1-minute bars through the bar feed;
- native trade tape for labels;
- timestamped capture/feed/store health events plus aggregate counters.

The session must prove all four streams come from the selected source. A source-mixed hub event is
rejected rather than accepted because InstrumentId matches.

### 18.4 Buffering and backpressure

Each raw stream has its own bounded channel and drop counter. L2 is never allowed to crowd quotes,
bars, or labels out of one shared queue.

Proposed starting capacities are operational settings, not model schema:

| Stream | Capacity | Full behavior |
|---|---:|---|
| Final bars | 4,096 | fail capture; bars must not drop silently |
| Quotes | 250,000 | drop oldest + mark capture degraded |
| Depth snapshots | 100,000 | drop oldest + mark capture degraded |
| Trades | 250,000 | drop oldest + mark labels degraded |
| Health journal | 16,384 | fail/unsealable if primary and bounded recovery writes both fail |

A capture with any drop may remain useful for research but cannot become a promotion dataset unless the
promotion policy explicitly allows the measured rate. The default promotion rule rejects drops.

### 18.5 Final-bar availability

The capture layer stamps AvailableTimeUtc when a final bar is received. OpenTimeUtc remains the interval
identity. If bars are derived, the derivation policy and source stream are recorded and Derived=true.

Native and derived bars do not share a model coordinate by default.

### 18.6 L2 snapshot identity

The capture layer assigns or preserves a stable source-scoped SnapshotId. Flattened level rows carry
that identity. Snapshot reconstruction never groups by timestamp alone.

### 18.7 Capture quality counters

At minimum, the manifest records:

- received, written, rejected, late, duplicate, and dropped rows per stream;
- first/last event, ingest, and available times;
- sequence resets and gaps;
- approximate-time counts;
- crossed, empty, and one-sided book counts;
- maximum and percentile event-to-ingest lag;
- writer failures and retries;
- process interruption/recovery;
- selected instrument/source and resolved canonical identity.

---

## 19. Raw lake

### 19.1 Layout

The lake is local and outside Git:

~~~text
raw-lake/
  schema-v1/
    source_key=<safe-hash>/
      instrument_key=<safe-hash>/
        available_date=YYYY-MM-DD/
          quotes/part-<sha256>.parquet
          trades/part-<sha256>.parquet
          depth-snapshots/part-<sha256>.parquet
          depth-levels/part-<sha256>.parquet
          bars/part-<sha256>.parquet
          quality-events/part-<sha256>.parquet
          batch-manifests/<sha256>.json
    captures/
      <capture-session-id>.json
~~~

The partition date comes from AvailableTimeUtc, never EventTimeUtc. A late observation therefore enters
a new immutable availability batch instead of rewriting an already closed event-date partition.
Path components are fixed-format hashes/allow-listed encodings of canonical IDs, never raw broker or
instrument strings; the manifest carries the actual identifiers. Partition names are descriptive only.
File hashes and manifest fields are authoritative.

### 19.2 Write protocol

The current local lake's direct-to-final and skip-if-exists behavior is not sufficient for model data.
The new writer uses:

1. create a unique staging file in the target filesystem;
2. stream bounded Parquet row groups;
3. flush and close;
4. calculate SHA-256 and row statistics;
5. atomically rename to its content-addressed or collision-free final name;
6. write one immutable content-addressed batch manifest;
7. atomically publish the batch manifest;
8. add its hash to the staging capture manifest;
9. publish the sealed capture manifest atomically after all writers drain.

Existing final files are never silently treated as the requested export. Their hashes and logical
ranges must match, or the new export gets a new identity.

### 19.3 Quote row

The quote Parquet schema retains every QuoteObservation field. Prices remain float64 in the raw lake;
tensor compilation converts transformed values to float32.

### 19.4 Depth snapshot and level rows

Depth uses two tables. One depth-snapshots row stores the common provenance envelope plus SnapshotId,
bid count, ask count, crossed/rejected status, and snapshot quality. One depth-levels row stores
SnapshotId, side, rank, price, and size. This preserves an atomic empty or one-sided snapshot and avoids
reconstructing identity from equal timestamps. The compiler requires every level row to resolve to
exactly one snapshot row.

### 19.5 Final bar row

The bar schema includes OpenTimeMicros, CloseTimeMicros, AvailableTimeMicros, OHLCV, source, size,
IsFinal, Derived, and quality flags.

### 19.6 Trade-label row

The raw trade schema retains event/ingest time, price, size, aggressor, source, sequence, and
approximation. The compiler verifies native-tape eligibility from capture metadata; it does not infer
eligibility from non-empty rows.

### 19.7 Quality-event row

Every drop, write retry/failure, sequence reset, reconnect, clock regression, and recovery transition is
an append-only event with CaptureSessionId, FeedEpochId, affected stream, AvailableTimeUtc, stable event
code, affected sequence/time range, count, and bounded diagnostics. Aggregate capture totals are derived
from this stream; they are not used to invent decision-time feature values. If neither the primary
journal nor its size/free-space-bounded recovery spool can persist an incident, the capture is failed
and cannot seal.

### 19.8 Capture manifest

The canonical JSON manifest includes:

~~~json
{
  "schemaVersion": 1,
  "captureSessionId": "<uuid>",
  "feedEpochIds": ["<uuid>"],
  "state": "sealed",
  "instrumentId": "<canonical>",
  "source": "<broker>",
  "startedUtc": "<utc>",
  "sealedUtc": "<utc>",
  "streams": {
    "quotes": { "rows": 0, "dropped": 0, "files": [] },
    "depthSnapshots": { "rows": 0, "dropped": 0, "files": [] },
    "depthLevels": { "rows": 0, "files": [] },
    "bars":   { "rows": 0, "dropped": 0, "files": [] },
    "trades": { "rows": 0, "dropped": 0, "files": [] },
    "qualityEvents": { "rows": 0, "files": [] }
  },
  "batchManifestHashes": [],
  "quality": {},
  "software": {},
  "sha256": "<canonical-manifest-hash>"
}
~~~

The manifest format is generic raw-data provenance. Model feature names and architecture are not
written into a public exporter. Its sha256 value is computed from canonical JSON with the sha256 field
omitted, avoiding a recursive self-hash.

---

## 20. Data audit

The first Pro CLI command is audit. It runs before tensor compilation and produces a machine-readable
report plus a concise human summary.

### 20.1 Audit checks

- manifest and file hash integrity;
- sealed capture state;
- instrument and source consistency;
- monotonic availability replay after deterministic tie ordering;
- duplicate identities;
- sequence reset/gap distribution;
- event-to-ingest lag distribution;
- quote validity, crossed market, zero/negative price and size;
- L2 side ordering, duplicate levels, empty/one-sided snapshots, level count;
- bar interval/close/availability consistency and final-only policy;
- native trade-tape eligibility;
- overlap of all input streams and label windows;
- grid coverage and staleness estimates;
- session boundary detection;
- capture/drop/write failure counters;
- proposed training, validation, and holdout session counts.

### 20.2 Audit result

~~~text
Pass
Warning
Fail
~~~

Fail blocks dataset compilation. Warning requires an explicit policy acceptance recorded in the
dataset manifest. The default promotion dataset policy rejects stream drops, mixed sources, unknown
depth provenance, bar look-ahead, and synthetic labels.

---

## 21. Causal tensor dataset compiler

### 21.1 Ownership

The compiler is a Pro .NET console tool using the same Pro NumericMarketTensorBuilder assembly as the
live provider. It reads raw Parquet, replays by availability, constructs examples, and writes immutable
raw-engineered numeric shards. It does not fit a global scaler.

Python does not read raw market tables for feature construction.

### 21.2 Compiler state machine

~~~text
Created
  -> VerifyingRawInputs
  -> Replaying
  -> BuildingExamples
  -> AssigningSplits
  -> WritingRawEngineeredShards
  -> VerifyingShards
  -> Sealed

Any active state -> Failed | Cancelled
~~~

Resume starts from the last verified immutable stage output. It never appends blindly to a partial
shard.

### 21.3 Replay algorithm

~~~text
open all selected raw partitions
verify capture manifests and hashes
merge rows by deterministic availability ordering
reset state at source/session/reconnect/gap boundaries

for each configured one-minute interval boundary:
    feature_cutoff = interval close
    decision_issued = feature_cutoff + decision_lag
    update feature state only with observations available by feature_cutoff
    build 64-bar context
    build 240-step L1 grid
    build 240-step L2 grid
    calculate causal quality vector
    record anchor and causal log-volume reference
    attach the next five fixed-interval genuine-tape targets after label grace
    emit example only if eligibility policy passes

after replay:
    assign chronological splits
    purge overlapping examples
    write immutable raw-engineered tensors, targets, masks, and metadata
~~~

The compiler may hold pending labels, but future target data is never fed back into feature state.

### 21.4 Empty intervals and current chart semantics

V1 chooses fixed clock boundaries. An interval without an eligible genuine print retains its horizon
position and has invalid POC/buy/sell/delta masks; volume is valid zero only when tape completeness for
that interval is proven, otherwise volume is also masked. Horizons are never compressed to the next
non-empty bar.

The current FootprintTimeBucketer seals when later prints arrive and can skip empty intervals, so it is
not the transformer decision clock. The Pro provider owns the monotonic boundary clock and matches
predictions/labels by explicit target interval. Existing chart/RLS sealing remains intact until an
independent shared-core change deliberately aligns it; the transformer must not infer its horizon from
OnBarSealed arrival order.

### 21.5 Out-of-order prints

The label builder cannot seal backward. Prints older than an already sealed interval are counted late
and excluded from that interval. This behavior must be shared by offline compilation and delayed live
scoring.

### 21.6 Interchange format

The proposed format is Apache Arrow IPC with fixed-size list columns. Reasons:

- no pickle/code execution;
- streaming record batches;
- efficient Python/PyArrow reads;
- fixed tensor dimensions;
- cross-language schema introspection;
- deterministic shard hashing.

Values in these shards are the raw engineered contract values from Sections 12 and 14, before robust
normalization. Fold-specific training loaders apply their frozen scaler. Optional normalized caches are
derived artifacts keyed by dataset, split, and scaler hash and never replace the canonical shards.

An implementation spike must prove stable .NET/Python round trips before Arrow is finalized. If it
fails, use a versioned simple binary tensor format plus JSON metadata; do not fall back to Python
feature duplication.

### 21.7 Shard row

Each row contains:

- example_id;
- feature_cutoff_micros and decision_issued_micros;
- session_id;
- anchor_price and anchor_source;
- log_volume_reference;
- flattened fixed-size bars tensor and mask;
- flattened fixed-size L1 tensor and mask;
- flattened fixed-size L2 tensor and mask;
- quality vector;
- target tensor and target mask;
- raw provenance range IDs;
- feature schema hash;
- target schema hash.

Reference values are needed only for reconstruction/evaluation; they are not silently concatenated to
model inputs.
Fold membership is not a single shard column because an expanding fold may later reuse a validation
session as training. split.json maps ExampleId/session groups to a role independently for every fold
and for the sealed outer holdout.

### 21.8 Dataset identity

The dataset ID is:

~~~text
SHA256(
    canonical semantic dataset manifest with dataset_id omitted
)
~~~

That semantic manifest already contains the ordered raw-capture/shard hashes, builder identity, schema
hashes, and split policy. Non-deterministic creation host, wall-clock duration, and local absolute paths
belong in a separate receipt.json and cannot change DatasetId.

Changing one input, field order, mask rule, split, normalization policy/schema, or label rule creates a
new dataset ID. Changing fitted fold/final scaler values changes the run/candidate normalization hash,
not the raw-engineered DatasetId.

---

## 22. Chronological split and leakage control

### 22.1 No random row split

Examples are grouped by trading session and ordered chronologically. Randomizing the full dataset
before splitting is forbidden.

### 22.2 Outer holdout

The most recent eligible block of sessions is sealed as an outer holdout before architecture tuning.
Its manifest and hash are recorded. Training code can read it only in the final evaluate command after
the candidate configuration is frozen.

### 22.3 Walk-forward folds

Use expanding or rolling session folds:

~~~text
fold 1: train sessions 1..N       validate N+1..N+M
fold 2: train sessions 1..N+M     validate N+M+1..N+2M
...
~~~

The exact N, M, and fold count depend on audited coverage and are stored in split.json.

### 22.4 Purge and embargo

Each example records its dependency interval from the earliest input slot through the final target close
plus label grace. With 64 one-minute bars and five future one-minute targets, the nominal span is at
least 69 minutes before session/reset effects. At every split boundary:

- purge examples whose input window crosses the boundary;
- purge examples whose future target horizon crosses the boundary;
- apply an embargo at least as long as the maximum dependency span unless an exact non-overlap
  calculation proves a smaller safe exclusion;
- reset session-dependent state.

The implementation computes overlap from actual timestamps, not a guessed row count.

### 22.5 Train-only fitting

Only the fold's training sessions may determine:

- feature median/MAD;
- target scale if used;
- clipping diagnostics;
- early-stopping optimizer decisions;
- any calibration or threshold.

Validation and holdout use frozen values.

Every fold writes an immutable scaler artifact under the run, containing centers/scales/bypass masks,
sample counts, training ExampleId/session-range hash, policy hash, and scaler hash. Training/validation
load the same raw-engineered shards and apply that fold artifact in the model wrapper. A validation
session that becomes training in a later expanding fold is normalized only by the later fold's newly
fitted training-only scaler.

### 22.6 Mini-batch order

Training examples may be shuffled within the training partition after features and targets are fixed.
Validation and evaluation remain chronological and deterministic.

---

## 23. Dataset manifest

The sealed manifest contains:

- dataset ID and schema version;
- instrument/source/timeframe coordinate;
- raw capture-session and feed-epoch IDs/hashes;
- eligible time/session range;
- decision-time and target rules;
- complete tensor field/shape/dtype/mask contract;
- staleness/reset/session policy;
- top-K and grid cadence;
- feature and target schema hashes;
- split and purge/embargo assignments;
- normalization policy/schema, but no fitted fold statistics;
- row counts by split and target-validity pattern;
- excluded row counts by reason;
- quality distributions;
- shard names, sizes, row counts, and hashes;
- builder source/assembly identity;
- deterministic compiler/tool identities that affect output;
- explicit warnings accepted by policy.

receipt.json may record creation host, tool runtime versions, durations, and local diagnostics outside
the semantic identity. No training run may modify the sealed dataset directory.

---

## 24. Pro tooling layout

Proposed private Python package:

~~~text
tools/numeric-transformer/
  pyproject.toml
  README.md
  configs/
    v1-reference.yaml
  daxalgo_numeric_transformer/
    cli.py
    config.py
    dataset.py
    masks.py
    metrics.py
    baselines.py
    checkpoint.py
    export.py
    package.py
    model/
      attention.py
      blocks.py
      depth_encoder.py
      temporal_encoder.py
      decoder.py
      numeric_transformer.py
    training/
      trainer.py
      scheduler.py
      walk_forward.py
      evaluator.py
  tests/
~~~

This is separate from public/tools/python-ml. The existing sidecar is an LLM/Paper Lab service and must
not inherit heavy PyTorch/ONNX training dependencies.

### 24.1 Proposed commands

~~~text
dotnet dax-numeric-ml audit       --config <capture-config>
dotnet dax-numeric-ml prepare     --config <dataset-config>
dotnet dax-numeric-ml verify      --dataset <dataset-id>

python -m daxalgo_numeric_transformer train
python -m daxalgo_numeric_transformer evaluate
python -m daxalgo_numeric_transformer export
python -m daxalgo_numeric_transformer inspect
~~~

These names are proposed. No command exists until implementation is authorized.

---

## 25. Training configuration

A canonical validated configuration is stored with every run.

~~~yaml
schema_version: 1
dataset_id: "<sha256>"
seed: 20260721

model:
  d_model: 128
  d_level: 32
  heads: 4
  bar_layers: 3
  l1_layers: 3
  depth_level_layers: 1
  depth_time_layers: 3
  decoder_layers: 2
  ffn_width: 256
  level_ffn_width: 64
  micro_patch_width: 4
  dropout: 0.10

loss:
  huber_delta: 1.0
  price_weight: 1.0
  volume_weight: 0.5
  delta_weight: 0.5
  horizon_weight: inverse_sqrt

optimizer:
  kind: adamw
  learning_rate: 0.0003
  weight_decay: 0.01
  warmup_fraction: 0.05
  schedule: cosine
  gradient_clip_norm: 1.0

training:
  max_epochs: 100
  batch_size: 64
  early_stop_patience: 10
  mixed_precision: false
  deterministic_reference: true
~~~

Schema validation rejects unknown fields by default so a misspelled option cannot silently use a
default.

---

## 26. Training run lifecycle

~~~text
Created
  -> DatasetVerified
  -> Initialized
  -> Training
  -> FoldEvaluation
  -> CandidateFrozen
  -> FinalDevelopmentTraining
  -> FinalHoldout
  -> ExportEligible | Rejected
  -> Completed

Training -> Checkpointed -> Training
Any active state -> Cancelled | Failed
~~~

### 26.1 Run identity

The run ID hashes:

- dataset ID;
- canonical training config;
- trainer/model source revision;
- dependency lock hash;
- seed.

Two runs with the same identity may be compared for determinism; they are not silently overwritten.

### 26.2 Optimizer and schedule

V1 uses AdamW, linear warmup for the first 5 percent of optimizer steps, then cosine decay. The
schedule is based on actual optimizer steps after gradient accumulation.

Alternative optimizers or schedules require a new run configuration and cannot be selected after the
outer holdout is opened.

### 26.3 Early stopping

Early stopping uses the predeclared primary validation metric, proposed as mean one-step total-POC MAE
in ticks aggregated by session. Patience is counted in completed validation epochs.

The best checkpoint is chosen from validation only. The final holdout never chooses an epoch.

### 26.4 Checkpoint contents

Checkpoint directory:

~~~text
runs/<run-id>/checkpoints/<step>/
  weights.safetensors
  optimizer-state/
  scheduler.json
  sampler-state.json
  amp-scaler.json        # only when mixed precision is enabled
  rng-state.json
  trainer-state.json
  metrics.json
  hashes.json
~~~

Promoted artifacts never include optimizer state. Weight files use a non-executable tensor format.
Any local optimizer serialization requiring trusted deserialization remains inside the private run
directory and is never loaded from an imported package.
Checkpoint creation writes a sibling staging directory, closes and hashes every file, atomically renames
the directory, then updates the run manifest. Resume ignores partial/unreferenced checkpoints.

### 26.5 Resume

Resume verifies dataset, config, code/dependency, and checkpoint hashes. A mismatch starts a new run or
fails; it never resumes under a changed identity.

### 26.6 Failure handling

- NaN/Inf input: reject batch and fail the run with example IDs.
- NaN/Inf loss or gradient: save diagnostics, stop, do not export.
- Out of memory: stop with measured batch details; manual new config required.
- Training-process interruption: resume only from the last verified checkpoint.
- Empty-valid-target batch: skip with counter; excessive rate fails audit.
- Divergent validation: early stop and mark rejected.

### 26.7 Final candidate training and export identity

Walk-forward folds select the architecture, hyperparameters, and a fixed final optimizer-step rule. The
proposed rule is the median best optimizer step across eligible folds, recorded before opening the outer
holdout. Then:

1. freeze config/source/dependency/dataset/split identities and the final seed;
2. fit one final input/target scaler on all pre-holdout development sessions only;
3. initialize a new random model and train on that development partition for the frozen step count;
4. atomically seal the exact final checkpoint and scaler hashes;
5. evaluate that checkpoint once on the untouched outer holdout without selecting an epoch/threshold;
6. mark that exact evaluated checkpoint/scaler export-eligible only if the offline data/skill/robustness
   gates pass; export then enables ONNX parity, worker, package, and shadow gates required for promotion.

No all-data refit after seeing holdout results inherits the evaluation. A retrain, changed step count,
threshold, scaler policy, or config is a new candidate/model-selection cycle and requires new untouched
evidence plus shadow evaluation.

---

## 27. Baselines

Every fold evaluates the transformer and all applicable baselines on identical examples:

1. zero-change POC relative to the cutoff reference price;
2. last valid mid/POC-style naive reference;
3. regression consensus used by Volume Footprint;
4. existing online RLS FootprintNextBarPredictor;
5. bars-only transformer ablation;
6. bars + L1 ablation;
7. bars + L2 ablation;
8. full bars + L1 + L2 model.

The RLS baseline must be fed only causally available data and genuine labels for a fair comparison.
Its current live feature contract may not match the transformer input contract, so the report states
that distinction.

---

## 28. Evaluation metrics

### 28.1 Price heads

For total, buy, and sell POC at each horizon:

- MAE in ticks;
- median absolute error;
- RMSE in ticks;
- directional hit rate relative to the cutoff reference price;
- signed bias;
- 90th and 99th percentile absolute error;
- per-session error;
- block-bootstrap confidence interval by session.

### 28.2 Volume head

- MAE in log-volume residual;
- median absolute percentage error where stable;
- weighted absolute percentage error;
- signed bias;
- error by volume regime.

### 28.3 Delta head

- MAE of normalized delta;
- delta-sign hit rate;
- balanced accuracy for positive/negative delta;
- signed bias;
- error by aggressor-classification quality.

### 28.4 Robustness slices

Report every metric by:

- session;
- time of day;
- volatility/range regime;
- spread regime;
- L1/L2 missingness bucket;
- event-lag/staleness bucket;
- sequence-gap bucket;
- one-sided-book occurrence;
- source reconnect period;
- horizon.

### 28.5 Operational metrics

- model parameter count;
- ONNX file size;
- isolated worker model-load time;
- worker-only inference p50/p95/p99;
- end-to-end request p50/p95/p99;
- peak worker RSS;
- host RSS change;
- tensor/IPC file size;
- timeout and fallback rate.

Performance values are measured results in an evaluation report, never copied from goals.

---

## 29. Proposed model promotion gates

The exact thresholds must be approved before final holdout evaluation. Proposed V1 policy:

1. all data, leakage, determinism, parity, and package integrity gates pass;
2. one-step total-POC MAE beats the selected simple baseline on a majority of walk-forward folds;
3. block-bootstrap confidence interval for pooled improvement does not indicate a material regression;
4. no horizon has an unexplained severe tail-error regression;
5. volume and delta heads are no worse than their declared baselines beyond approved tolerance;
6. full model adds measurable value over bars-only and at least one microstructure ablation;
7. PyTorch and ONNX predictions match within the declared tolerance;
8. target-machine worker latency and memory budgets pass;
9. no NaN, Inf, stale-result publication, unbounded growth, or process leak occurs;
10. shadow results remain directionally consistent with offline results over the required session count.

If the model fails, it stays a research artifact. Thresholds cannot be weakened after seeing the final
holdout without creating a new model-selection cycle and new holdout.

---

## 30. ONNX export and parity

### 30.1 Export wrapper

The exported module is not the training core alone. It wraps the frozen model with:

1. input finite checks represented in worker validation;
2. input normalization and clipping constants;
3. mask-aware zeroing;
4. the custom transformer;
5. inverse price/volume target normalization;
6. validation of the already bounded delta fraction, with no second tanh;
7. the fixed forecast output ordering.

The graph has fixed batch-one shapes from Section 12 and no dynamic axes. Export runs from an eval-mode
CPU checkpoint in float32 with dropout disabled. The chosen ONNX opset and minimum/maximum supported
ONNX Runtime versions are pinned in the package rather than assumed from the developer machine.

### 30.2 Graph checks

Export fails unless all of these pass:

- ONNX structural checker and shape inference;
- exact allow-listed input/output names, ranks, dtypes, and dimensions;
- allow-listed operator domains/types plus node/initializer count limits;
- no external-data reference, custom operator, training operator, control-flow graph, or uncontrolled
  dynamic dimension;
- parameter count and file-size ceilings;
- no initializer or metadata path outside the package;
- all initializers and graph bytes are covered by the package hash set;
- batch-one CPU execution produces finite output for every golden case.

### 30.3 Parity suite

Parity compares the same raw engineered tensors through:

1. the PyTorch export wrapper;
2. ONNX Runtime in the standalone worker;
3. the managed client encode/decode path.

Golden cases include:

- fully valid synthetic tensors with known values;
- all L2 missing;
- sparse/one-sided depth;
- stale L1 with finalized-bar anchor fallback;
- all-masked rows inside otherwise valid modalities;
- extreme but clipped numeric values;
- real held-out examples selected before export and retained only in the protected evaluation store;
- repeated identical calls for determinism.

Proposed starting tolerances are absolute error at most 1e-4 and relative error at most 1e-3 for each
forecast cell. They are configuration values and must be validated against the actual exporter/runtime;
they are not weakened for one troublesome example. The result stores worst-case error and example ID.
Only generated golden tensors are copied into the package; real-case inputs remain licensed local data
and contribute report/example hashes without their bytes.

---

## 31. Model package and registry

### 31.1 Package contents

A candidate is an immutable directory outside Git:

~~~text
packages/objects/<package-sha256>/
  manifest.json
  model.onnx
  feature-contract.json
  label-contract.json
  normalization.json
  training-config.json
  data-receipt.json
  metrics.json
  parity-report.json
  environment-lock.json
  sha256sums.json
  signature.json
  golden/
    inputs-*.bin
    outputs-*.json
~~~

The package contains no raw market data, credentials, optimizer state, Python pickle, executable script,
or arbitrary plugin. Golden inputs are tiny numeric fixtures reviewed for licensing and privacy.

### 31.2 Manifest

The canonical manifest binds at least:

- package schema version and package hash;
- model kind and run/checkpoint IDs;
- InstrumentId, source, tick size, session calendar, and one-minute timeframe;
- feature, target, normalization, split, dataset, and builder hashes;
- exact tensor names, shapes, dtypes, byte order, and output ordering;
- decision lag, label grace, staleness, grid, patch, and horizon policies;
- model parameter count and ONNX file size;
- opset and supported worker/runtime versions;
- clamp, rounding, and abstention policies;
- evaluation and parity report hashes;
- build source revision, dependency-lock hash, and creation UTC;
- license/retention classification without embedding vendor credentials.

Canonical JSON uses UTF-8, stable property ordering, invariant numeric formatting, and no insignificant
whitespace before hashing. To avoid a recursive self-hash, sha256sums.json covers sorted payload files
but excludes itself, manifest.json, and signature.json; the manifest binds the hash-set digest;
PackageHash is SHA-256 of the canonical manifest with packageHash omitted. signature.json is then
created over that PackageHash. Validation repeats exactly that recipe.

Hashes provide integrity, not publisher authenticity. A promotable package also carries a detached,
domain-separated signature over PackageHash and schema version from a configured trusted signer; the
manifest records signer/key ID and algorithm. Verification uses a pinned public key/certificate trust
store. Unsigned developer packages may run only in an explicit local research mode and cannot enter
shadow, approval, or promotion states.

### 31.3 Validation before load

Import and worker load both validate independently:

1. package path is under the configured registry root;
2. only allow-listed regular files exist;
3. no symlink, reparse-point escape, traversal, device path, or external ONNX data exists;
4. file count and individual/total size limits pass;
5. every digest matches sha256sums.json;
6. the canonical package identity recomputes; trusted signature verifies in any shadow/promotable mode,
   while an explicit unsigned local mode routes only to ResearchOnly;
7. manifest coordinate and graph signature match;
8. runtime/opset compatibility passes;
9. golden inference passes before the package becomes Ready.

A structural/integrity/golden failure quarantines the object with a reason. A valid but unsigned object
in explicit local research mode becomes ResearchOnly rather than Quarantined. Neither state can replace
the active model.

### 31.4 Registry

Do not store ONNX bytes in the existing online-model JSON payload. Use a Pro-private content-addressed
object store plus a small SQLite control database:

~~~text
ModelObject(package_hash, state, coordinate_hash, imported_utc, validation_report)
ModelPointer(coordinate_hash, active_hash, previous_hash, pointer_generation, updated_utc)
ModelTransition(id, coordinate_hash, from_hash, to_hash, reason, actor, utc)
~~~

Pointer changes occur in one transaction. Import is idempotent by content hash. Promotion increments
pointer_generation. Rollback swaps the active and previous verified hashes. An object referenced by any active,
previous, evaluation, or shadow record is not eligible for cleanup.

### 31.5 Lifecycle

~~~text
Imported -> IntegrityValidated -> Validated -> Shadow -> Approved -> Promoted -> Retired
                 |                 |
                 +-> ResearchOnly  +-------------------------------> Quarantined
Any validation state ------------------------------------------------> Quarantined
~~~

Only an explicit operator action promotes or rolls back. Drift or runtime failure may demote to fallback,
but it never auto-promotes another candidate.
ResearchOnly may be used by offline inspect/parity/benchmark commands with an explicit developer flag;
it cannot enter chart shadow/display, Approved, or Promoted.

---

## 32. Isolated inference worker

### 32.1 ADR-0006 compliance

Python and ONNX Runtime remain outside the C# build and WPF process. The production artifact is an
independently built Pro worker distribution, for example a reviewed one-folder executable bundle. The
managed application references no onnxruntime, Python.NET, embedded interpreter, native inference DLL,
or worker project. The worker is a child process reached only through versioned HTTP/JSON on 127.0.0.1.

The worker distribution itself is Authenticode/vendor-signed and its expected publisher/hash is pinned
by the Pro installer/update manifest. A Job Object supplies lifetime/resource containment but is not a
security sandbox. Production acceptance additionally requires a restricted token/AppContainer-style
profile, read-only access only to the selected package object, write access only to the bounded run/log
directory, and outbound-network denial except required loopback. If that confinement cannot be
implemented on a supported host, the residual current-user file/network authority is documented and
only trusted signed packages are permitted; arbitrary imported ONNX is out of scope.

The existing AI/Paper Lab sidecar is not reused. Numeric inference has its own dependency lock, process,
port, health state, and failure boundary.

### 32.2 Launch and handshake

The managed supervisor:

1. creates a unique current-user-only run directory;
2. generates a 256-bit session token and nonce, writes the token to an ACL-restricted one-use file in
   that directory, and keeps its in-memory copy;
3. launches the hidden worker with bind address 127.0.0.1, port 0, registry root, run directory, and
   bounded resource settings;
4. assigns it to a Windows Job Object configured for kill-on-host-exit;
5. waits for one atomically written, size-bounded ready.json handshake;
6. verifies PID, protocol version, nonce, selected loopback port, and executable identity;
7. calls the authenticated health endpoint before loading a model.

The worker never binds 0.0.0.0, accepts remote hosts, reads application credentials, or inherits a
general-purpose writable working directory. It reads then removes the one-use token file. Secrets are
not placed on a command line or in logs.

The managed HTTP handler disables system/environment proxy use, targets the literal 127.0.0.1 address,
does not follow redirects, and never forwards authorization across origins. Worker access logging is
disabled or explicitly scrubs Authorization before any sink.

### 32.3 HTTP protocol

All requests carry the random bearer token, protocol version, request ID, and bounded Content-Length.
Proposed endpoints:

| Method/path | Purpose |
|---|---|
| GET /v1/health | liveness, readiness, protocol, loaded package hash |
| POST /v1/models/load | load one validated registry object by package hash |
| POST /v1/predict | fixed batch-one inference |
| POST /v1/models/unload | dispose the current session |
| POST /v1/shutdown | graceful child shutdown |

The load request accepts packageHash plus runtimeSessionId, runtimeGeneration, and coordinateHash, never an
arbitrary path. The worker resolves the object under its preconfigured registry root, repeats
containment/hash/graph/golden validation plus the configured signature policy, and binds the created
ONNX session to that exact runtime tuple. An unsigned ResearchOnly object requires an explicit worker
research flag and cannot be loaded by the chart-managed worker.

### 32.4 Predict request

The JSON envelope contains:

~~~json
{
  "protocolVersion": 1,
  "requestId": "<uuid>",
  "runtimeSessionId": "<uuid>",
  "decisionId": "<stable-id>",
  "runtimeGeneration": 1,
  "packageHash": "<sha256>",
  "coordinateHash": "<sha256>",
  "featureCutoffUtc": "<utc>",
  "featureContractHash": "<sha256>",
  "tensors": {
    "bars": "<base64 little-endian float32>",
    "bars_mask": "<base64 uint8>",
    "l1": "<base64 little-endian float32>",
    "l1_mask": "<base64 uint8>",
    "l2": "<base64 little-endian float32>",
    "l2_level_mask": "<base64 uint8>",
    "l2_snapshot_mask": "<base64 uint8>",
    "quality": "<base64 little-endian float32>"
  }
}
~~~

Base64 keeps the control seam JSON while avoiding hundreds of thousands of locale-sensitive numeric
tokens. Exact decoded byte lengths are derived from the loaded manifest. The worker rejects an unknown
field, duplicate field, invalid base64, wrong length, non-finite float, non-0/1 mask, stale runtimeGeneration,
violated nested L2 snapshot/level mask, wrong coordinate, or wrong package/contract hash before ONNX
execution.

The response explicitly echoes requestId, runtimeSessionId, runtimeGeneration, decisionId,
coordinateHash, and packageHash, plus 25 ordered finite forecast values, worker queue time, inference
time, and a status. It never returns stack traces or filesystem paths to the chart.

Predict/unload requests must match the loaded runtimeSessionId, runtimeGeneration, coordinateHash, and
packageHash. The worker rejects a stale tuple before inference; the managed client repeats the same
identity check before publication.

### 32.5 Concurrency and resource bounds

V1 worker owns exactly one loaded package and one ONNX session, processes one inference at a time, and
has no internal unbounded queue. The managed client uses a capacity-one latest-wins mailbox. A second
chart with the same coordinate shares the result stream; a different coordinate uses fallback while the
single V1 slot is occupied. Multi-model concurrency is a later measured design, not an implicit process
fan-out.

Hard limits cover request bytes, JSON depth, handshake/log line size, startup/load/inference timeout,
worker memory, thread count, and log retention. ONNX intra/inter-op thread settings are explicit. The
worker performs a fixed warmup before Ready and reports measured load/warmup duration.

### 32.6 Worker state and recovery

~~~text
Stopped -> Starting -> Handshaking -> Loading -> Warming -> Ready
                                      |                    |
                                      v                    v
                                    Failed <---- Degraded/CircuitOpen
Ready -> Unloading -> Stopped
Any state -> Stopping -> Stopped
~~~

On timeout, broken protocol, process exit, native exception, invalid output, or excessive memory, the
client discards the result and falls back. The supervisor may make one bounded restart attempt after
cooldown. Repeated failures open the circuit for the session. A late response can never publish after
runtimeGeneration, decision, coordinate, or session lifetime changes.

---

## 33. Neutral provider seam and Pro session

### 33.1 Public contract shape

The shared Core contract remains model-agnostic. Proposed concepts are:

- IFootprintForecastProvider: creates an optional session for a resolved instrument/source/timeframe;
- IFootprintForecastSession: exposes status and an asynchronous stream of forecast results and owns
  cancellation/disposal;
- FootprintForecastResult: decision identity, cutoff, target interval identities, five numeric forecast
  values per interval, safe provider/model metadata, and a quality/status code;
- NullFootprintForecastProvider: reports unavailable and performs no work.

It contains no tensor names, feature list, ONNX type, architecture name, training command, package path,
or Pro namespace. The target interval open/close timestamps are explicit; consumers do not infer a
horizon from array arrival order.

### 33.2 Dependency injection

Shared shell composition registers Null using an additive/replaceable pattern. Basic and Intermediate
therefore keep current RLS/regression behavior without a Pro binary. Pro composition registers the real
provider after shared shell/footprint registration so the intended final registration wins.

Public Core is a published SDK dependency, so adding the seam is an additive public API/versioning
event and needs full shared Windows closure review. No public assembly references a private project.

### 33.3 Pro session responsibilities

The real provider session:

1. resolves the active package for the exact coordinate;
2. joins ref-counted canonical bars, L1, L2, and genuine trade streams;
3. maintains bounded availability-stamped rings;
4. runs the monotonic interval clock independent of print-driven chart sealing;
5. snapshots inputs by FeatureCutoffUtc;
6. calls the shared Pro NumericMarketTensorBuilder;
7. enqueues a capacity-one request to the worker client;
8. validates and reconstructs chart values;
9. publishes only a current result;
10. records shadow/prediction data and later matches real labels.

The provider ref-counts one underlying runtime per active coordinate. Multiple charts on the same
coordinate share feed joins, decision clock, tensors, and worker results while retaining independent
subscription lifetimes. This is how the single-worker rule avoids duplicate requests.

Proposed starting bounds are 128 finalized bars, 32,768 quotes, 4,096 depth snapshots, 512 pending
decisions, and one pending inference. Bounds also include time windows; counts are not permission to keep
arbitrarily old data. Every overwrite, supersession, missed clock, and expired score increments a metric.

### 33.4 Session lifecycle

~~~text
Disabled -> Resolving -> StartingWorker -> Loading -> Warming -> ShadowReady
                                               |                  |
                                               v                  v
                                            Fallback <-> Degraded/CircuitOpen
ShadowReady -> DisplayReady -> Fallback
Any active state -> Stopping -> Disposed
~~~

ML disabled, unresolved source, coordinate mismatch, absent package, or worker failure yields the Null/
existing baseline path. Enabling does not block the UI while loading. Disposal increments
runtimeGeneration, cancels feed/clock/HTTP work, expires pending scores, and ignores all late callbacks.

---

## 34. Volume Footprint integration

### 34.1 Attachment point

The existing print-driven OnBarSealed path continues to own the online RLS/regression comparison. The
transformer provider does not run inference synchronously from that callback because its fixed-clock
semantics and L1/L2 inputs are different. Instead, VolumeFootprintViewModel creates a provider session
when ML is enabled and the instrument/source/timeframe are resolved, then consumes its asynchronous
forecast stream.

This preserves the canonical ingest and renderer while avoiding a second ad hoc copy of market data in
the UI layer.

### 34.2 UI-thread rule

Feed joins, rings, tensor construction, HTTP encoding, worker load, inference, scoring, and package I/O
run outside the UI thread. A completed result is validated first, then the view model posts only the
small chart projection through the existing UI dispatcher. Render invalidation remains coalesced.

The UI thread must never:

- wait synchronously for worker startup or HTTP;
- load/checksum an ONNX package;
- build the L2 tensor;
- retry a failed prediction;
- perform shadow scoring;
- process unbounded status/log events.

### 34.3 Timestamp mapping

Each of the five forecast points includes TargetOpenTimeUtc and TargetCloseTimeUtc. The view model keeps
the latest valid viewGeneration keyed by target interval and projects it to the chart's forecast model. The
renderer positions ghost bars by those timestamps, not by assuming that five array elements follow the
last non-empty footprint bar.

The five outputs remain compatible with the existing visual meaning:

- Poc;
- BuyPoc;
- SellPoc;
- TotalVolume;
- Delta.

Price outputs are snapped to the package tick size with the same declared rounding policy used for
labels. Volume is non-negative and |Delta| cannot exceed TotalVolume after reconstruction. A forecast
that fails any invariant is discarded as a whole.

### 34.4 View-model lifecycle

The view model, not the window, owns the provider session. Required behavior:

1. ML enabled: create asynchronously if no current session exists;
2. symbol/source/timeframe change: increment viewGeneration, dispose old session, create a new coordinate;
3. ML disabled: dispose session and clear transformer-only visual state;
4. synchronous view-model Dispose: increment viewGeneration, cancel/detach immediately, and start bounded
   background teardown without waiting on the UI thread;
5. late result/status: compare viewGeneration and lifetime token, then discard if stale.

Future implementation must preserve the current user change that removed window-owned disposal; it
must not reintroduce an OnClosed ownership path.

### 34.5 Status and fallback presentation

The chart may show a compact state such as Disabled, Loading, Shadow, Ready, Fallback, or Error plus a
safe reason. It must not display local paths, stack traces, bearer tokens, feature values, or proprietary
model internals. When transformer output is unavailable, existing RLS/regression forecasts continue and
the chart remains usable.

---

## 35. Shadow scoring, drift, and rollout

### 35.1 Prediction record

Every shadow/visible decision stores a bounded Pro-private record:

~~~text
prediction_id, decision_id, feature_cutoff, issued_at
coordinate_hash, package_hash, runtime_generation, view_generation
target interval identities for h=1..5
transformed outputs and reconstructed chart values
input eligibility/quality summary and masks
queue, serialization, inference, and end-to-end latency
fallback/status reason
~~~

Raw L1/L2 tensors are not copied into routine logs. If diagnostic tensor retention is explicitly enabled,
it uses a short-lived protected store and the market-data license policy.

### 35.2 Score record

After each target interval closes plus label grace, the session matches by interval identity:

~~~text
prediction_id, package_hash, target interval
realized target values and per-head masks
transformer and baseline errors
tape completeness/aggressor quality
score time and expiration reason
~~~

Missing or invalid labels expire; they are never scored as zero. A superseded display prediction can
still be scored if it was a valid shadow decision, but records remain tied to their original package,
runtimeGeneration, and viewGeneration.

### 35.3 Baseline alignment

Offline and shadow reports use identical eligible rows. The existing one-step RLS baseline is evaluated
prediction-before-update in chronological order. Regression consensus is frozen to a versioned
configuration. A baseline that cannot produce all five horizons/heads is reported only for its valid
scope; missing baseline values are not counted as wins.

### 35.4 Drift signals

Versioned windows track:

- input robust-z median/MAD and clipping rate by feature group;
- bar/L1/L2 mask and staleness rates;
- carried-forward, approximate-time, gap, drop, crossed, and one-sided rates;
- anchor fallback rate;
- forecast offset, volume, delta, and horizon coherence distributions;
- realized error, bias, tail error, and label coverage after labels mature;
- worker timeout, restart, queue supersession, and latency rates.

Thresholds are learned/selected from training and shadow reference periods before promotion. Runtime
does not update model scalers from these observations.

### 35.5 Automated response

- transient warning: remain Ready and expose status;
- sustained input drift: mark Degraded and suppress transformer display;
- invalid output or schema mismatch: immediate fallback;
- repeated timeout/native failure: open circuit and stop the worker after bounded recovery;
- sustained realized-error regression: demote transformer display and recommend rollback;
- any drift: never auto-train, auto-import, or auto-promote.

### 35.6 Rollout sequence

~~~text
Offline-only
  -> hidden Pro shadow
  -> reviewed shadow report
  -> explicit operator-approved package
  -> opt-in visible forecast
  -> reviewed operational period
  -> optional default-on decision
~~~

Each arrow is a gate, not elapsed time. The previous verified package and the existing baseline remain
available throughout rollout.

---

## 36. Observability and failure recovery

### 36.1 Metrics

Use bounded counters/histograms tagged only by safe coordinate/package identifiers:

- capture rows, gaps, drops, retries, and writer failures by stream;
- audit exclusion reasons and coverage;
- compiler examples, invalid masks, shard bytes, and deterministic hash failures;
- training step/loss/gradient/checkpoint/validation metrics;
- package validation/parity/load results;
- session state, decisions, abstentions, supersessions, label expirations, and fallbacks;
- worker process starts/exits, load/warmup/inference latency, RSS, and circuit state;
- chart publication and stale-result rejection counts.

Metric label cardinality is bounded. Observation IDs, request IDs, and full exception messages belong in
structured diagnostic events, not metric labels.

### 36.2 Recovery matrix

| Failure | Required behavior |
|---|---|
| Capture queue full | Preserve ingest; count exact affected stream/range; degrade or reject capture |
| Capture writer fails | Bounded retry, then protected spool with hard byte/free-space limits; exhaustion fails capture; never claim discarded flush success |
| App exits during capture | Mark unclean; reconcile committed hashes/sequence ranges before sealing |
| Lake .partial/orphan | Ignore or quarantine; adopt only after full hash/schema/range verification |
| Compiler stops | Reuse only immutable hash-verified stage outputs |
| Training stops | Resume only a complete identity-matching checkpoint including RNG/sampler state |
| Package import stops | Active pointer remains untouched; clean unreferenced staging safely |
| Active package corrupt | Try previous verified pointer, otherwise Null/RLS |
| Worker fails to start/load | Stay on fallback; bounded restart/cooldown; no UI block |
| Predict timeout/error | Discard request; count; open circuit after policy threshold |
| Non-finite output | Reject entire forecast; quarantine/restart according to repeat policy |
| Late result | Reject by runtimeSessionId, runtimeGeneration, viewGeneration, decision, and package identity |
| Missing label | Expire pending score and lower coverage; never fabricate an error value |
| Drift threshold | Degrade/demote display; never auto-promote replacement |

### 36.3 Logging

Logs are structured, size/retention bounded, and scrubbed. They may contain package hashes, state
transitions, counts, timings, and stable error codes. They do not contain broker credentials, bearer
tokens, complete tensors, raw order books, arbitrary package metadata, or licensed market rows by
default.

---

## 37. Proposed repository ownership and file map

All paths are proposed until implementation is separately authorized.

### 37.1 Shared public, neutral-only changes

~~~text
public/src/windows/Core/TradingTerminal.Core/
  Domain/
    MarketDataObservationProvenance.cs
    DepthObservation.cs
    MarketDataRecords.cs or additive FinalBarObservation.cs
  Ml/
    IFootprintForecastProvider.cs
    FootprintForecastContracts.cs
    NullFootprintForecastProvider.cs

public/src/windows/Core/TradingTerminal.Core/MarketData/
  IMarketDataHub.cs
  IMarketDataStore.cs

public/src/windows/Pipeline/TradingTerminal.MarketData/
  MarketDataIngestService.cs
  MarketDataHub.cs
  Store/... provenance/source-safe backend changes
  Archive/... provenance and restore verification changes
  Archive/Lake/... neutral manifest/depth export changes

public/src/windows/Charts/TradingTerminal.VolumeFootprint/
  VolumeFootprintViewModel.cs
  VolumeFootprintModels.cs
  VolumeFootprintPanel.xaml.cs
  VolumeFootprintServiceCollectionExtensions.cs
~~~

The exact public type location is confirmed during implementation against the current API surface.
Public changes are limited to generally correct provenance, availability, opaque forecast/session, Null
fallback, explicit target timestamps, and lifecycle behavior. They do not reveal the Pro feature schema.

Shared tests cover contracts, source isolation, store/archive/lake round trips, fixed target timestamp
rendering, Null behavior, cancellation, and Basic/Intermediate composition.

### 37.2 Professional .NET projects

~~~text
src/windows/MachineLearning/
  TradingTerminal.Ml.NumericTransformer/
    Contracts/          private schema and canonical serializers
    Tensor/             NumericMarketTensorBuilder and masks
    Time/               monotonic decision/label clock
    Packages/           package validator and registry

  TradingTerminal.Ml.NumericTransformer.Client/
    Capture/            Pro App capture hosted service and writers
    Runtime/            provider/session, bounded rings, scoring, drift
    Worker/             managed HTTP codec, supervisor, Job Object lifecycle

src/windows/Tools/
  TradingTerminal.Ml.NumericTransformer.Cli/
    Commands/           audit, import, prepare, verify, register, promote, rollback
    Lake/               raw manifest readers and compiler orchestration
~~~

Both libraries are WPF-free. The Client has managed dependencies only and no ONNX/Python/native package.
The CLI does not reference WPF or the production worker distribution.

### 37.3 Professional Python and worker

~~~text
tools/numeric-transformer/
  pyproject.toml
  lockfile
  configs/v1-reference.yaml
  daxalgo_numeric_transformer/
    model/              attention, blocks, depth, temporal, decoder
    training/           trainer, folds, checkpoints, evaluation
    export/             ONNX wrapper, parity, package reports
    worker/             loopback service, validation, ONNX session
    cli.py
  tests/
  packaging/            independent worker build recipe and license inventory
~~~

The existing public Python ML sidecar remains unchanged. Training and worker dependency sets may share
source but the production worker build excludes trainer, CUDA, notebook, and development packages.

### 37.4 Tests and fixtures

~~~text
tests/TradingTerminal.Tests.Pro.Headless/
  NumericTransformer/Tensor/
  NumericTransformer/Capture/
  NumericTransformer/Compiler/
  NumericTransformer/Packages/
  NumericTransformer/WorkerProtocol/
  NumericTransformer/Runtime/

tests/TradingTerminal.Tests.Pro/
  NumericTransformer/Composition/
  NumericTransformer/VolumeFootprintLifecycle/

tests/fixtures/numeric-transformer/
  tiny neutral/generated fixture manifests and golden tensors
~~~

No licensed raw market sample is checked in. Generated fixtures use invented prices/sizes and explicit
licenses/README provenance.

### 37.5 Composition and solution changes

Future implementation updates
src/windows/Shell/TradingTerminal.App/Composition/AppDependencyInjection.cs,
src/windows/Shell/TradingTerminal.App/TradingTerminal.App.csproj, and TradingTerminal.Pro.slnx, then
refreshes generated context because project/source-set graphs changed. Shared public projects/solutions
are updated only for shared neutral Windows changes.

### 37.6 Existing dirty-work hazards

At document creation, the root and public worktrees already contain extensive user changes, including
TradingTerminal.Pro.slnx, the Pro App project, generated context, shared solution/project files, and a
public VolumeFootprint window lifecycle edit. Implementation must
re-read status and targeted diffs immediately before each phase, merge additively, and avoid regenerating
context over unrelated in-flight work. Never restore window-local disposal or reformat unrelated files.

---

## 38. Phased implementation plan and exit gates

### Phase 0 - Freeze research coordinate and policies

Work:

- choose one instrument, source, and one-minute session calendar;
- confirm market-data capture/training/derived-model rights;
- approve decision lag, label grace, source reset, staleness, and empty-interval rules;
- name target Windows hardware and worker packaging method;
- predeclare split, primary metric, promotion/non-inferiority, and operational budgets.

Exit gate: a versioned coordinate/policy record exists and no blocking item in Section 42 is unresolved.

### Phase 1 - Correct shared provenance and optional seam

Work:

- add availability/provenance/depth snapshot identity without breaking existing projections;
- add exact quantity-unit/scale support if the chosen venue is fractional;
- make source-scoped depth round trips deterministic;
- expose neutral persistence/drop telemetry and export manifests;
- add opaque provider/result/session and Null behavior;
- keep current Basic/Intermediate composition and tick-primary bars intact.

Exit gate: focused public contract/store/archive/lake/Null/lifecycle tests pass in both shared Windows
closures; no Pro identifiers appear under public/.

### Phase 2 - Build capture and raw lake

Work:

- add explicit Pro App capture control/hosted service;
- implement per-stream bounded writers, journal, availability partitions, atomic manifests, and recovery;
- run a short synthetic/paper capture first, then a licensed pilot;
- audit coverage, lag, gaps, labels, storage rate, and reconnect behavior.

Exit gate: a sealed capture re-verifies byte-for-byte after restart; zero unexplained drops; the selected
source proves simultaneous required streams and label eligibility.

### Phase 3 - Build deterministic tensor compiler

Work:

- implement canonical private contracts and hash serializers;
- implement monotonic decision/label clocks and NumericMarketTensorBuilder;
- compile immutable raw engineered shards and manifests;
- add future-shift, tie-order, mask, session reset, and .NET/Python reader parity tests.

Exit gate: identical inputs/config/source revision produce identical dataset IDs and tensor bytes; all
causality/leakage tests pass.

### Phase 4 - Implement and train the custom transformer

Work:

- implement primitive custom attention, GEGLU blocks, hierarchical L2, patch pooling, fusion, and direct
  decoder;
- add unit/gradient/mask/determinism/tiny-overfit tests;
- run simple baselines and ablations before expensive tuning;
- execute purged walk-forward training/evaluation with immutable checkpoints.

Exit gate: training is reproducible within declared tolerance, baselines are valid, final holdout policy
is respected, and a candidate either passes promotion gates or is explicitly rejected.

### Phase 5 - Export, package, and isolate serving

Work:

- export fixed-shape wrapper ONNX;
- implement package validator/registry and golden parity;
- build the independently packaged worker and managed supervisor/client;
- validate protocol, containment, authentication, crash recovery, memory, and target-hardware latency.

Exit gate: a quarantined corrupt package cannot load, a valid package passes three-path parity, worker
crashes cannot affect the WPF host, and operational budgets pass.

### Phase 6 - Integrate hidden shadow mode

Work:

- register real Pro provider after shared Null;
- integrate view-model session/status/timestamp projection without UI inference;
- persist bounded prediction/score records and baseline comparison;
- run multiple representative sessions with forced disable/source change/disposal/worker crash tests.

Exit gate: zero stale publications or lifecycle leaks, chart ingest/render remains responsive, and shadow
metrics match offline direction within the predeclared tolerance.

### Phase 7 - Controlled promotion

Work:

- review data, skill, robustness, parity, latency, drift, and shadow reports;
- approve a package and opt-in UI exposure;
- exercise rollback and fallback runbooks;
- monitor the approved operational window.

Exit gate: explicit operator approval, successful rollback drill, no unresolved severe slice regression,
and complete evidence bundle. Default-on is a separate product decision.

### Phase 8 - Optional later work

Only after V1 evidence:

- uncertainty/quantile heads;
- multiple instruments/sources/timeframes;
- multiple concurrent worker models;
- incremental cached temporal state;
- alternate depth/context sizes;
- execution/strategy consumption.

Every item requires a new schema/ADR or explicit scope; none is smuggled into V1.

---

## 39. Verification matrix

### 39.1 Data contracts and capture

| Area | Required checks |
|---|---|
| Provenance | source/instrument/capture/sequence survive hub-store-archive-lake round trips |
| Availability | feature bytes are unchanged when a record's availability moves after cutoff |
| Depth identity | equal timestamps and empty/one-sided snapshots remain distinct |
| Source isolation | a row from another broker is rejected, never silently filtered into an example |
| Backpressure | every forced drop/failure changes counters/state and blocks default promotion |
| Atomicity | crash at every write stage leaves only recoverable partial or verified immutable objects |
| Bar rule | canonical live store remains tick-primary; research bar capture is separate |
| Reconnect | new FeedEpochId/reset semantics are deterministic and journaled |

### 39.2 Tensor builder and compiler

- exact field order, shape, dtype, byte order, and mask golden tests;
- no nearest/centered/backward fill; as-of availability join only;
- session/reconnect/gap reset tests;
- fixed boundary, decision-lag, label-grace, empty-interval, and late-print tests;
- POC/buy/sell tie-breaking parity with declared footprint rules;
- future-shift test: adding any later record leaves earlier tensor bytes unchanged;
- deterministic tie-order test for equal availability/event times;
- duplicate/conflicting observation identity tests;
- scaler train-fold-only and masked-value exclusion tests;
- content hashes stable across repeated compilation and restart;
- Python shard reader returns byte/numeric parity with .NET writer.

### 39.3 Model

- Q/K/V and head reshape math against a small hand-calculated example;
- masked-softmax all-valid, partial, all-masked, and invalid-query cases;
- temporal output at position i has zero gradient with respect to a later token j;
- L2 level attention never crosses SnapshotId;
- patch pooling never reads a later patch/frame;
- missing modality forces exact zero gate and finite output;
- horizon queries receive no targets/teacher forcing;
- output shape/order and delta bound;
- exact parameter count and hard ceiling;
- deterministic seeded forward/backward reference;
- tiny generated dataset overfit test;
- checkpoint/resume matches uninterrupted sampler/optimizer/RNG progression;
- NaN/Inf input, loss, gradient, and output fail safely.

### 39.4 Evaluation and leakage

- chronological session groups only;
- dependency-interval purge and embargo assertions;
- validation selects epochs/config, holdout selects nothing;
- baseline prediction-before-update order;
- identical eligible-example mask across compared models;
- block-bootstrap/session aggregation reproducibility;
- ablation labels and manifests cannot accidentally reference a different dataset/fold;
- report includes all declared slices and coverage denominators.

### 39.5 ONNX and package

- PyTorch/worker/managed-codec golden parity;
- fixed graph signature/opset/initializer and no external/custom operators;
- traversal, symlink/reparse escape, extra file, oversized file/request, bad hash, malformed JSON/base64,
  non-finite tensor, wrong shape, wrong package/coordinate/runtimeGeneration tests;
- atomic import/pointer promotion/rollback under injected interruption;
- corrupt active/previous package behavior;
- package does not contain optimizer/pickle/raw data/credentials;
- license inventory and worker dependency lock are reproducible.

### 39.6 Worker and runtime

- loopback-only bind and bearer-token rejection;
- ready-handshake PID/nonce/protocol/path/size validation;
- Job Object kills child when host exits;
- startup, load, predict, shutdown timeout and cancellation;
- capacity-one supersession and single-session concurrency;
- process crash/native exception/invalid response/circuit-breaker recovery;
- late response rejected after disable, coordinate, runtimeGeneration, viewGeneration, or disposal change;
- repeated enable/disable/source switch leaves no process, subscription, timer, task, or handler leak;
- RSS/log/ring/pending-score bounds under long soak;
- target CPU warmed p50/p95/p99 and end-to-end latency benchmark.

Performance thresholds are benchmark assertions on named hardware, not wall-clock unit tests in a busy
CI environment.

### 39.7 Volume Footprint and composition

- Null provider preserves Basic/Intermediate current behavior;
- Pro real registration wins without public-to-private dependency;
- ML-disabled creates no transformer session/process;
- Loading/failed worker never blocks the UI thread;
- timestamped horizons render in correct wall-clock positions across empty bars;
- current RLS/regression behavior remains available;
- stale transformer result cannot replace a newer viewGeneration;
- view-model owns cancellation/disposal; window does not regain ownership;
- chart closes/symbol changes while a prediction is in flight;
- render invalidation remains coalesced and bounded.

### 39.8 Deterministic end-to-end golden

One generated, non-licensed fixture must traverse the complete chain: stamped quote/depth/bar/trade and
health events -> atomic raw lake/manifest -> deterministic replay/tensor shard -> fixed signed test
package -> loopback worker -> managed reconstruction -> target-timestamp chart projection -> label-grace
realization and shadow score. The test includes an empty interval, a one-sided book, and a late print,
and asserts stable IDs/hashes/numeric outputs/status at every boundary. Component tests do not substitute
for this full-chain golden.

### 39.9 Build and repository checks

Run focused tests first, then only the affected closures:

~~~powershell
dotnet build TradingTerminal.Pro.slnx
dotnet build public/TradingTerminal.Windows.Basic.slnf
dotnet build public/TradingTerminal.Windows.Intermediate.slnf
~~~

Run the Python unit/parity suite from the locked environment and the standalone worker protocol tests.
After project/solution/routed source-set changes, refresh context and run summary/check/deep-check as
required by the repository protocol. Review both root and public diffs for boundary leakage before handoff.

---

## 40. Operational workflow and runbook

### 40.1 Capture

1. Connect the approved broker/source in the Pro terminal.
2. Open the Pro capture control and select the exact audited InstrumentId/source/one-minute coordinate.
3. Confirm local destination, free space, license classification, and quality policy.
4. Start capture; record CaptureSessionId and monitor all streams/counters/lag/gaps.
5. On reconnect, record a new FeedEpochId; on source fallback, stop and create a new capture session.
6. Stop capture; drain writers; verify immutable batch hashes; seal the capture manifest.
7. Copy/backup only by verified immutable object hash.

The terminal never starts capture silently merely because a footprint chart is open.

### 40.2 Audit and compile

Proposed command form:

~~~powershell
dotnet dax-numeric-ml audit --capture <capture-session-id> --policy <policy.json>
dotnet dax-numeric-ml prepare --captures <selection.json> --config <dataset.json>
dotnet dax-numeric-ml verify --dataset <dataset-id>
~~~

Audit Fail stops. Warning requires a recorded acceptance and cannot bypass hard rules such as source
mixing, unprovable bar availability, corrupt hashes, or synthetic labels. Preparation writes a new
immutable dataset; it never mutates a prior one.

### 40.3 Train and evaluate

~~~powershell
python -m daxalgo_numeric_transformer train --config <run.yaml>
python -m daxalgo_numeric_transformer evaluate --run <run-id>
python -m daxalgo_numeric_transformer inspect --run <run-id>
~~~

The operator reviews training/validation curves, coverage, every baseline/ablation, fold variability,
tail/slice regressions, dependency identity, and deterministic-resume evidence before opening the final
holdout. A rejected run remains immutable and cannot be relabeled promoted.

### 40.4 Export and register

~~~powershell
python -m daxalgo_numeric_transformer export --run <run-id>
dotnet dax-numeric-ml register --package <package-directory>
dotnet dax-numeric-ml validate --package <package-hash>
~~~

Registration imports by content hash and leaves active pointers unchanged. Validation includes worker
goldens on target-like hardware. The worker distribution itself is installed by the Pro application
installer/update channel, not copied from a training virtual environment.

### 40.5 Shadow and promote

~~~powershell
dotnet dax-numeric-ml shadow --coordinate <coordinate> --package <package-hash>
dotnet dax-numeric-ml report-shadow --package <package-hash>
dotnet dax-numeric-ml promote --coordinate <coordinate> --package <package-hash>
~~~

These command names are proposed. Promotion requires explicit confirmation and records actor/reason/
evidence hashes. It does not occur from the chart UI as a side effect of selecting a model.

### 40.6 Rollback

1. Disable transformer display or let the circuit move to fallback.
2. Inspect stable error codes, worker state, package validation, and data-quality metrics.
3. Transactionally point the coordinate to previous_hash.
4. restart/load and run golden warmup;
5. confirm fallback/previous package behavior in shadow before display;
6. quarantine the suspect package if integrity/runtime validity failed;
7. preserve diagnostic evidence within retention policy.

Rollback never deletes the suspect object or its evidence automatically.

### 40.7 Retention and cleanup

Raw captures, datasets, runs, packages, predictions, and diagnostics have separate approved retention
policies. Cleanup operates only on explicit roots, verifies resolved containment, and refuses objects
referenced by manifests, active/previous pointers, reports, or legal/license holds. A dry-run inventory
precedes any material deletion.

---

## 41. Security, privacy, licensing, and Git hygiene

### 41.1 Local data boundary

Licensed OHLC, L1, L2, trades, tensors, predictions, and model artifacts remain local unless a separate
approved data-processing agreement explicitly authorizes transfer. Training code must not add telemetry,
cloud experiment tracking, remote notebooks, or model uploads by default.

### 41.2 Credentials and access

- broker credentials/tokens never enter capture rows, manifests, checkpoints, packages, or logs;
- raw/model/registry/run directories are current-user protected;
- the confined worker receives read-only access to one selected registry object plus a short-lived local
  authentication secret and has no general outbound network capability;
- imported packages are untrusted until containment/signature/hash/schema/golden validation completes;
- no pickle or arbitrary code is loaded from a package;
- HTTP binds to 127.0.0.1 only and uses a per-launch high-entropy token;
- worker/log/request size and time limits resist accidental resource exhaustion.

### 41.3 Market-data and model rights

Before capture, confirm the broker/exchange license permits local retention, depth storage, model
training, derived weights, backup, and the intended user/deployment scope. A trained model may still be
a derivative work subject to restrictions. Datasets, weights, screenshots, and proprietary feature/
architecture details are never copied into public/.

### 41.4 Repository exclusions

Git ignore rules must cover at least local environments/caches, raw lake, datasets, runs, checkpoints,
ONNX/package objects, registry databases, worker run directories, and diagnostic tensors. Typical weight
extensions such as .pt, .pth, .ckpt, .safetensors, and .onnx are denied by default. Only small reviewed
generated fixtures/contracts/reports intended for tests may be tracked.

No absolute developer-machine path is written to a checked-in manifest or golden.

---

## 42. Risks and decisions still requiring evidence

### 42.1 Blocking decisions before implementation or capture

| Decision | Why it blocks |
|---|---|
| Instrument and broker/source | Defines coordinate, rights, tick/session rules, and actual stream coverage |
| Market-data license | May prohibit retention, training, derived weights, or distribution |
| Session calendar/time zone | Defines bars, resets, fixed horizons, purge spans, and DST behavior |
| Decision lag and label grace | Changes causality, coverage, and live/offline parity |
| Genuine tape/aggressor policy | Determines whether POC/buy/sell/delta labels are trustworthy |
| Finalized-bar origin | Native and derived availability/volume semantics differ |
| Worker distribution/update method | Must satisfy ADR-0006, licensing, reproducibility, and offline install |
| Package signer and trust-root custody | Required to distinguish approved models from merely intact files |
| Target Windows CPU/memory | Required for measurable serving budgets |
| Raw/model storage and retention | Capture cannot begin safely without capacity and cleanup policy |
| Promotion/non-inferiority thresholds | Must be frozen before final holdout/shadow review |
| Public API design/versioning | Shared Core change affects published SDK and both Windows editions |

### 42.2 Technical risks

| Risk | Mitigation |
|---|---|
| Transformer does not beat simple baselines | baseline-first gates; reject without product pressure |
| L2 gaps/nonstationarity dominate | hard provenance/quality gates, masks, ablations, drift fallback |
| Availability timestamps are only approximate | explicit flag/slices; reject coordinates with unsafe ambiguity |
| Venue/session regime changes | chronological folds, regime slices, shadow/drift, no live scaler update |
| Label sparsity in buy/sell POC | per-head masks/coverage; do not substitute total POC |
| JSON IPC overhead | base64 fixed binary tensors, one request per minute, separate end-to-end benchmark |
| Native worker crashes/leaks | subprocess/Job Object, bounded restart/circuit, no host native dependency |
| Multiple charts demand models | V1 single slot/share same coordinate; fallback for conflicting coordinate |
| Feature code diverges | one .NET builder, canonical contracts, goldens, raw engineered ONNX inputs |
| Holdout overfitting | predeclared gates, immutable reports, new cycle after any threshold/config change |
| Package supply-chain issue | lock/inventory, content hashes, allow-list, no code/pickle, golden load |
| Storage growth | measured pilot rate, per-artifact retention, manifest-aware cleanup |

### 42.3 Architecture values that remain hypotheses

The following are V1 starting points, not truths: 64 bars, 120 seconds, 500 ms grid, top 10 levels,
four-frame patches, model width 128, three temporal layers, five horizons, feature list, loss weights,
25 ms worker target, and the parameter/file ceilings. Change one only through a versioned config/schema
and an evaluation that preserves the untouched holdout policy.

---

## 43. Definition of done

### 43.1 Architecture/documentation milestone

This design milestone is complete when:

- architecture, tensor/target contracts, causality, capture, lake, compiler, training, export, serving,
  chart integration, rollout, recovery, and ownership are internally consistent;
- Pro/public and Windows repository boundaries are explicit;
- unknowns are listed as decisions rather than hidden assumptions;
- the document passes Markdown/link/encoding review and the task record captures checks/risks.

### 43.2 Dataset milestone

- provenance/availability contracts pass round trips;
- a sealed licensed capture and audit exist;
- no unexplained drops/source mixing/synthetic labels;
- deterministic compiler and leakage tests pass;
- dataset/scaler/split identities are immutable and reproducible.

### 43.3 Model milestone

- custom architecture tests and parameter ceiling pass;
- baselines/ablations and walk-forward/final-holdout reports are complete;
- checkpoint/resume and environment identity are reproducible;
- candidate passes predeclared skill/robustness gates or is honestly rejected.

### 43.4 Serving milestone

- fixed ONNX export and three-path parity pass;
- package import/load is secure and atomic;
- worker is isolated, loopback-only, bounded, supervised, and meets target-hardware budgets;
- every error path falls back without affecting ingest/UI;
- rollback is tested.

### 43.5 Product milestone

- shared editions preserve Null/current behavior;
- Pro chart lifecycle, timestamp placement, shadow scoring, and drift work without leaks/stale results;
- shadow report passes approved gates;
- operator explicitly promotes an opt-in package;
- context is synchronized, proportional builds/tests pass, and no Pro detail entered public/.

The project is not done merely because a training loss decreased or an ONNX file loaded.

---

## 44. Glossary

| Term | Meaning |
|---|---|
| Event time | Source-reported market occurrence time |
| Ingest time | First receipt/normalization time inside the adapter |
| Available time | First time an observation was eligible for downstream use; causality authority |
| Feature cutoff | Wall-clock interval boundary used to select model inputs |
| Decision issued | Actual time the delayed scheduler/request ran |
| Label grace | Bounded time after target close allowed for genuine tape arrival |
| Coordinate | Exact model scope: instrument, source, timeframe, schemas, builder |
| Raw lake | Immutable provenance-preserving capture objects before feature construction |
| Dataset ID | Content identity of raw receipts, contracts, split policy, and raw-engineered shards |
| Package hash | Content identity of model graph and all serving/evidence contracts |
| Shadow mode | Live inference/scoring without displaying transformer output |
| Fallback | Existing RLS/regression/Null path used when transformer is unavailable |
| Horizon query | Learned numeric query that directly predicts one future fixed interval |
| Patch pooling | Causal compression of four 500 ms frames into one two-second token |
