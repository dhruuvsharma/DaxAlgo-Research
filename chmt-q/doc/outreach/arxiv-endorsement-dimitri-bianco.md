# arXiv endorsement outreach — Dimitri Bianco

Author: Dhruv Sharma · Project: CHMT-Q (chmt-q workspace) · Drafted: 2026-08-10

Goal: get an arXiv endorsement so a first-time submission can be posted (q-fin.TR / q-fin.CP / cs.LG).

## Before sending

- Register on arXiv and note your arXiv username — you'll need it once he agrees.
- Host the two PDFs somewhere linkable (e.g., Google Drive with view-only link) or attach them, then
  replace `[PDF link]` below. The empirical paper is
  `publication/chmtq-arxiv-manuscript.pdf`; the architecture paper is
  `publication/chmtq-transformer-architecture-paper.pdf`.
- If you prefer email: use the subject line below. If you prefer a YouTube channel DM / X DM: use the
  short variant at the bottom.
- Do not send raw data, weights, checkpoints, or prediction tapes. The PDFs are the only artifact to
  share. Keep all claims research-only and non-promotable.

---

## Primary message (email)

**Subject:** arXiv endorsement request — an honest negative-results paper on order-book Transformers

Hi Dimitri,

I've been following your channel for about two years — your order book and backtesting videos are a
big part of why I got into this space — so you were the first person I thought of for this.

I'm submitting my first paper to arXiv and need an endorsement to post it. It's a negative-results
study: **"When Better Proper Scores Do Not Imply Tradable Edge: A Leakage-Aware Failure Analysis of a
Causal Hierarchical Market Transformer with Quantile Outputs."**

The short version: I built CHMT-Q, a ~2.09M-parameter causal Transformer that consumes bars, level-1
and level-2 order book state, and quality features, and outputs ordered quantiles for five
footprint targets at five one-minute horizons. Trained on ~25.8k examples from OKX BTC-USDT spot
(June 2026), it improved weighted normalized pinball loss by 2.26% over an empirical marginal
baseline on a development-forward partition. That's where the interesting part starts: the better
score did not become signal or money. H5 median MAE barely moved, correlations were ~0.08, predicted
medians were shrunk to ~15% of realized dispersion, every 10–90% interval crossed zero (so the
preregistered policy made zero trades), and a diagnostic median-sign policy lost 68% at 12 bps per
side. The paper audits why: the targets are footprint primitives, not executable returns; gross edge
(0.45 bps per round trip) is dwarfed by costs; heavy window overlap inflates the nominal sample size;
one seed and four forward days limit generalization claims; and several output pathologies are
documented. I also disclose a real integrity incident — a later projection physically read the
nominal holdout, so I retired it and report no confirmatory result from it — which is why the paper
is leakage-aware rather than a "sealed holdout" story.

I think it fits your audience: it's a rigorous demonstration that a favorable-looking metric can fail
in practice, with a full mechanics companion ("CHMT-Q: Causal Hierarchical Market Transformer with
Quantiles") and an immutable artifact ledger (SHA-256 hashes for dataset, protocol, run, tape, and
results). Nothing here is promotable as trading edge — it's a methodology contribution.

What I'd need from you: an arXiv endorsement in q-fin.TR (or q-fin.CP / cs.LG). If you have an arXiv
account with publication history, you can endorse me in about two minutes at
https://arxiv.org/auth/endorse — you'll need my arXiv username, which I'll send once you say you're
willing. If you can't endorse, even a quick "no" or a pointer to someone who might is genuinely
helpful.

Happy to share the PDFs ([PDF link]) or walk through the details on a call if you'd like to look
before deciding. And regardless of the answer — thank you for two years of genuinely good content.

Best,
Dhruv Sharma

---

## Short variant (YouTube channel DM / X DM)

Hi Dimitri — long-time viewer (2 years). I built CHMT-Q, a 2.09M-param order-book Transformer, and
wrote an honest negative-results paper showing that a 2.26% proper-score improvement produced zero
usable edge (every uncertainty interval crossed zero; costs crushed the diagnostic policies). It's
leakage-aware and fully artifact-ledgered — the kind of "favorable metric that fails in practice"
study your audience might like. I need an arXiv endorsement to submit (q-fin.TR / cs.LG). Could you
help, or point me to someone who can? Happy to share the PDFs. Thanks for the content over the years.

---

## Follow-up notes

- If he says yes: send your arXiv username so he can endorse at https://arxiv.org/auth/endorse.
- If he says he has no arXiv publication history: he cannot endorse (arXiv requires the endorser to
  have published in the category). Thank him anyway and consider asking for a referral.
- Never describe the June 25–30 period as sealed or untouched in any correspondence; it is retired.
