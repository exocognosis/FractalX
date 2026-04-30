# Validation Framework

This document turns the FractalX proposal into a staged testing plan. The included pytest suite validates mechanics; this framework describes the next level of empirical validation.

## Stage 1: Unit and Property Tests

Status: implemented as the initial pytest suite.

Goals:

- Validate box-counting dimension estimates on synthetic point clouds.
- Validate deterministic multi-scale interference scoring.
- Validate hybrid score composition.
- Keep input validation strict enough to fail early on bad experiment data.

Useful future additions:

- Property-based tests for scale monotonicity and finite outputs.
- Regression fixtures for known scorer outputs.
- Tests for random Fourier feature or locality-sensitive hashing approximations.

## Stage 2: Offline Synthetic Benchmarks

Goal: verify that the scorer behaves sensibly before using public recommender datasets.

Recommended synthetic cases:

- Clustered interests with unrelated distractor clusters.
- Hierarchical point clouds with nested sub-interests.
- Histories with one noisy engagement injected into an otherwise stable cluster.
- Cold-start histories with fewer than 20 points.
- Power-user histories with more than 1000 points.

Metrics:

- Top-k recovery of held-out in-cluster candidates.
- Suppression rate for distractor candidates.
- Sensitivity to `S_max`, `lambda0`, wave-vector construction, and weighting scheme.

## Stage 3: Public Dataset Experiments

Recommended datasets:

- MovieLens-25M.
- Amazon Reviews.
- Yelp Open Dataset.

Experiment outline:

1. Train or import item embeddings.
2. Split each user's interaction history into observed history and held-out relevance labels.
3. Build the FractalX user field from observed history.
4. Score held-out candidates with:
   - cosine similarity baseline,
   - transformer or learned scorer baseline when available,
   - non-fractal multi-scale kernel baseline,
   - FractalX interference scorer,
   - hybrid baseline plus FractalX scorer.
5. Compare NDCG@k, MRR, precision@k, recall@k, and long-tail recall.

Minimum ablations:

- `S_max in {1, 2, 4, 8}`.
- `lambda0` grid search.
- PCA wave vectors versus random wave vectors.
- dyadic weights versus uniform weights.
- box-counting dimension regularization enabled versus disabled.

## Stage 4: Negative Feedback Correlation

Goal: test whether destructive interference corresponds to explicit rejection.

Labels:

- "Not interested" events.
- Hide or mute events.
- Low dwell time after impression.
- Explicit downvotes where available.

Evaluation:

- Correlation between negative interference score and negative feedback.
- Precision/recall for identifying rejected candidates.
- Calibration curves across score buckets.

## Stage 5: Production Readiness Checks

Before any production ranking use, test:

- Runtime and memory cost per user and candidate batch.
- Incremental updates when new engagements arrive.
- Approximation quality for random Fourier features or LSH.
- Stability under embedding model upgrades.
- Diversity and exploration safeguards to avoid narrowing the feed.
- Adversarial robustness against embedding-targeted content.

## Acceptance Bar

FractalX should be considered empirically promising only if it improves long-tail recall or negative-feedback suppression at fixed precision against strong baselines, without materially reducing diversity or increasing latency beyond the platform budget.
