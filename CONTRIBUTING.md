# Contributing

FractalX is open for discussion, critique, implementation, and validation.

Useful contributions include:

- Mathematical corrections or simplifications.
- References to adjacent work in recommendation systems, spectral methods, fractal analysis, or embedding geometry.
- Offline evaluation plans and reproducible benchmark scripts.
- Implementations of the scorer as a lightweight auxiliary ranking module.
- Failure cases, negative results, and falsification attempts.

## Suggested Discussion Format

When proposing a change, include:

1. What claim or mechanism the change affects.
2. Why the current version is incomplete, incorrect, or underspecified.
3. How the proposed change could be validated.

## Validation Standard

The core hypotheses should be treated as falsifiable claims. Empirical improvements should be compared against at least:

- Cosine-similarity ranking.
- A transformer scorer baseline.
- A non-fractal multi-scale kernel baseline.

Metrics should include NDCG@k, MRR, precision/recall at fixed thresholds, and long-tail recall.
