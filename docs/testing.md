# Testing FractalX

FractalX includes a lightweight Python reference implementation and pytest suite. The tests are meant to pin down the proposal's core mechanics before larger offline recommender experiments are added.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```

If you already have `numpy` and `pytest` available, `python3 -m pytest` is enough.

## Test Layout

- `tests/test_dimension.py` validates box-counting dimension behavior.
- `tests/test_interference.py` validates multi-scale interference scoring and hybrid ranking behavior.
- `src/fractalx/dimension.py` contains the reference box-counting estimator.
- `src/fractalx/interference.py` contains the reference multi-scale interference scorer.

## What the Current Tests Prove

The current test suite checks that:

- A 1D synthetic line embedded in 2D estimates near dimension 1.
- A 2D synthetic square grid estimates near dimension 2.
- Degenerate histories return dimension 0 instead of producing unstable regressions.
- Invalid arrays, non-finite values, invalid scales, and negative amplitudes are rejected.
- A candidate in phase with a weighted engagement cluster receives a positive score.
- A candidate at an opposite phase receives a negative score.
- `score(candidates)` equals `score_by_scale(candidates) @ scale_weights`.
- `combine_with_transformer_score` implements `R_final = R_trans + alpha * S_total`.

## What the Current Tests Do Not Prove

These are reference and regression tests, not empirical validation of the recommender hypothesis. They do not prove that:

- Real user histories have stable non-integer fractal dimension.
- Interference scoring improves ranking metrics on real datasets.
- Destructive interference correlates with explicit negative feedback.
- The scorer is robust to adversarial embedding manipulation.

Those claims belong in offline dataset experiments and online A/B tests. See `docs/validation-framework.md`.

## Recommended Test Command

Use this command before committing:

```bash
python3 -m pytest
```

For a clean-install check:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
python -m pytest
```
