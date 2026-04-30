import numpy as np

from prototype import FractalInterferenceScorer, cosine_mean_scores, run_toy_experiment


def test_dimension_changes_scale_weights():
    low_dim_history = np.column_stack(
        [np.linspace(0.0, 1.0, 80), np.zeros(80)]
    )
    grid = np.linspace(0.0, 1.0, 9)
    xx, yy = np.meshgrid(grid, grid)
    high_dim_history = np.column_stack([xx.ravel(), yy.ravel()])

    low = FractalInterferenceScorer(
        smax=4,
        lambda0=1.0,
        wave_mode="pca",
        random_state=7,
    ).fit(low_dim_history)
    high = FractalInterferenceScorer(
        smax=4,
        lambda0=1.0,
        wave_mode="pca",
        random_state=7,
    ).fit(high_dim_history)

    assert high.intrinsic_dimension_ > low.intrinsic_dimension_
    assert not np.allclose(high.scale_weights_, low.scale_weights_)
    assert high.scale_weights_[-1] > low.scale_weights_[-1]


def test_toy_experiment_exposes_mean_vector_failure_mode():
    result = run_toy_experiment(seed=11)

    assert result["cosine_relevant_margin"] <= 0.05
    assert result["fractal_relevant_margin"] > 0.15
    assert result["fractal_top_label"] == "relevant"


def test_score_matches_candidate_count_and_is_finite():
    history = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
    candidates = np.array([[1.0, 0.05], [0.0, 1.0], [-1.0, -0.05]])

    scorer = FractalInterferenceScorer(
        smax=3,
        lambda0=1.0,
        wave_mode="random",
        n_wave_vectors=6,
        random_state=3,
    ).fit(history, np.ones(history.shape[0]))

    scores = scorer.score(candidates)
    cosine_scores = cosine_mean_scores(history, candidates)

    assert scores.shape == (3,)
    assert cosine_scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(np.isfinite(cosine_scores))
