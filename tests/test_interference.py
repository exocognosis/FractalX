import numpy as np
import pytest

from fractalx.interference import (
    MultiScaleInterferenceScorer,
    combine_with_transformer_score,
)


def test_interference_scorer_boosts_constructive_and_suppresses_destructive_candidates():
    history = np.array(
        [
            [0.00, 0.00],
            [0.00, 0.02],
            [0.02, 0.00],
        ]
    )
    amplitudes = np.array([3.0, 2.0, 2.0])
    candidates = np.array(
        [
            [0.01, 0.01],
            [0.25, 0.25],
        ]
    )

    scorer = MultiScaleInterferenceScorer(
        lambda0=1.0,
        max_scale=1,
        wave_vectors=np.eye(2),
    ).fit(history, amplitudes)

    scores = scorer.score(candidates)

    assert scores[0] > 0.0
    assert scores[1] < 0.0
    assert scores[0] > scores[1]


def test_score_by_scale_matches_weighted_total_score():
    history = np.array(
        [
            [0.00, 0.00],
            [0.00, 0.04],
            [0.04, 0.00],
        ]
    )
    candidates = np.array(
        [
            [0.01, 0.01],
            [0.18, 0.18],
        ]
    )

    scorer = MultiScaleInterferenceScorer(
        lambda0=1.0,
        max_scale=4,
        wave_vectors=np.eye(2),
    ).fit(history)

    by_scale = scorer.score_by_scale(candidates)
    total = scorer.score(candidates)

    assert by_scale.shape == (2, 4)
    assert np.all(np.isfinite(by_scale))
    assert np.allclose(total, by_scale @ scorer.scale_weights)


def test_hybrid_ranker_adds_scaled_interference_score():
    transformer_scores = np.array([0.3, 0.3])
    interference_scores = np.array([2.0, -1.0])

    final_scores = combine_with_transformer_score(
        transformer_scores,
        interference_scores,
        alpha=0.25,
    )

    assert np.allclose(final_scores, np.array([0.8, 0.05]))


def test_interference_scorer_rejects_invalid_history_and_alpha():
    scorer = MultiScaleInterferenceScorer()

    with pytest.raises(ValueError, match="2D"):
        scorer.fit(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="non-negative"):
        scorer.fit(np.array([[0.0], [1.0]]), amplitudes=np.array([1.0, -1.0]))

    with pytest.raises(ValueError, match="alpha"):
        combine_with_transformer_score(np.array([1.0]), np.array([1.0]), alpha=-0.1)
