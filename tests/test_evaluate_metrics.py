import numpy as np

from evaluate import (
    long_tail_recall_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
)


def test_ranking_metrics_match_hand_computed_values():
    scores = np.array([0.2, 0.9, 0.8, 0.1])
    relevant = np.array([False, True, False, True])
    long_tail = np.array([False, False, False, True])

    assert np.isclose(ndcg_at_k(scores, relevant, k=3), 1.0 / (1.0 + 1.0 / np.log2(3)))
    assert np.isclose(mean_reciprocal_rank(scores, relevant), 1.0)
    assert long_tail_recall_at_k(scores, relevant, long_tail, k=3) == 0.0
    assert long_tail_recall_at_k(scores, relevant, long_tail, k=4) == 1.0


def test_long_tail_recall_returns_nan_when_user_has_no_long_tail_positive():
    scores = np.array([0.2, 0.9])
    relevant = np.array([False, True])
    long_tail = np.array([True, False])

    assert np.isnan(long_tail_recall_at_k(scores, relevant, long_tail, k=1))
