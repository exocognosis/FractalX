"""FractalX reference prototype.

This file is intentionally self-contained so an ML engineer can skim the
scoring mechanics without following package internals. It implements:

* a high-dimensional intrinsic-dimension estimate,
* a configurable multi-scale phase-superposition scorer,
* dimension-conditioned scale weighting and destructive suppression,
* a tiny toy experiment that exposes a mean-vector cosine failure mode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


WaveMode = Literal["pca", "random"]


def l2_normalize_rows(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return row-wise L2-normalized values with zero rows left at zero."""

    array = np.asarray(values, dtype=float)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)


def cosine_mean_scores(
    history_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    engagement_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Cosine similarity to the user's weighted mean engagement vector."""

    history = _as_2d_finite(history_embeddings, "history_embeddings")
    candidates = _as_2d_finite(candidate_embeddings, "candidate_embeddings")
    if candidates.shape[1] != history.shape[1]:
        raise ValueError("candidate_embeddings must match history dimensionality")

    weights = _engagement_weights(engagement_weights, history.shape[0])
    mean_vector = np.average(history, axis=0, weights=weights)
    mean_norm = np.linalg.norm(mean_vector)
    if mean_norm == 0.0:
        return np.zeros(candidates.shape[0], dtype=float)

    candidate_norms = np.linalg.norm(candidates, axis=1)
    numerator = candidates @ mean_vector
    denominator = np.maximum(candidate_norms * mean_norm, 1e-12)
    return numerator / denominator


def estimate_intrinsic_dimension_twonn(
    points: np.ndarray,
    *,
    sample_size: int = 512,
    random_state: int = 0,
) -> float:
    """Estimate intrinsic dimension using TwoNN.

    Box-counting is conceptually aligned with the README, but it is brittle in
    high-dimensional recommender embeddings unless each user has many thousands
    of engagement points. TwoNN is a more honest default here because it uses
    nearest-neighbor distance ratios and needs fewer samples.
    """

    point_array = _as_2d_finite(points, "points")
    if point_array.shape[0] < 3:
        return 1.0

    rng = np.random.default_rng(random_state)
    if point_array.shape[0] > sample_size:
        chosen = rng.choice(point_array.shape[0], size=sample_size, replace=False)
        point_array = point_array[chosen]

    distances = _pairwise_distances(point_array)
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, kth=1, axis=1)[:, :2]
    r1 = nearest[:, 0]
    r2 = nearest[:, 1]

    valid = (r1 > 1e-12) & np.isfinite(r1) & np.isfinite(r2)
    if np.count_nonzero(valid) < 3:
        return 1.0

    mu = np.sort(r2[valid] / r1[valid])
    mu = np.clip(mu, 1.0 + 1e-12, None)
    n = mu.size

    # TwoNN linearizes log(1 - F(mu)) = -D * log(mu). We trim the final point
    # because its empirical CDF value is exactly one.
    x = np.log(mu[:-1])
    y = -np.log(1.0 - (np.arange(1, n) / n))
    slope = float(np.dot(x, y) / max(np.dot(x, x), 1e-12))

    # Exact grids and equally spaced trajectories create nearest-neighbor ties
    # that push TwoNN upward. A PCA participation-ratio cap is a conservative
    # tie guard: it keeps obvious line-like histories near dimension 1 while
    # preserving higher estimates for genuinely spread-out histories.
    centered = point_array - np.mean(point_array, axis=0, keepdims=True)
    variances = np.linalg.svd(centered, compute_uv=False) ** 2
    if np.sum(variances) <= 1e-12:
        return 1.0
    pca_dimension = float((np.sum(variances) ** 2) / np.sum(variances ** 2))
    return float(np.clip(min(slope, pca_dimension), 0.1, point_array.shape[1]))


@dataclass
class FractalInterferenceScorer:
    """Multi-scale complex-interference scorer for user-item embeddings."""

    smax: int = 4
    lambda0: float | None = None
    wave_mode: WaveMode = "pca"
    n_wave_vectors: int = 8
    dimension_weight: float = 1.0
    destructive_gain: float = 0.75
    phase_smoothing: bool = True
    random_state: int = 0

    def fit(
        self,
        history_embeddings: np.ndarray,
        engagement_weights: np.ndarray | None = None,
    ) -> "FractalInterferenceScorer":
        """Fit the user-specific interference field."""

        history = _as_2d_finite(history_embeddings, "history_embeddings")
        weights = _engagement_weights(engagement_weights, history.shape[0])

        if self.smax < 1:
            raise ValueError("smax must be positive")
        if self.n_wave_vectors < 1:
            raise ValueError("n_wave_vectors must be positive")
        if self.wave_mode not in {"pca", "random"}:
            raise ValueError("wave_mode must be 'pca' or 'random'")
        if self.lambda0 is not None and self.lambda0 <= 0.0:
            raise ValueError("lambda0 must be positive when provided")

        self.history_embeddings_ = np.asarray(history, dtype=float)
        self.engagement_weights_ = weights / np.maximum(np.sum(weights), 1e-12)
        self.intrinsic_dimension_ = estimate_intrinsic_dimension_twonn(
            self.history_embeddings_,
            random_state=self.random_state,
        )
        self.dimension_ratio_ = float(
            np.clip(self.intrinsic_dimension_ / history.shape[1], 0.0, 1.0)
        )
        self.wave_vectors_ = self._build_wave_vectors(self.history_embeddings_)
        self.lambda0_ = (
            float(self.lambda0)
            if self.lambda0 is not None
            else self._default_lambda0(self.history_embeddings_)
        )
        self.scale_weights_ = self._dimension_conditioned_scale_weights()
        self.destructive_multiplier_ = 1.0 + self.destructive_gain * (
            1.0 - self.dimension_ratio_
        )
        return self

    def score(self, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Score candidates by dimension-conditioned multi-scale interference."""

        self._require_fit()
        candidates = _as_2d_finite(candidate_embeddings, "candidate_embeddings")
        if candidates.shape[1] != self.history_embeddings_.shape[1]:
            raise ValueError("candidate_embeddings must match history dimensionality")

        by_scale = self.score_by_scale(candidates)
        scores = by_scale @ self.scale_weights_
        negative = scores < 0.0
        scores[negative] *= self.destructive_multiplier_
        return scores

    def score_by_scale(self, candidate_embeddings: np.ndarray) -> np.ndarray:
        """Return one raw interference score per candidate and scale."""

        self._require_fit()
        candidates = _as_2d_finite(candidate_embeddings, "candidate_embeddings")
        if candidates.shape[1] != self.history_embeddings_.shape[1]:
            raise ValueError("candidate_embeddings must match history dimensionality")

        scores = np.empty((candidates.shape[0], self.smax), dtype=float)
        for scale_index in range(1, self.smax + 1):
            wavelength = self.lambda0_ * (2.0 ** -(scale_index - 1))
            scores[:, scale_index - 1] = self._single_scale_score(
                candidates,
                wavelength=wavelength,
            )
        return scores

    def _single_scale_score(self, candidates: np.ndarray, *, wavelength: float) -> np.ndarray:
        deltas = candidates[:, np.newaxis, :] - self.history_embeddings_[np.newaxis, :, :]
        projections = np.einsum("cnd,kd->cnk", deltas, self.wave_vectors_)
        phases = 2.0 * np.pi * projections / wavelength

        # Average across wave vectors to reduce phase aliasing from any one
        # direction; then superpose weighted engagement history.
        per_history_field = np.mean(np.exp(1j * phases), axis=2)
        if self.phase_smoothing:
            distances = np.linalg.norm(deltas, axis=2)
            envelope = np.exp(-0.5 * (distances / max(wavelength, 1e-12)) ** 2)
            per_history_field *= envelope
        field = per_history_field @ self.engagement_weights_
        return 2.0 * np.real(np.conjugate(field))

    def _dimension_conditioned_scale_weights(self) -> np.ndarray:
        scales = np.arange(1, self.smax + 1, dtype=float)
        base = 2.0 ** -(scales - 1.0)
        fine_scale_position = (scales - 1.0) / max(self.smax - 1.0, 1.0)

        # Higher intrinsic dimension means the user's history occupies a more
        # complex region, so fine scales get more weight. Low-D histories keep a
        # broad, conservative filter.
        fractal_boost = 1.0 + self.dimension_weight * self.dimension_ratio_ * fine_scale_position
        weights = base * fractal_boost
        return weights / np.sum(weights)

    def _build_wave_vectors(self, history: np.ndarray) -> np.ndarray:
        if self.wave_mode == "random":
            rng = np.random.default_rng(self.random_state)
            vectors = rng.normal(size=(self.n_wave_vectors, history.shape[1]))
            return l2_normalize_rows(vectors)

        centered = history - np.mean(history, axis=0, keepdims=True)
        if np.allclose(centered, 0.0):
            return np.eye(history.shape[1])[: self.n_wave_vectors]

        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        vectors = vh[: self.n_wave_vectors]
        if vectors.shape[0] < self.n_wave_vectors:
            rng = np.random.default_rng(self.random_state)
            extra = rng.normal(
                size=(self.n_wave_vectors - vectors.shape[0], history.shape[1])
            )
            vectors = np.vstack([vectors, extra])
        return l2_normalize_rows(vectors)

    def _default_lambda0(self, history: np.ndarray) -> float:
        if history.shape[0] < 2:
            return 1.0
        distances = _pairwise_distances(history)
        finite = distances[np.triu_indices_from(distances, k=1)]
        finite = finite[finite > 1e-12]
        if finite.size == 0:
            return 1.0
        return float(np.clip(np.median(finite) * 2.0, 0.05, 10.0))

    def _require_fit(self) -> None:
        if not hasattr(self, "history_embeddings_"):
            raise ValueError("fit must be called before score")


def run_toy_experiment(seed: int = 0) -> dict[str, float | str]:
    """Run a deterministic two-interest toy experiment.

    The history has two symmetric interests. The weighted mean vector is near
    zero, so cosine-to-mean cannot distinguish relevant cluster candidates from
    off-cluster candidates. The interference field still has constructive peaks
    around both historical clusters.
    """

    rng = np.random.default_rng(seed)
    left = rng.normal(loc=(-1.0, 0.0), scale=0.03, size=(24, 2))
    right = rng.normal(loc=(1.0, 0.0), scale=0.03, size=(24, 2))
    history = np.vstack([left, right])
    weights = np.ones(history.shape[0])

    candidates = np.array(
        [
            [-1.02, 0.02],
            [1.02, -0.01],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    labels = np.array(["relevant", "relevant", "off_cluster", "off_cluster"])

    cosine_scores = cosine_mean_scores(history, candidates, weights)
    fractal_scores = FractalInterferenceScorer(
        smax=4,
        lambda0=2.0,
        wave_mode="pca",
        n_wave_vectors=2,
        random_state=seed,
    ).fit(history, weights).score(candidates)

    relevant = labels == "relevant"
    off_cluster = ~relevant
    cosine_margin = float(np.mean(cosine_scores[relevant]) - np.mean(cosine_scores[off_cluster]))
    fractal_margin = float(np.mean(fractal_scores[relevant]) - np.mean(fractal_scores[off_cluster]))
    top_label = str(labels[int(np.argmax(fractal_scores))])

    return {
        "cosine_relevant_margin": cosine_margin,
        "fractal_relevant_margin": fractal_margin,
        "fractal_top_label": top_label,
    }


def _as_2d_finite(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _engagement_weights(weights: np.ndarray | None, n_rows: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_rows, dtype=float)
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise ValueError("engagement_weights must be a 1D array matching history rows")
    if not np.all(np.isfinite(array)):
        raise ValueError("engagement_weights must contain only finite values")
    if np.any(array < 0.0):
        raise ValueError("engagement_weights must be non-negative")
    if np.sum(array) <= 0.0:
        raise ValueError("engagement_weights must contain at least one positive value")
    return array


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    squared_norms = np.sum(points * points, axis=1, keepdims=True)
    squared = squared_norms + squared_norms.T - 2.0 * (points @ points.T)
    return np.sqrt(np.maximum(squared, 0.0))


if __name__ == "__main__":
    toy = run_toy_experiment()
    print("Toy experiment:")
    for key, value in toy.items():
        print(f"  {key}: {value}")
