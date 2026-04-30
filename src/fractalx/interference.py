"""Multi-scale interference scorer for candidate recommendation items."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class MultiScaleInterferenceScorer:
    """Score candidates against a weighted engagement point cloud.

    The implementation follows the proposal's auxiliary scorer shape:
    engagement history contributes a complex field, each candidate samples that
    field across dyadic scales, and scale scores are combined with weights
    proportional to ``2 ** -s``.
    """

    lambda0: float = 1.0
    max_scale: int = 4
    wave_vectors: np.ndarray | None = None
    scale_weights: np.ndarray | None = None
    candidate_amplitude: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda0) or self.lambda0 <= 0.0:
            raise ValueError("lambda0 must be a positive finite value")
        if int(self.max_scale) != self.max_scale or self.max_scale < 1:
            raise ValueError("max_scale must be a positive integer")
        self.max_scale = int(self.max_scale)
        if not np.isfinite(self.candidate_amplitude) or self.candidate_amplitude < 0.0:
            raise ValueError("candidate_amplitude must be non-negative")
        self.scale_weights = _normalized_scale_weights(self.max_scale, self.scale_weights)

    def fit(
        self,
        history: np.ndarray,
        amplitudes: Iterable[float] | None = None,
    ) -> "MultiScaleInterferenceScorer":
        """Store a user's weighted engagement history."""

        history_array = _as_2d_finite_array(history, name="history")
        amplitude_array = _as_amplitudes(amplitudes, expected_size=history_array.shape[0])

        self.history_ = history_array
        self.amplitudes_ = amplitude_array
        self.wave_vectors_ = _resolve_wave_vectors(self.wave_vectors, history_array)
        return self

    def score_by_scale(self, candidates: np.ndarray) -> np.ndarray:
        """Return raw single-scale interference scores for each candidate."""

        self._require_fitted()
        candidate_array = _as_2d_finite_array(candidates, name="candidates")
        if candidate_array.shape[1] != self.history_.shape[1]:
            raise ValueError("candidates must have the same dimensionality as history")

        scores = np.empty((candidate_array.shape[0], self.max_scale), dtype=float)
        for scale_index in range(1, self.max_scale + 1):
            wavelength = self.lambda0 * (2.0 ** -scale_index)
            scores[:, scale_index - 1] = self._score_single_scale(
                candidate_array,
                wavelength=wavelength,
            )
        return scores

    def score(self, candidates: np.ndarray) -> np.ndarray:
        """Return the weighted multi-scale interference score."""

        return self.score_by_scale(candidates) @ self.scale_weights

    def _score_single_scale(self, candidates: np.ndarray, *, wavelength: float) -> np.ndarray:
        deltas = candidates[:, np.newaxis, :] - self.history_[np.newaxis, :, :]
        projections = np.einsum("cnd,kd->cnk", deltas, self.wave_vectors_)
        phases = 2.0 * np.pi * projections / wavelength
        field_by_history = np.mean(np.exp(1j * phases), axis=2)
        field = np.sum(self.amplitudes_[np.newaxis, :] * field_by_history, axis=1)
        return 2.0 * self.candidate_amplitude * np.real(np.conjugate(field))

    def _require_fitted(self) -> None:
        if not hasattr(self, "history_"):
            raise ValueError("fit must be called before scoring candidates")


def combine_with_transformer_score(
    transformer_scores: np.ndarray,
    interference_scores: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Combine base model scores with the FractalX auxiliary score."""

    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("alpha must be a non-negative finite value")

    transformer_array = np.asarray(transformer_scores, dtype=float)
    interference_array = np.asarray(interference_scores, dtype=float)
    if transformer_array.shape != interference_array.shape:
        raise ValueError("transformer_scores and interference_scores must have the same shape")
    if not np.all(np.isfinite(transformer_array)):
        raise ValueError("transformer_scores must contain only finite values")
    if not np.all(np.isfinite(interference_array)):
        raise ValueError("interference_scores must contain only finite values")

    return transformer_array + alpha * interference_array


def _as_2d_finite_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one row and one dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_amplitudes(
    amplitudes: Iterable[float] | None,
    *,
    expected_size: int,
) -> np.ndarray:
    if amplitudes is None:
        return np.ones(expected_size, dtype=float)

    amplitude_array = np.asarray(list(amplitudes), dtype=float)
    if amplitude_array.ndim != 1 or amplitude_array.shape[0] != expected_size:
        raise ValueError("amplitudes must be a 1D sequence matching history length")
    if not np.all(np.isfinite(amplitude_array)):
        raise ValueError("amplitudes must contain only finite values")
    if not np.all(amplitude_array >= 0.0):
        raise ValueError("amplitudes must be non-negative")
    return amplitude_array


def _normalized_scale_weights(
    max_scale: int,
    scale_weights: Iterable[float] | None,
) -> np.ndarray:
    if scale_weights is None:
        weights = 2.0 ** -np.arange(1, max_scale + 1, dtype=float)
    else:
        weights = np.asarray(list(scale_weights), dtype=float)

    if weights.ndim != 1 or weights.shape[0] != max_scale:
        raise ValueError("scale_weights must be a 1D sequence with max_scale values")
    if not np.all(np.isfinite(weights)):
        raise ValueError("scale_weights must contain only finite values")
    if not np.all(weights >= 0.0):
        raise ValueError("scale_weights must be non-negative")
    if float(np.sum(weights)) == 0.0:
        raise ValueError("scale_weights must contain at least one positive value")
    return weights / np.sum(weights)


def _resolve_wave_vectors(
    wave_vectors: np.ndarray | None,
    history: np.ndarray,
) -> np.ndarray:
    if wave_vectors is None:
        return _principal_wave_vectors(history)

    vectors = _as_2d_finite_array(wave_vectors, name="wave_vectors")
    if vectors.shape[1] != history.shape[1]:
        raise ValueError("wave_vectors must have the same dimensionality as history")
    return _normalize_rows(vectors)


def _principal_wave_vectors(history: np.ndarray) -> np.ndarray:
    centered = history - np.mean(history, axis=0, keepdims=True)
    if np.allclose(centered, 0.0):
        return np.eye(history.shape[1], dtype=float)

    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    active_vectors = vh[singular_values > 1e-12]
    if active_vectors.size == 0:
        return np.eye(history.shape[1], dtype=float)
    return _normalize_rows(active_vectors)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("wave_vectors must not contain zero vectors")
    return vectors / norms[:, np.newaxis]
