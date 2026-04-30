"""Fractal-dimension estimators for user-interest point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BoxCountingResult:
    """Result from a box-counting dimension estimate."""

    dimension: float
    scales: np.ndarray
    counts: np.ndarray
    log_inverse_scales: np.ndarray
    log_counts: np.ndarray
    intercept: float
    r_squared: float


def estimate_box_counting_dimension(
    points: np.ndarray,
    *,
    scales: Iterable[float] | None = None,
) -> BoxCountingResult:
    """Estimate the box-counting dimension of a finite point cloud.

    Points are normalized to the unit hypercube before counting occupied boxes,
    so the estimate is invariant to per-axis affine scaling.
    """

    point_array = _as_2d_finite_array(points, name="points")
    scale_array = _as_positive_scale_array(scales)

    active_points = _normalize_active_dimensions(point_array)
    if active_points.shape[1] == 0:
        counts = np.ones(scale_array.shape, dtype=int)
        logs = np.log(1.0 / scale_array)
        return BoxCountingResult(
            dimension=0.0,
            scales=scale_array,
            counts=counts,
            log_inverse_scales=logs,
            log_counts=np.zeros_like(logs),
            intercept=0.0,
            r_squared=1.0,
        )

    counts = np.array(
        [_count_occupied_boxes(active_points, scale) for scale in scale_array],
        dtype=int,
    )
    log_inverse_scales = np.log(1.0 / scale_array)
    log_counts = np.log(counts)

    if np.all(counts == counts[0]):
        return BoxCountingResult(
            dimension=0.0,
            scales=scale_array,
            counts=counts,
            log_inverse_scales=log_inverse_scales,
            log_counts=log_counts,
            intercept=float(log_counts[0]),
            r_squared=1.0,
        )

    slope, intercept = np.polyfit(log_inverse_scales, log_counts, deg=1)
    fitted = slope * log_inverse_scales + intercept
    residual_sum_squares = float(np.sum((log_counts - fitted) ** 2))
    total_sum_squares = float(np.sum((log_counts - np.mean(log_counts)) ** 2))
    r_squared = 1.0 if total_sum_squares == 0.0 else 1.0 - (
        residual_sum_squares / total_sum_squares
    )

    return BoxCountingResult(
        dimension=max(0.0, float(slope)),
        scales=scale_array,
        counts=counts,
        log_inverse_scales=log_inverse_scales,
        log_counts=log_counts,
        intercept=float(intercept),
        r_squared=float(r_squared),
    )


def _as_2d_finite_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one point and one dimension")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_positive_scale_array(scales: Iterable[float] | None) -> np.ndarray:
    if scales is None:
        scale_array = 2.0 ** -np.arange(1, 9, dtype=float)
    else:
        scale_array = np.asarray(list(scales), dtype=float)

    if scale_array.ndim != 1 or scale_array.size < 2:
        raise ValueError("scales must be a 1D sequence with at least two values")
    if not np.all(np.isfinite(scale_array)):
        raise ValueError("scales must contain only finite values")
    if not np.all(scale_array > 0.0):
        raise ValueError("scales must contain only positive values")
    return scale_array


def _normalize_active_dimensions(points: np.ndarray) -> np.ndarray:
    minimums = np.min(points, axis=0)
    spans = np.ptp(points, axis=0)
    active = spans > 0.0

    if not np.any(active):
        return np.empty((points.shape[0], 0), dtype=float)

    normalized = (points[:, active] - minimums[active]) / spans[active]
    return np.clip(normalized, 0.0, 1.0)


def _count_occupied_boxes(normalized_points: np.ndarray, scale: float) -> int:
    bins_per_axis = max(1, int(np.ceil(1.0 / scale)))
    box_indices = np.floor(normalized_points / scale).astype(np.int64)
    box_indices = np.clip(box_indices, 0, bins_per_axis - 1)
    return int(np.unique(box_indices, axis=0).shape[0])
