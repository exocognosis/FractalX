"""Reference utilities for testing the FractalX proposal."""

from fractalx.dimension import BoxCountingResult, estimate_box_counting_dimension
from fractalx.interference import (
    MultiScaleInterferenceScorer,
    combine_with_transformer_score,
)

__all__ = [
    "BoxCountingResult",
    "MultiScaleInterferenceScorer",
    "combine_with_transformer_score",
    "estimate_box_counting_dimension",
]
