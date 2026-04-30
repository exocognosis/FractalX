import numpy as np
import pytest

from fractalx.dimension import estimate_box_counting_dimension


def test_box_counting_dimension_separates_line_from_area():
    scales = 2.0 ** -np.arange(2, 7)

    line_t = np.linspace(0.0, 1.0, 2048)
    line = np.column_stack([line_t, line_t])

    grid_axis = np.linspace(0.0, 1.0, 64)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    square = np.column_stack([xx.ravel(), yy.ravel()])

    line_result = estimate_box_counting_dimension(line, scales=scales)
    square_result = estimate_box_counting_dimension(square, scales=scales)

    assert 0.85 <= line_result.dimension <= 1.15
    assert 1.75 <= square_result.dimension <= 2.15
    assert square_result.dimension > line_result.dimension + 0.65
    assert line_result.r_squared > 0.98
    assert square_result.r_squared > 0.98


def test_box_counting_dimension_handles_degenerate_histories():
    result = estimate_box_counting_dimension(np.ones((12, 3)))

    assert result.dimension == 0.0
    assert result.r_squared == 1.0
    assert np.all(result.counts == 1)


def test_box_counting_dimension_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="2D"):
        estimate_box_counting_dimension(np.array([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="finite"):
        estimate_box_counting_dimension(np.array([[0.0, np.nan]]))

    with pytest.raises(ValueError, match="positive"):
        estimate_box_counting_dimension(np.array([[0.0], [1.0]]), scales=[0.5, 0.0])
