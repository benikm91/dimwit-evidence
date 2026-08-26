"""Case 08 — reshaping with the pre-transpose shape (NumPy).

Original: pymc-devs/pymc#7178, fixed by PR #7180.
"""

import numpy as np

SAMPLE_DIMS = ("chain", "draw")


def _transpose_to_front(data, dims, sample_dims):
    order = list(sample_dims) + [d for d in dims if d not in sample_dims]
    moved = np.transpose(data, [dims.index(d) for d in order])
    return moved, tuple(order)


def to_point_list_buggy(data, dims, sample_dims=SAMPLE_DIMS):
    """As shipped: `rest` is read from the ORIGINAL array, not the transposed one."""
    moved, _ = _transpose_to_front(data, dims, sample_dims)
    rest = data.shape[len(sample_dims):]          # <- the defect
    return moved.reshape((-1, *rest))


def to_point_list_fixed(data, dims, sample_dims=SAMPLE_DIMS):
    """After PR #7180: transpose first, then read the shape."""
    moved, _ = _transpose_to_front(data, dims, sample_dims)
    rest = moved.shape[len(sample_dims):]
    return moved.reshape((-1, *rest))


def _leading():
    """chain, draw already at the front: the case that always worked."""
    return np.arange(3 * 2 * 5, dtype=float).reshape(3, 2, 5), ("chain", "draw", "team")


def _not_leading():
    """The reporter's case: dims are ("team", "draw", "chain")."""
    return np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3), ("team", "draw", "chain")


# --------------------------------------------------------------------------- tests

def test_it_is_correct_while_the_sample_dims_are_leading():
    data, dims = _leading()
    assert to_point_list_buggy(data, dims).shape == to_point_list_fixed(data, dims).shape == (6, 5)
    assert np.allclose(to_point_list_buggy(data, dims), to_point_list_fixed(data, dims))


def test_it_silently_reshapes_into_the_wrong_rectangle():
    """Erroneous behaviour: 30 elements laid out as (10, 3) instead of (6, 5)."""
    data, dims = _not_leading()
    assert to_point_list_buggy(data, dims).shape == (10, 3)
    assert to_point_list_fixed(data, dims).shape == (6, 5)


def test_numpy_raises_nothing_because_the_element_count_matches():
    """reshape only checks the product. 6*5 == 10*3, so the defect is invisible to it."""
    data, dims = _not_leading()
    assert to_point_list_buggy(data, dims).size == data.size


def test_the_elements_are_identical_and_the_row_boundaries_are_not():
    """The sharpest statement of the defect: same buffer, wrong row length.

    A caller that treats one row as one posterior draw is reading values cut at the wrong
    boundary — five teams per draw became three.
    """
    data, dims = _not_leading()
    buggy = to_point_list_buggy(data, dims)
    fixed = to_point_list_fixed(data, dims)
    assert np.allclose(buggy.ravel(), fixed.ravel())    # identical memory
    assert buggy.shape[1] != fixed.shape[1]             # different rows
    assert np.allclose(buggy[0], fixed[0][:3])          # row 0 was truncated mid-draw
