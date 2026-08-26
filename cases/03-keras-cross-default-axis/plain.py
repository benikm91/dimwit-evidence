"""Case 03 — cross product taken along the wrong axis (NumPy).

Original: keras-team/keras#23219.
"""

import numpy as np


def cross_fixed(a, b):
    """NumPy / JAX / TensorFlow semantics: the last axis holds the 3 components."""
    return np.cross(a, b, axisa=-1, axisb=-1, axisc=-1)


def cross_buggy(a, b):
    """`torch.cross(dim=None)` semantics: whichever axis is first found to have size 3."""
    axis = next(i for i, s in enumerate(a.shape) if s == 3)
    return np.cross(a, b, axisa=axis, axisb=axis, axisc=axis)


def _square_example():
    a = np.arange(9.0).reshape(3, 3)
    b = np.arange(9.0)[::-1].reshape(3, 3)
    return a, b


# --------------------------------------------------------------------------- tests

def test_shapes_are_identical_so_nothing_can_flag_it():
    a, b = _square_example()
    assert cross_buggy(a, b).shape == cross_fixed(a, b).shape == (3, 3)


def test_the_two_conventions_disagree_on_a_3x3():
    """Erroneous behaviour: the torch backend returned different numbers from every other."""
    a, b = _square_example()
    assert not np.allclose(cross_buggy(a, b), cross_fixed(a, b))


def test_the_regression_case_from_the_fix():
    """The (2, 3, 3) case the PR added a test for: first size-3 axis is 1, not 2."""
    a = np.arange(2 * 3 * 3).reshape(2, 3, 3).astype("float64")
    b = np.arange(2 * 3 * 3)[::-1].reshape(2, 3, 3).astype("float64")
    assert not np.allclose(cross_buggy(a, b), cross_fixed(a, b))


def test_the_bug_is_invisible_until_two_axes_collide():
    """With only one length-3 axis both conventions agree — which is why it shipped."""
    a = np.arange(2 * 4 * 3).reshape(2, 4, 3).astype("float64")
    b = np.arange(2 * 4 * 3)[::-1].reshape(2, 4, 3).astype("float64")
    assert np.allclose(cross_buggy(a, b), cross_fixed(a, b))
