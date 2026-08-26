"""Case 04 — horizontal displacement normalised by the height (NumPy).

Original: pytorch/vision#9299, fixed by PR #9300.
"""

import numpy as np

RNG = np.random.default_rng(7)


def displacement_buggy(alpha, size):
    """As shipped. `size` is (height, width); dx is divided by size[0] = height."""
    height, width = size
    dx = RNG.uniform(-1.0, 1.0, size=(height, width))
    dx = dx * alpha[0] / size[0] # accidentally normalised by height
    dy = RNG.uniform(-1.0, 1.0, size=(height, width))
    dy = dy * alpha[1] / size[1] # accidentally normalised by width
    return np.stack([dx, dy], axis=-1)


def displacement_fixed(alpha, size):
    """After PR #9300: horizontal by width, vertical by height."""
    height, width = size
    dx = RNG.uniform(-1.0, 1.0, size=(height, width))
    dx = dx * alpha[0] / size[1]
    dy = RNG.uniform(-1.0, 1.0, size=(height, width))
    dy = dy * alpha[1] / size[0]
    return np.stack([dx, dy], axis=-1)


def _scale(alpha, size, fn):
    """Deterministic magnitude comparison: replace the noise by its maximum, 1.0."""
    if fn is displacement_buggy:
        return alpha[0] / size[0], alpha[1] / size[1]
    return alpha[0] / size[1], alpha[1] / size[0]


# --------------------------------------------------------------------------- tests

def test_shapes_are_identical():
    """Nothing about the output distinguishes the two versions."""
    assert displacement_buggy((50.0, 50.0), (100, 800)).shape == (100, 800, 2)
    assert displacement_fixed((50.0, 50.0), (100, 800)).shape == (100, 800, 2)


def test_the_bug_is_invisible_on_square_images():
    """size[0] == size[1], so the swap has no effect. Every square unit test passes."""
    assert _scale((50.0, 50.0), (256, 256), displacement_buggy) == _scale(
        (50.0, 50.0), (256, 256), displacement_fixed
    )


def test_non_square_horizontal_displacement_is_eight_times_too_large():
    """Erroneous behaviour on a 100x800 image."""
    bx, by = _scale((50.0, 50.0), (100, 800), displacement_buggy)
    fx, fy = _scale((50.0, 50.0), (100, 800), displacement_fixed)
    assert np.isclose(bx / fx, 8.0), f"horizontal off by {bx / fx}x"
    assert np.isclose(by / fy, 1 / 8), f"vertical off by {by / fy}x"


def test_fixed_normalisation_is_isotropic_in_pixels():
    """The correct version moves a pixel by the same *pixel* distance in both directions."""
    height, width = 100, 800
    fx, fy = _scale((50.0, 50.0), (height, width), displacement_fixed)
    assert np.isclose(fx * width, fy * height)
