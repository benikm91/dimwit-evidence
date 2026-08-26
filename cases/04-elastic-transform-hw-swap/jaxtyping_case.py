"""Case 04 under jaxtyping — `plain_jax.py` with as much typing as jaxtyping can express.

jaxtyping names axes only inside an *array* annotation, as in `Float[Array, "height width"]`;
there is no type for a shape tuple. The defect lives entirely in the two numbers of `size`,
and the most that can be said about them is `tuple[int, int]` — under which the two entries
have the same type and swapping them is well typed. `displacement` takes no array argument
at all, so the return annotation is the only thing jaxtyping gets to check here. This file
is therefore plain_jax.py plus one annotation, which is the honest result rather than a
weakness of how it was written.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

KEY = jax.random.key(7)


@jaxtyped(typechecker=beartype)
def displacement_buggy(
    alpha: tuple[float, float],
    size: tuple[int, int],
) -> Float[Array, "height width 2"]:
    """As shipped. `size` is (height, width); dx is divided by size[0] = height."""
    height, width = size
    dx = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dx = dx * alpha[0] / size[0]  # accidentally normalised by height
    dy = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dy = dy * alpha[1] / size[1]  # accidentally normalised by width
    return jnp.stack([dx, dy], axis=-1)


@jaxtyped(typechecker=beartype)
def displacement_fixed(
    alpha: tuple[float, float],
    size: tuple[int, int],
) -> Float[Array, "height width 2"]:
    """After PR #9300: horizontal by width, vertical by height."""
    height, width = size
    dx = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dx = dx * alpha[0] / size[1]
    dy = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dy = dy * alpha[1] / size[0]
    return jnp.stack([dx, dy], axis=-1)


def _scale(alpha, size, fn):
    """Deterministic magnitude comparison: replace the noise by its maximum, 1.0."""
    if fn is displacement_buggy:
        return alpha[0] / size[0], alpha[1] / size[1]
    return alpha[0] / size[1], alpha[1] / size[0]


# --------------------------------------------------------------------------- tests

def test_the_annotation_is_satisfied_by_the_buggy_version():
    """MISSED. The returned array has exactly the shape it claims; only the divisors swap."""
    assert displacement_buggy((50.0, 50.0), (100, 800)).shape == (100, 800, 2)
    assert displacement_fixed((50.0, 50.0), (100, 800)).shape == (100, 800, 2)


def test_the_size_tuple_carries_no_axis_identity():
    """`tuple[int, int]` is one type twice over, so a caller may pass (width, height)."""
    assert displacement_buggy((50.0, 50.0), (800, 100)).shape == (800, 100, 2)


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


def test_results_differ():
    a = displacement_buggy((50.0, 50.0), (100, 800))
    c = displacement_fixed((50.0, 50.0), (100, 800))
    assert not np.allclose(np.asarray(a), np.asarray(c))
