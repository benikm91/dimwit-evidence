"""Case 03 in JAX."""

import jax.numpy as jnp
import numpy as np


def cross_buggy(a, b):
    """Reproduces the torch default: first axis of size 3."""
    axis = next(i for i, s in enumerate(a.shape) if s == 3)
    return jnp.cross(a, b, axisa=axis, axisb=axis, axisc=axis)


def cross_fixed(a, b):
    return jnp.cross(a, b, axisa=-1, axisb=-1, axisc=-1)


# --------------------------------------------------------------------------- tests

def test_jax_accepts_both_and_returns_the_same_shape():
    a = jnp.arange(9.0).reshape(3, 3)
    b = jnp.arange(9.0)[::-1].reshape(3, 3)
    assert cross_buggy(a, b).shape == cross_fixed(a, b).shape == (3, 3)


def test_results_differ():
    a = jnp.arange(9.0).reshape(3, 3)
    b = jnp.arange(9.0)[::-1].reshape(3, 3)
    assert not np.allclose(np.asarray(cross_buggy(a, b)), np.asarray(cross_fixed(a, b)))


def test_jnp_cross_itself_requires_a_size_three_axis_only():
    """jnp.cross validates the extent (3) and nothing about which axis it is."""
    a = jnp.zeros((3, 3))
    assert jnp.cross(a, a, axisa=0, axisb=0, axisc=0).shape == (3, 3)
    assert jnp.cross(a, a, axisa=1, axisb=1, axisc=1).shape == (3, 3)
