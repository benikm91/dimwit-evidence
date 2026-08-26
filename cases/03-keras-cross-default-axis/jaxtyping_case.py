"""Case 03 under jaxtyping — the clearest `missed` in the whole dossier."""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def cross_buggy(
    a: Float[Array, "n three"],
    b: Float[Array, "n three"],
) -> Float[Array, "n three"]:
    axis = next(i for i, s in enumerate(a.shape) if s == 3)
    return jnp.cross(a, b, axisa=axis, axisb=axis, axisc=axis)


@jaxtyped(typechecker=beartype)
def cross_fixed(
    a: Float[Array, "n three"],
    b: Float[Array, "n three"],
) -> Float[Array, "n three"]:
    return jnp.cross(a, b, axisa=-1, axisb=-1, axisc=-1)


# --------------------------------------------------------------------------- tests

def test_both_satisfy_the_annotation():
    """MISSED. `n` and `three` both bind to 3, so the axes are interchangeable."""
    a = jnp.arange(9.0).reshape(3, 3)
    b = jnp.arange(9.0)[::-1].reshape(3, 3)
    assert cross_buggy(a, b).shape == (3, 3)
    assert cross_fixed(a, b).shape == (3, 3)


def test_and_still_disagree_numerically():
    a = jnp.arange(9.0).reshape(3, 3)
    b = jnp.arange(9.0)[::-1].reshape(3, 3)
    assert not np.allclose(np.asarray(cross_buggy(a, b)), np.asarray(cross_fixed(a, b)))


def test_naming_the_axis_three_does_not_pin_it_down():
    """Even the deliberately descriptive name `three` carries no identity: it is a size."""
    a = jnp.zeros((3, 3))
    assert cross_buggy(a, a).shape == cross_fixed(a, a).shape
