"""Case 09 under jaxtyping — `plain_jax.py` with the axis names written into annotations.

A clean win for annotations, at run time: unlike a size, a *rank* cannot coincide by
accident, so this class of defect is reliably caught the moment the call is executed.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

CENTERS = jnp.array([0.0, 0.5, 1.0, 1.5])


@jaxtyped(typechecker=beartype)
def ft_ao_buggy(
    Gv: Float[Array, "gpoint 3"],
    centers: Float[Array, "center"] = CENTERS,
) -> Float[Array, "gpoint center"]:
    """As shipped. The body assumes `Gv` is (N, 3) and never checks that it is."""
    G = jnp.asarray(Gv, dtype=jnp.float32)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


@jaxtyped(typechecker=beartype)
def ft_ao_fixed(
    Gv: Float[Array, "gpoint 3"],
    centers: Float[Array, "center"] = CENTERS,
) -> Float[Array, "gpoint center"]:
    """After PR #3340: one line at the top, restoring the assumption the body makes."""
    G = jnp.asarray(Gv, dtype=jnp.float32).reshape(-1, 3)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


GRID = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


# --------------------------------------------------------------------------- tests

def test_a_rank_one_argument_is_rejected():
    """RUN-TIME detection: `Float[Array, "gpoint 3"]` requires rank 2."""
    with pytest.raises(Exception):
        ft_ao_buggy(GRID[1])


def test_a_proper_grid_behaves_the_same_either_way():
    assert np.allclose(np.asarray(ft_ao_buggy(GRID)), np.asarray(ft_ao_fixed(GRID)))


def test_the_reshaped_call_is_accepted():
    assert ft_ao_fixed(GRID[1].reshape(1, 3)).shape == (1, 4)


def test_rank_is_the_one_thing_annotations_always_pin_down():
    """Unlike sizes, rank cannot coincide by accident, so this class is reliably caught."""
    assert ft_ao_fixed(GRID).shape == (3, 4)
    with pytest.raises(Exception):
        ft_ao_fixed(jnp.zeros((2, 2, 3)))
