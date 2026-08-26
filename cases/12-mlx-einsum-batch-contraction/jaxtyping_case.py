"""Case 12 under jaxtyping."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def batched_matmul_buggy(
    a: Float[Array, "batch row inner"],
    b: Float[Array, "inner col"],
) -> Float[Array, "batch row col"]:
    return jnp.tensordot(a, b, axes=([1], [0]))


@jaxtyped(typechecker=beartype)
def batched_matmul_fixed(
    a: Float[Array, "batch row inner"],
    b: Float[Array, "inner col"],
) -> Float[Array, "batch row col"]:
    return jnp.tensordot(a, b, axes=([2], [0]))


# --------------------------------------------------------------------------- tests

def test_the_square_case_is_missed():
    """MISSED. batch == row == inner == 3, so every name binds and the shapes agree."""
    a = jnp.zeros((3, 3, 3))
    b = jnp.zeros((3, 4))
    assert batched_matmul_buggy(a, b).shape == (3, 3, 4)


def test_the_rectangular_case_is_rejected():
    """RUN-TIME detection once `row` and `inner` differ."""
    a = jnp.zeros((2, 5, 3))
    b = jnp.zeros((3, 4))
    with pytest.raises(Exception):
        batched_matmul_buggy(a, b)


def test_the_fixed_version_is_always_accepted():
    assert batched_matmul_fixed(jnp.zeros((2, 5, 3)), jnp.zeros((3, 4))).shape == (2, 5, 4)
    assert batched_matmul_fixed(jnp.zeros((3, 3, 3)), jnp.zeros((3, 4))).shape == (3, 3, 4)
