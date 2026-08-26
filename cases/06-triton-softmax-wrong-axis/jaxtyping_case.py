"""Case 06 under jaxtyping — `plain_jax.py` with the axis names written into annotations.

Softmax keeps its input's shape, so the boundary annotation is satisfied by both versions:
the corrupted intermediate lives entirely inside the body, between the reduction and the
subtraction, where no annotation reaches.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def softmax_buggy(
    x: Float[Array, "row col"], dim: int, keep_dims: bool = False
) -> Float[Array, "row col"]:
    """As shipped: `keep_dims` reaches the two internal reductions."""
    z = x - x.max(axis=dim, keepdims=keep_dims)
    num = jnp.exp(z)
    den = num.sum(axis=dim, keepdims=keep_dims)
    return num / den


@jaxtyped(typechecker=beartype)
def softmax_fixed(x: Float[Array, "row col"], dim: int) -> Float[Array, "row col"]:
    """After PR #11409: `keep_dims=True` in both reductions, and no parameter."""
    z = x - x.max(axis=dim, keepdims=True)
    num = jnp.exp(z)
    return num / num.sum(axis=dim, keepdims=True)


REPORTED = jnp.array([[0.0, 2.0], [3.0, 1.0]])


# --------------------------------------------------------------------------- tests

def test_the_square_tile_satisfies_the_annotation_and_is_wrong():
    """MISSED. `row` and `col` both bind to 2, and the output really is 2x2."""
    out = softmax_buggy(REPORTED, dim=1)
    assert out.shape == (2, 2)
    assert np.allclose(np.asarray(out).sum(axis=1), [0.3979, 5.4493], atol=1e-4)


def test_the_non_square_tile_is_caught_by_the_broadcast_not_the_annotation():
    """RUN-TIME, but it is jnp raising inside the body, not the type checker at the edge."""
    with pytest.raises(Exception):
        softmax_buggy(jnp.arange(6.0).reshape(2, 3), dim=1)


def test_the_fixed_version_is_accepted_for_both():
    assert softmax_fixed(REPORTED, dim=1).shape == (2, 2)
    assert softmax_fixed(jnp.arange(6.0).reshape(2, 3), dim=1).shape == (2, 3)


def test_dim_is_an_int_so_the_annotation_cannot_say_which_axis_it_names():
    """The reduced axis is chosen by a number that carries no axis identity."""
    assert softmax_buggy(REPORTED, dim=0).shape == (2, 2)


# ------------------------------------------------------- minimal scope, with annotations
#
# Softmax's actual scope is a vector. Annotating that scope removes the defect and, unlike
# the untyped versions, jaxtyping enforces the scope — at run time, on the executed call.

@jaxtyped(typechecker=beartype)
def softmax_vector(v: Float[Array, "n"]) -> Float[Array, "n"]:
    """Softmax at the scope the operation actually has: one vector in, one out."""
    z = v - v.max(axis=0, keepdims=False)
    num = jnp.exp(z)
    return num / num.sum(axis=0, keepdims=False)


def test_at_vector_scope_the_flag_is_harmless():
    """`keepdims=False` drops the axis, and a scalar returning to a vector cannot misalign."""
    assert np.allclose(float(softmax_vector(REPORTED[0]).sum()), 1.0)


def test_a_tile_handed_to_the_vector_function_is_rejected():
    """RUN-TIME detection: `Float[Array, "n"]` is rank 1, so the 2-D tile does not fit."""
    with pytest.raises(Exception):
        softmax_vector(REPORTED)


def test_vmap_composes_with_the_annotation():
    """The tile version is the vector version, lifted — and it is correct."""
    out = jax.vmap(softmax_vector)(REPORTED)
    assert np.allclose(np.asarray(out).sum(axis=1), 1.0)
