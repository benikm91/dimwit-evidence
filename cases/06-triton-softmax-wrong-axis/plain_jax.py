"""Case 06 in JAX — the same program as `plain.py`.

`jnp` follows the same left-padding broadcast rule, so the defect transfers unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np


def softmax_buggy(x, dim, keep_dims=False):
    """As shipped: `keep_dims` reaches the two internal reductions."""
    z = x - x.max(axis=dim, keepdims=keep_dims)
    num = jnp.exp(z)
    den = num.sum(axis=dim, keepdims=keep_dims)
    return num / den


def softmax_fixed(x, dim):
    """After PR #11409: `keep_dims=True` in both reductions, and no parameter."""
    z = x - x.max(axis=dim, keepdims=True)
    num = jnp.exp(z)
    return num / num.sum(axis=dim, keepdims=True)


REPORTED = jnp.array([[0.0, 2.0], [3.0, 1.0]])   # the 2x2 tile from the issue


# --------------------------------------------------------------------------- tests

def test_the_reported_output_is_reproduced_exactly():
    """The numbers in the issue: row sums of 0.3979 and 5.4493 instead of 1 and 1."""
    out = np.asarray(softmax_buggy(REPORTED, dim=1))
    assert np.allclose(out, [[0.2689, 0.1289], [5.4018, 0.0474]], atol=1e-4)
    assert np.allclose(out.sum(axis=1), [0.3979, 5.4493], atol=1e-4)


def test_the_fixed_version_returns_distributions():
    assert np.allclose(np.asarray(softmax_fixed(REPORTED, dim=1)).sum(axis=1), 1.0)


def test_the_shape_is_unchanged_so_nothing_downstream_can_flag_it():
    assert softmax_buggy(REPORTED, dim=1).shape == softmax_fixed(REPORTED, dim=1).shape == (2, 2)


def test_the_default_axis_is_accidentally_correct():
    """A length-C vector left-pads onto the C axis, which is the axis it came from."""
    assert np.allclose(
        np.asarray(softmax_buggy(REPORTED, dim=0)), np.asarray(softmax_fixed(REPORTED, dim=0))
    )


def test_a_non_square_tile_raises_instead_of_lying():
    """Only square tiles are silent. The same code on (2, 3) cannot broadcast at all."""
    rect = jnp.arange(6.0).reshape(2, 3)
    with np.testing.assert_raises(Exception):
        softmax_buggy(rect, dim=1)
    assert np.allclose(np.asarray(softmax_fixed(rect, dim=1)).sum(axis=1), 1.0)


def test_jit_compiles_the_square_case_without_complaint():
    """JAX-specific: the silent path is statically well-shaped, so tracing proves nothing."""
    fn = jax.jit(lambda x: softmax_buggy(x, dim=1))
    assert fn(REPORTED).shape == (2, 2)


# ------------------------------------------------- minimal scope, without a type system
#
# Defining softmax on a vector removes the defect — but nothing enforces the scope.

def softmax_vector(v):
    """Softmax at the scope the operation actually has: one vector in, one out."""
    z = v - v.max(axis=0, keepdims=False)
    num = jnp.exp(z)
    return num / num.sum(axis=0, keepdims=False)


def test_at_vector_scope_the_flag_is_harmless():
    """`keepdims=False` drops the axis, and a scalar returning to a vector cannot misalign."""
    assert np.allclose(float(softmax_vector(REPORTED[0]).sum()), 1.0)


def test_but_nothing_stops_a_tile_going_into_the_vector_function():
    """MISSED. The scope is a docstring: a 2-D argument is accepted and silently wrong."""
    out = jnp.asarray(softmax_vector(REPORTED))
    assert out.shape == (2, 2)
    assert not np.allclose(np.asarray(out).sum(axis=1), 1.0)
