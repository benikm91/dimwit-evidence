"""Case 08 in JAX — the same program as `plain.py`."""

import jax
import jax.numpy as jnp
import numpy as np

SAMPLE_DIMS = ("chain", "draw")


def _transpose_to_front(data, dims, sample_dims):
    order = list(sample_dims) + [d for d in dims if d not in sample_dims]
    moved = jnp.transpose(data, [dims.index(d) for d in order])
    return moved, tuple(order)


def to_point_list_buggy(data, dims, sample_dims=SAMPLE_DIMS):
    """As shipped: `rest` is read from the ORIGINAL array, not the transposed one."""
    moved, _ = _transpose_to_front(data, dims, sample_dims)
    rest = data.shape[len(sample_dims):]          # <- the defect
    return jnp.reshape(moved, (-1, *rest))


def to_point_list_fixed(data, dims, sample_dims=SAMPLE_DIMS):
    """After PR #7180: transpose first, then read the shape."""
    moved, _ = _transpose_to_front(data, dims, sample_dims)
    rest = moved.shape[len(sample_dims):]
    return jnp.reshape(moved, (-1, *rest))


def _leading():
    """chain, draw already at the front: the case that always worked."""
    return jnp.arange(3 * 2 * 5, dtype=jnp.float32).reshape(3, 2, 5), ("chain", "draw", "team")


def _not_leading():
    """The reporter's case: dims are ("team", "draw", "chain")."""
    return jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3), ("team", "draw", "chain")


# --------------------------------------------------------------------------- tests

def test_it_is_correct_while_the_sample_dims_are_leading():
    data, dims = _leading()
    assert to_point_list_buggy(data, dims).shape == to_point_list_fixed(data, dims).shape == (6, 5)
    assert np.allclose(np.asarray(to_point_list_buggy(data, dims)), np.asarray(to_point_list_fixed(data, dims)))


def test_it_silently_reshapes_into_the_wrong_rectangle():
    """Erroneous behaviour: 30 elements laid out as (10, 3) instead of (6, 5)."""
    data, dims = _not_leading()
    assert to_point_list_buggy(data, dims).shape == (10, 3)
    assert to_point_list_fixed(data, dims).shape == (6, 5)


def test_reshape_raises_nothing_because_the_element_count_matches():
    """reshape only checks the product. 6*5 == 10*3, so the defect is invisible to it."""
    data, dims = _not_leading()
    assert to_point_list_buggy(data, dims).size == data.size
    with np.testing.assert_raises(TypeError):
        jnp.reshape(data, (7, 5))


def test_the_elements_are_identical_and_the_row_boundaries_are_not():
    """Same buffer, wrong row length: five teams per draw became three."""
    data, dims = _not_leading()
    buggy = np.asarray(to_point_list_buggy(data, dims))
    fixed = np.asarray(to_point_list_fixed(data, dims))
    assert np.allclose(buggy.ravel(), fixed.ravel())
    assert buggy.shape[1] != fixed.shape[1]
    assert np.allclose(buggy[0], fixed[0][:3])


def test_jit_compiles_the_buggy_version():
    """JAX-specific: static shapes throughout, so XLA has no reason to object."""
    data, dims = _not_leading()
    assert jax.jit(lambda d: to_point_list_buggy(d, dims))(data).shape == (10, 3)
