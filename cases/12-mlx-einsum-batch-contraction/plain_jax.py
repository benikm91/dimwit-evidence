"""Case 12 in JAX — the same program as `plain.py`."""

import jax
import jax.numpy as jnp
import numpy as np

RNG = np.random.default_rng(12)


def batched_matmul_buggy(a, b):
    """Contracts `a`'s row axis instead of its inner axis."""
    return jnp.tensordot(a, b, axes=([1], [0]))


def batched_matmul_fixed(a, b):
    """Contracts the inner axis, which is what `bij,jk->bik` means."""
    return jnp.tensordot(a, b, axes=([2], [0]))


def _square_case():
    """batch == row == inner == 3: both contractions are legal."""
    return (
        jnp.asarray(RNG.normal(size=(3, 3, 3)), dtype=jnp.float32),
        jnp.asarray(RNG.normal(size=(3, 4)), dtype=jnp.float32),
    )


def _rectangular_case():
    return (
        jnp.asarray(RNG.normal(size=(2, 5, 3)), dtype=jnp.float32),
        jnp.asarray(RNG.normal(size=(3, 4)), dtype=jnp.float32),
    )


# --------------------------------------------------------------------------- tests

def test_both_contractions_are_legal_when_the_axes_coincide():
    a, b = _square_case()
    assert batched_matmul_buggy(a, b).shape == batched_matmul_fixed(a, b).shape == (3, 3, 4)


def test_and_they_give_different_answers():
    """Erroneous behaviour: the batch axis has been contracted away and rebuilt elsewhere."""
    a, b = _square_case()
    assert not np.allclose(
        np.asarray(batched_matmul_buggy(a, b)), np.asarray(batched_matmul_fixed(a, b))
    )


def test_the_fixed_version_agrees_with_einsum():
    a, b = _square_case()
    assert np.allclose(
        np.asarray(batched_matmul_fixed(a, b)), np.asarray(jnp.einsum("bij,jk->bik", a, b))
    )


def test_the_rectangular_case_crashes_instead():
    """When row != inner the mistake is a run-time error, which is how it gets found."""
    a, b = _rectangular_case()
    assert batched_matmul_fixed(a, b).shape == (2, 5, 4)
    with np.testing.assert_raises(TypeError):
        batched_matmul_buggy(a, b)


def test_einsum_subscripts_are_positional_too():
    """`bij,jk->bik` is a string; swapping two letters is as easy and as silent."""
    a, b = _square_case()
    assert not np.allclose(
        np.asarray(jnp.einsum("bij,jk->bik", a, b)), np.asarray(jnp.einsum("bji,jk->bik", a, b))
    )


def test_vmap_is_the_thing_that_removes_the_choice():
    """JAX-specific: lifting a plain matrix product over the batch leaves no axis to pick."""
    a, b = _square_case()
    assert np.allclose(
        np.asarray(jax.vmap(lambda m: m @ b)(a)), np.asarray(batched_matmul_fixed(a, b))
    )
