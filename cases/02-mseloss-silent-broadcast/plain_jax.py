"""Case 02 in JAX. JAX does not even emit PyTorch's warning."""

import jax
import jax.numpy as jnp
import pytest


def mse_as_written_upstream(pred, target):
    """pred [n, 1], target [n] -> residuals [n, n]. No error, no warning."""
    return jnp.mean((pred - target) ** 2)


def mse_strict(pred, target):
    """The same guard as `plain.py::mse_strict`: the shapes have to agree exactly."""
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    return jnp.mean((pred - target) ** 2)


# --------------------------------------------------------------------------- tests

def test_jax_broadcasts_silently():
    pred = jnp.zeros((100, 1))
    target = jnp.zeros(100)
    assert (pred - target).shape == (100, 100)
    assert float(mse_as_written_upstream(pred, target)) == 0.0


def test_fixed_rejects_the_column_vector():
    """The assertion is the whole fix, and it has to be written by hand."""
    with pytest.raises(AssertionError):
        mse_strict(jnp.zeros((100, 1)), jnp.zeros(100))


def test_jit_accepts_the_buggy_loss():
    """Shape inference under jit proves well-formedness, not intent."""
    pred = jnp.linspace(0, 1, 100).reshape(100, 1)
    target = jnp.linspace(0, 1, 100)
    assert float(jax.jit(mse_as_written_upstream)(pred, target)) > 0.0
    assert float(jax.jit(mse_strict)(pred.reshape(-1), target)) < 1e-12


def test_gradients_are_computed_happily_from_the_wrong_loss():
    """The defect survives autodiff: a clean gradient of the wrong objective."""
    g = jax.grad(lambda p, t: mse_as_written_upstream(p, t))(jnp.ones((10, 1)), jnp.zeros(10))
    assert g.shape == (10, 1)
