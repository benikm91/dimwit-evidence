"""Case 09 in JAX — the same program as `plain.py`."""

import jax
import jax.numpy as jnp
import numpy as np

CENTERS = jnp.array([0.0, 0.5, 1.0, 1.5])  # four basis-function centres


def ft_ao_buggy(Gv, centers=CENTERS):
    """As shipped. The body assumes `Gv` is (N, 3) and never checks that it is."""
    G = jnp.asarray(Gv, dtype=jnp.float32)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


def ft_ao_fixed(Gv, centers=CENTERS):
    """After PR #3340: one line at the top, restoring the assumption the body makes."""
    G = jnp.asarray(Gv, dtype=jnp.float32).reshape(-1, 3)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


GRID = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


# --------------------------------------------------------------------------- tests

def test_a_proper_grid_behaves_the_same_either_way():
    assert np.allclose(np.asarray(ft_ao_buggy(GRID)), np.asarray(ft_ao_fixed(GRID)))
    assert ft_ao_buggy(GRID).shape == (3, 4)


def test_a_single_vector_silently_loses_a_rank():
    """Erroneous behaviour: (4,) instead of (1, 4). This is the upstream regression test."""
    g = GRID[1]
    assert ft_ao_fixed(g.reshape(1, 3)).shape == (1, 4)
    assert ft_ao_buggy(g).shape == (4,)


def test_the_fix_makes_the_two_calls_agree():
    g = GRID[1]
    assert ft_ao_fixed(g).shape == ft_ao_fixed(g.reshape(1, 3)).shape == (1, 4)
    assert np.allclose(np.asarray(ft_ao_fixed(g)), np.asarray(ft_ao_fixed(g.reshape(1, 3))))


def test_the_rank_loss_propagates_as_a_wrong_sum_not_an_exception():
    """One branch sums over centres, the other over G points. Both are plausible numbers."""
    g = GRID[1]
    assert jnp.ndim(ft_ao_buggy(g).sum(axis=0)) == 0
    assert ft_ao_fixed(g).sum(axis=0).shape == (4,)


def test_jit_specialises_on_each_rank_without_complaint():
    """JAX-specific: the same source traces to two different programs, one per input rank."""
    fn = jax.jit(ft_ao_buggy)
    assert fn(GRID).shape == (3, 4)
    assert fn(GRID[1]).shape == (4,)   # retraced, recompiled, still wrong
