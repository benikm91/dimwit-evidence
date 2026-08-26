"""Case 09 — a single G-vector passed as (3,) instead of (1, 3) (NumPy).

Original: pyscf/pyscf#2961, fixed by PR #3340. The physics is replaced by a stand-in with
the same shape behaviour: one complex-ish weight per (G-vector, basis centre) pair.
"""

import numpy as np

CENTERS = np.array([0.0, 0.5, 1.0, 1.5])  # four basis-function centres


def ft_ao_buggy(Gv, centers=CENTERS):
    """As shipped. The body assumes `Gv` is (N, 3) and never checks that it is."""
    G = np.asarray(Gv, dtype=float)
    phase = np.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


def ft_ao_fixed(Gv, centers=CENTERS):
    """After PR #3340: one line at the top, restoring the assumption the body makes."""
    G = np.asarray(Gv, dtype=float).reshape(-1, 3)
    phase = np.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


GRID = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


# --------------------------------------------------------------------------- tests

def test_a_proper_grid_behaves_the_same_either_way():
    assert np.allclose(ft_ao_buggy(GRID), ft_ao_fixed(GRID))
    assert ft_ao_buggy(GRID).shape == (3, 4)


def test_a_single_vector_silently_loses_a_rank():
    """Erroneous behaviour: (4,) instead of (1, 4). This is the upstream regression test.

    The body reduces over the last axis expecting one number per G-vector. Given a bare
    `(3,)` it reduces the *only* axis, so `phase` is a scalar, and the multiply against
    `centers` then broadcasts a scalar over the centres instead of a column over a grid.
    """
    g = GRID[1]
    ref = ft_ao_fixed(g.reshape(1, 3))
    dat = ft_ao_buggy(g)
    assert ref.shape == (1, 4)
    assert dat.shape == (4,)
    assert dat.shape != ref.shape


def test_the_fix_makes_the_two_calls_agree():
    g = GRID[1]
    assert ft_ao_fixed(g).shape == ft_ao_fixed(g.reshape(1, 3)).shape == (1, 4)
    assert np.allclose(ft_ao_fixed(g), ft_ao_fixed(g.reshape(1, 3)))


def test_the_rank_loss_propagates_as_a_wrong_sum_not_an_exception():
    """A caller summing over the G-grid gets a per-centre total from one branch and a
    per-centre value from the other. Both are length-4 arrays of plausible numbers."""
    g = GRID[1]
    buggy_total = ft_ao_buggy(g).sum(axis=0)      # sums over CENTRES: a scalar
    fixed_total = ft_ao_fixed(g).sum(axis=0)      # sums over G points: shape (4,)
    assert np.ndim(buggy_total) == 0
    assert fixed_total.shape == (4,)


def test_numpy_never_objects():
    """(3,) against (4,) via the trailing-axis rule: legal at every step."""
    g = np.zeros(3)
    assert (np.exp(-0.5 * (g**2).sum(axis=-1))[..., None] * CENTERS).shape == (4,)
