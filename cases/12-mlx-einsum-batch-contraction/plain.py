"""Case 12 — batched contraction over the wrong axis (NumPy).

Original: ml-explore/mlx#4125.
"""

import numpy as np

RNG = np.random.default_rng(12)


def batched_matmul_buggy(a, b):
    """Contracts `a`'s row axis instead of its inner axis."""
    return np.tensordot(a, b, axes=([1], [0]))


def batched_matmul_fixed(a, b):
    """Contracts the inner axis, which is what `bij,jk->bik` means."""
    return np.tensordot(a, b, axes=([2], [0]))


def _square_case():
    """batch == row == inner == 3: both contractions are legal."""
    return RNG.normal(size=(3, 3, 3)), RNG.normal(size=(3, 4))


def _rectangular_case():
    return RNG.normal(size=(2, 5, 3)), RNG.normal(size=(3, 4))


# --------------------------------------------------------------------------- tests

def test_both_contractions_are_legal_when_the_axes_coincide():
    a, b = _square_case()
    assert batched_matmul_buggy(a, b).shape == batched_matmul_fixed(a, b).shape == (3, 3, 4)


def test_and_they_give_different_answers():
    """Erroneous behaviour: the batch axis has been contracted away and rebuilt elsewhere."""
    a, b = _square_case()
    assert not np.allclose(batched_matmul_buggy(a, b), batched_matmul_fixed(a, b))


def test_the_fixed_version_agrees_with_einsum():
    a, b = _square_case()
    assert np.allclose(batched_matmul_fixed(a, b), np.einsum("bij,jk->bik", a, b))


def test_the_rectangular_case_crashes_instead():
    """When row != inner the mistake is a run-time error, which is how it gets found."""
    a, b = _rectangular_case()
    assert batched_matmul_fixed(a, b).shape == (2, 5, 4)
    try:
        batched_matmul_buggy(a, b)
    except ValueError:
        return
    raise AssertionError("expected a shape error")


def test_einsum_subscripts_are_positional_too():
    """`bij,jk->bik` is a string; swapping two letters is as easy and as silent."""
    a, b = _square_case()
    assert not np.allclose(
        np.einsum("bij,jk->bik", a, b),
        np.einsum("bji,jk->bik", a, b),
    )
