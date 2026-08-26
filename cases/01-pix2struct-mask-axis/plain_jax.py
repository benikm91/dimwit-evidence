"""Case 01 — the same defect in JAX. See plain.py for the narrative."""

import jax.numpy as jnp
import numpy as np
from jax.nn import softmax

NEG = -1e9


def attention_buggy(scores, key_padding_mask):
    additive = (1.0 - key_padding_mask) * NEG
    return softmax(scores + additive[:, None, :, None], axis=-1)


def attention_fixed(scores, key_padding_mask):
    additive = (1.0 - key_padding_mask) * NEG
    return softmax(scores + additive[:, None, None, :], axis=-1)


def _example():
    scores = jnp.zeros((1, 1, 3, 3), dtype=jnp.float32)
    mask = jnp.array([[1.0, 1.0, 0.0]], dtype=jnp.float32)
    return scores, mask


# --------------------------------------------------------------------------- tests

def test_jax_does_not_object():
    """jnp broadcasting accepts both. Same shape out, no warning, no error."""
    scores, mask = _example()
    assert attention_buggy(scores, mask).shape == (1, 1, 3, 3)
    assert attention_fixed(scores, mask).shape == (1, 1, 3, 3)


def test_jit_does_not_object_either():
    """Even tracing the function under jit reveals nothing: shapes are consistent."""
    import jax
    scores, mask = _example()
    jitted = jax.jit(attention_buggy)
    assert jitted(scores, mask).shape == (1, 1, 3, 3)


def test_buggy_attends_to_padding():
    scores, mask = _example()
    assert float(attention_buggy(scores, mask)[0, 0, :, 2].sum()) > 0.9


def test_fixed_ignores_padding():
    scores, mask = _example()
    assert np.allclose(np.asarray(attention_fixed(scores, mask)[0, 0, :, 2]), 0.0, atol=1e-6)
