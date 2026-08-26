"""Case 01 under jaxtyping.

The interesting result is that jaxtyping's verdict depends on whether the two sequence
axes happen to have the same length:

*  self-attention  (query == key)  -> annotation passes, defect MISSED
*  cross-attention (query != key)  -> annotation fails,  defect caught at RUN TIME

Pix2Struct's visual encoder is self-attention, so jaxtyping would not have helped.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

NEG = -1e9


@jaxtyped(typechecker=beartype)
def attention_buggy(
    scores: Float[Array, "batch heads query key"],
    key_padding_mask: Float[Array, "batch key"],
) -> Float[Array, "batch heads query key"]:
    additive = (1.0 - key_padding_mask) * NEG
    return jnp.asarray(scores + additive[:, None, :, None])


@jaxtyped(typechecker=beartype)
def attention_fixed(
    scores: Float[Array, "batch heads query key"],
    key_padding_mask: Float[Array, "batch key"],
) -> Float[Array, "batch heads query key"]:
    additive = (1.0 - key_padding_mask) * NEG
    return jnp.asarray(scores + additive[:, None, None, :])


# --------------------------------------------------------------------------- tests

def test_self_attention_buggy_passes_the_typechecker():
    """query == key == 3, so `query` and `key` unify and the annotation is satisfied."""
    scores = jnp.zeros((1, 1, 3, 3), dtype=jnp.float32)
    mask = jnp.ones((1, 3), dtype=jnp.float32)
    out = attention_buggy(scores, mask)  # no error: MISSED
    assert out.shape == (1, 1, 3, 3)


def test_cross_attention_buggy_is_rejected_at_runtime():
    """query = 4, key = 3. Now the broadcast produces [1,1,4,4] and the return annotation fails."""
    scores = jnp.zeros((1, 1, 4, 3), dtype=jnp.float32)
    mask = jnp.ones((1, 3), dtype=jnp.float32)
    with pytest.raises(Exception):
        attention_buggy(scores, mask)


def test_cross_attention_fixed_is_accepted():
    scores = jnp.zeros((1, 1, 4, 3), dtype=jnp.float32)
    mask = jnp.ones((1, 3), dtype=jnp.float32)
    assert attention_fixed(scores, mask).shape == (1, 1, 4, 3)
