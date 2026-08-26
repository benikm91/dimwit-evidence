"""Case 10 in JAX — the same program as `plain.py`."""

import jax
import jax.numpy as jnp
import numpy as np

VOCAB, DIM = 7, 4
EMB = jnp.asarray(np.random.default_rng(3).normal(size=(VOCAB, DIM)), dtype=jnp.float32)
W_OUT = jnp.asarray(np.random.default_rng(4).normal(size=(DIM, VOCAB)), dtype=jnp.float32)


def model(idx):
    """Written for [batch, seq]. Nothing here *requires* that, which is the problem."""
    x = EMB[idx]                                  # [..., seq, dim]
    x = x - x.mean(axis=0, keepdims=True)         # meant to centre across the batch
    return x @ W_OUT                              # [..., seq, vocab]


def next_token_logits_buggy(idx_1d):
    """As shipped: unbatched call, then index the leading axis."""
    logits = model(idx_1d)
    return logits[-1]


def next_token_logits_fixed(idx_1d):
    """After PR #166."""
    logits = model(idx_1d.reshape(1, -1))
    return logits[0, -1]


IDX = jnp.array([1, 3, 5, 2])


# --------------------------------------------------------------------------- tests

def test_the_unbatched_call_runs_and_returns_a_plausible_shape():
    """Erroneous behaviour: no exception, and the result looks exactly right."""
    assert next_token_logits_buggy(IDX).shape == (VOCAB,)


def test_the_batched_call_returns_the_same_shape():
    assert next_token_logits_fixed(IDX).shape == (VOCAB,)


def test_but_the_numbers_are_different():
    """The centring step ran over the sequence instead of over the batch."""
    assert not np.allclose(
        np.asarray(next_token_logits_buggy(IDX)), np.asarray(next_token_logits_fixed(IDX))
    )


def test_the_internal_tensor_lost_a_rank():
    assert model(IDX).shape == (4, VOCAB)                    # (seq, vocab)
    assert model(IDX.reshape(1, -1)).shape == (1, 4, VOCAB)  # (batch, seq, vocab)


def test_jit_compiles_the_unbatched_call():
    """JAX-specific: both ranks trace and compile, each to a different program."""
    assert jax.jit(model)(IDX).shape == (4, VOCAB)
    assert jax.jit(model)(IDX.reshape(1, -1)).shape == (1, 4, VOCAB)


def test_vmap_is_the_thing_that_would_have_forced_the_question():
    """Mapping the single-example model makes the batch axis explicit and external."""
    out = jax.vmap(model)(IDX.reshape(1, -1))
    assert out.shape == (1, 4, VOCAB)
    # and note it is NOT the same as calling the model on the 2-D input directly,
    # because the centring step now sees one example at a time
    assert not np.allclose(np.asarray(out), np.asarray(model(IDX.reshape(1, -1))))
