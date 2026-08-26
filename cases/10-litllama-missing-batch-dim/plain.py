"""Case 10 — model called without the batch dimension (NumPy).

Original: Lightning-AI/lit-llama#166.

The stand-in model contains one operation that assumes the leading axis is the batch — here
a centring step, but in real models it is a KV cache, a mask built from `x.shape[0]`, or a
normalisation. Everything else is rank-polymorphic, which is why the call went through.
"""

import numpy as np

VOCAB, DIM = 7, 4
RNG = np.random.default_rng(3)
EMB = RNG.normal(size=(VOCAB, DIM))
W_OUT = RNG.normal(size=(DIM, VOCAB))


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


IDX = np.array([1, 3, 5, 2])


# --------------------------------------------------------------------------- tests

def test_the_unbatched_call_runs_and_returns_a_plausible_shape():
    """Erroneous behaviour: no exception, and the result looks exactly right."""
    out = next_token_logits_buggy(IDX)
    assert out.shape == (VOCAB,)


def test_the_batched_call_returns_the_same_shape():
    assert next_token_logits_fixed(IDX).shape == (VOCAB,)


def test_but_the_numbers_are_different():
    """The centring step ran over the sequence instead of over the batch."""
    assert not np.allclose(next_token_logits_buggy(IDX), next_token_logits_fixed(IDX))


def test_the_internal_tensor_lost_a_rank():
    assert model(IDX).shape == (4, VOCAB)                    # (seq, vocab)
    assert model(IDX.reshape(1, -1)).shape == (1, 4, VOCAB)  # (batch, seq, vocab)


def test_sampling_from_the_wrong_logits_still_produces_valid_tokens():
    """Generation continues, producing fluent-looking nonsense rather than crashing."""
    logits = next_token_logits_buggy(IDX)
    token = int(np.argmax(logits))
    assert 0 <= token < VOCAB
