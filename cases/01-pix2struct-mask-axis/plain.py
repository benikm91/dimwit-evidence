"""Case 01 — Pix2Struct attention mask broadcast onto the wrong axis (NumPy).

Original: huggingface/transformers#23974, fixed by PR #23976.

The padding mask carries one flag per KEY position. It has to be broadcast along the
key axis of the [batch, heads, query, key] score tensor. The shipped code broadcast it
along the query axis instead.
"""

import numpy as np

NEG = -1e9


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention_buggy(scores, key_padding_mask):
    """As shipped. `mask[:, None, :, None]` -> [batch, 1, query, 1]."""
    additive = (1.0 - key_padding_mask) * NEG
    biased = scores + additive[:, None, :, None]
    return _softmax(biased, axis=-1)


def attention_fixed(scores, key_padding_mask):
    """After PR #23976. `mask[:, None, None, :]` -> [batch, 1, 1, key]."""
    additive = (1.0 - key_padding_mask) * NEG
    biased = scores + additive[:, None, None, :]
    return _softmax(biased, axis=-1)


def _example():
    """One batch, one head, three positions, the last of which is padding."""
    scores = np.zeros((1, 1, 3, 3), dtype=np.float32)
    mask = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)  # position 2 is padding
    return scores, mask


# --------------------------------------------------------------------------- tests

def test_buggy_and_fixed_agree_on_shape():
    """The whole problem: nothing about the shapes distinguishes the two versions."""
    scores, mask = _example()
    assert attention_buggy(scores, mask).shape == attention_fixed(scores, mask).shape == (1, 1, 3, 3)


def test_buggy_attends_to_padding():
    """Erroneous behaviour: every query still puts weight on the padded key."""
    scores, mask = _example()
    w = attention_buggy(scores, mask)
    assert w[0, 0, :, 2].sum() > 0.9, "buggy version attends to the padding token"


def test_buggy_destroys_the_last_query_row_instead():
    """The mask landed on the query axis, so query 2 was silenced rather than key 2."""
    scores, mask = _example()
    w = attention_buggy(scores, mask)
    # row 2 was pushed to -inf uniformly, so it comes back out as a uniform distribution
    assert np.allclose(w[0, 0, 2, :], np.full(3, 1 / 3), atol=1e-5)


def test_fixed_ignores_padding():
    """Correct behaviour: the padded key receives zero weight from every query."""
    scores, mask = _example()
    w = attention_fixed(scores, mask)
    assert np.allclose(w[0, 0, :, 2], 0.0, atol=1e-6)
    assert np.allclose(w.sum(axis=-1), 1.0, atol=1e-6)


def test_the_two_versions_disagree_numerically():
    """No exception anywhere — just different numbers."""
    scores, mask = _example()
    assert not np.allclose(attention_buggy(scores, mask), attention_fixed(scores, mask))
