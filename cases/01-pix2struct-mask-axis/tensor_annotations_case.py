"""Case 01 under tensor_annotations (google-deepmind/tensor_annotations) + mypy.

tensor_annotations is a *static* checker: axes are types (`Query` and `Key` are distinct
`NewType`s), so unlike jaxtyping it does not confuse two axes that share an extent. The
verdict nevertheless turns on how the code is written:

*  verbatim upstream code  (`mask[:, None, :, None]`)     -> mypy accepts, defect MISSED
   `Array.__getitem__` is typed `-> Any`; the expansion leaves the type system and the
   `Any` is silently accepted by `+` and by the return annotation.
*  rewritten through typed ops (`transpose` + trailing broadcast)
                                                           -> mypy rejects, caught statically

This file is therefore expected to FAIL mypy with exactly one error, in
`attention_buggy_typed`. The tests below run mypy on this file and assert precisely that.

Note the arithmetic is written `(mask - 1.0) * 1e9` rather than `(1.0 - mask) * -1e9`:
the stubs define no reflected operators (`__rsub__`), so `float - Array2` is a type error.

Run with `./scripts/check_tensor_annotations.sh` (not collected by the default pytest run).
"""

import re
from pathlib import Path
from typing import NewType

import jax.numpy as jnp
import numpy as np
from jax.nn import softmax
from tensor_annotations import axes
from tensor_annotations.jax import Array2, Array4, float32

Batch = axes.Batch
Heads = NewType("Heads", axes.Axis)
Query = NewType("Query", axes.Axis)
Key = NewType("Key", axes.Axis)

Scores = Array4[float32, Batch, Heads, Query, Key]
KeyMask = Array2[float32, Batch, Key]


# ------------------------------------------------------------------ verbatim upstream code

def attention_buggy(scores: Scores, key_padding_mask: KeyMask) -> Scores:
    additive = (key_padding_mask - 1.0) * 1e9            # Array2[Any, Batch, Key]
    expanded = additive[:, None, :, None]                # Any  <- leaves the type system
    return scores + expanded                             # Any, accepted as Scores


def attention_buggy_annotated(scores: Scores, key_padding_mask: KeyMask) -> Scores:
    """Annotating the intermediate does not help: any type may be assigned from `Any`.

    The annotation below is *false* (the value is [batch, 1, seq, 1]), and mypy accepts it.
    """
    additive = (key_padding_mask - 1.0) * 1e9
    expanded: Scores = additive[:, None, :, None]
    return scores + expanded


def attention_fixed(scores: Scores, key_padding_mask: KeyMask) -> Scores:
    additive = (key_padding_mask - 1.0) * 1e9
    return scores + additive[:, None, None, :]


# ------------------------------------------------- rewritten so the types carry the check
# No `None`-indexing: move [batch, key] to the trailing axes and use the stubs' trailing
# broadcast `Array4[.., A3, A4] + Array2[A3, A4]`, which is checked by axis identity.

def attention_buggy_typed(scores: Scores, key_padding_mask: KeyMask) -> Scores:
    """The upstream mistake in typed form: the mask lines up with `Query`, not `Key`."""
    additive = (key_padding_mask - 1.0) * 1e9            # Array2[Any, Batch, Key]
    hkbq = jnp.transpose(scores, (1, 3, 0, 2))           # Array4[Heads, Key, Batch, Query]
    masked = hkbq + additive                             # Query vs Key -> REJECTED by mypy
    return jnp.transpose(masked, (2, 0, 3, 1))


def attention_fixed_typed(scores: Scores, key_padding_mask: KeyMask) -> Scores:
    additive = (key_padding_mask - 1.0) * 1e9            # Array2[Any, Batch, Key]
    hqbk = jnp.transpose(scores, (1, 2, 0, 3))           # Array4[Heads, Query, Batch, Key]
    masked = hqbk + additive                             # Batch/Key line up -> accepted
    return jnp.transpose(masked, (2, 0, 1, 3))           # back to [Batch, Heads, Query, Key]


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)


def _mypy_errors() -> list[tuple[int, str]]:
    from mypy import api  # pyright: ignore[reportMissingImports]  (only in the check env)
    stdout, _, _ = api.run(["--no-incremental", "--no-error-summary", str(_SOURCE)])
    return [(int(m.group(1)), m.group(2))
            for m in re.finditer(r"^.*?:(\d+): error: (.*)$", stdout, re.MULTILINE)]


def _line_of(fragment: str) -> int:
    return next(i for i, line in enumerate(_SOURCE.read_text().splitlines(), 1)
                if fragment in line)


def test_mypy_rejects_only_the_typed_buggy_version():
    """Exactly one static error, on the typed buggy broadcast. In particular the verbatim
    buggy code (`attention_buggy`, `attention_buggy_annotated`) is NOT flagged: MISSED."""
    errors = _mypy_errors()
    assert errors == [(
        _line_of("masked = hkbq + additive"),
        'Unsupported operand types for + ("Array4[float32, Heads, Key, Batch, Query]" '
        'and "Array2[Any, Batch, Key]")  [operator]',
    )]


def _example():
    scores = jnp.zeros((1, 1, 3, 3), dtype=jnp.float32)
    mask = jnp.array([[1.0, 1.0, 0.0]], dtype=jnp.float32)
    return scores, mask


def test_annotations_are_not_enforced_at_run_time():
    """tensor_annotations has no run-time component: every buggy version runs silently."""
    scores, mask = _example()
    for f in (attention_buggy, attention_buggy_annotated, attention_buggy_typed):
        assert f(scores, mask).shape == (1, 1, 3, 3)


def test_buggy_attends_to_padding():
    scores, mask = _example()
    for f in (attention_buggy, attention_buggy_typed):
        probs = softmax(f(scores, mask), axis=-1)
        assert float(probs[0, 0, :, 2].sum()) > 0.9


def test_fixed_ignores_padding():
    scores, mask = _example()
    for f in (attention_fixed, attention_fixed_typed):
        probs = softmax(f(scores, mask), axis=-1)
        assert np.allclose(np.asarray(probs[0, 0, :, 2]), 0.0, atol=1e-6)


def test_typed_rewrite_agrees_with_fixed_on_cross_attention():
    scores = jnp.arange(2 * 2 * 4 * 3, dtype=jnp.float32).reshape(2, 2, 4, 3)
    mask = jnp.array([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    assert np.allclose(
        np.asarray(attention_fixed_typed(scores, mask)),
        np.asarray(attention_fixed(scores, mask)),
    )
