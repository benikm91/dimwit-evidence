"""Case 10 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

lit-llama's `generate` called a `[batch, seq]` model on a 1-D token vector. With the model
typed `idx: Array2[int32, Batch, Seq]` and `generate`'s `idx: Array1[int32, Seq]`:

*  the upstream call, `model(idx_cond)` with
   `idx_cond = idx if T <= max_len else idx[-max_len:]`                     -> REJECTED
   Slicing is `Any`, so `idx_cond` is `Array1[int32, Seq] | Any`. A union is checked
   member by member, and the `Array1` member is not an `Array2`: rank is in the class.
*  the fix, `idx.reshape(1, -1)` (JAX's spelling of upstream's `.view(1, -1)`)
                                                                            -> rejected too
   `Array1` has no `reshape` method in the stubs: a stub gap, not a detection.
*  the fix as `jnp.reshape(idx, (1, -1))`: `reshape: Any`                   -> accepted, unchecked

The model body is untyped (`emb[idx]` is indexing, `Any`), and so is `logits[-1]`.

Expected to FAIL both checkers with exactly the errors in `test_mypy` / `test_pyright`.
Run with `./scripts/check_tensor_annotations.sh`.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NewType, reveal_type

import jax.numpy as jnp
import numpy as np
from tensor_annotations import axes
from tensor_annotations.jax import Array1, Array2, Array3, float32, int32

Batch = axes.Batch
Seq = NewType("Seq", axes.Axis)
Vocab = NewType("Vocab", axes.Axis)
Embed = NewType("Embed", axes.Axis)

Tokens = Array2[int32, Batch, Seq]
TokenVector = Array1[int32, Seq]
Embedding = Array2[float32, Vocab, Embed]
Unembedding = Array2[float32, Embed, Vocab]
Logits = Array3[float32, Batch, Seq, Vocab]

VOCAB, DIM = 7, 4
EMB = jnp.asarray(np.random.default_rng(3).normal(size=(VOCAB, DIM)), dtype=jnp.float32)
W_OUT = jnp.asarray(np.random.default_rng(4).normal(size=(DIM, VOCAB)), dtype=jnp.float32)


def model(idx: Tokens, emb: Embedding, w_out: Unembedding) -> Logits:
    """Written for [batch, seq]."""
    x = emb[idx]                                  # indexing: Any
    x = x - x.mean(axis=0, keepdims=True)
    return x @ w_out


def next_token_logits_buggy(idx: TokenVector, max_len: int, emb: Embedding, w_out: Unembedding):
    """As shipped: unbatched call, then index the leading axis."""
    T = idx.shape[0]
    idx_cond = idx if T <= max_len else idx[-max_len:]
    logits = model(idx_cond, emb, w_out)          # [Seq] | Any for [Batch, Seq] -> REJECTED
    return logits[-1]


def next_token_logits_fixed(idx: TokenVector, max_len: int, emb: Embedding, w_out: Unembedding):
    """After PR #166, in JAX's spelling. `Array1.reshape` does not exist in the stubs."""
    T = idx.shape[0]
    idx_cond = idx if T <= max_len else idx[-max_len:]
    logits = model(idx_cond.reshape(1, -1), emb, w_out)
    return logits[0, -1]


def next_token_logits_fixed_fn(idx: TokenVector, max_len: int, emb: Embedding, w_out: Unembedding):
    """The same fix with the function form: `jnp.reshape` is Any -> accepted, unchecked."""
    T = idx.shape[0]
    idx_cond = idx if T <= max_len else idx[-max_len:]
    logits = model(jnp.reshape(idx_cond, (1, -1)), emb, w_out)
    return logits[0, -1]


def _control(idx: TokenVector, emb: Embedding) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(idx if idx.shape[0] <= 3 else idx[-3:])   # Array1[int32, Seq] | Any
    reveal_type(emb[jnp.zeros((1, 3), dtype=jnp.int32)])  # Any: indexing
    reveal_type(jnp.reshape(idx, (1, -1)))                # Any


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)


def _line_of(fragment: str) -> int:
    return next(i for i, line in enumerate(_SOURCE.read_text().splitlines(), 1)
                if fragment in line)


def _clean(msg: str) -> str:
    return (msg.replace("tensor_annotations.jax.", "").replace("tensor_annotations.axes.", "")
               .replace("tensor_annotations_case.", ""))


def _mypy(severity: str) -> list[tuple[int, str]]:
    from mypy import api  # pyright: ignore[reportMissingImports]  (only in the check env)
    stdout, _, _ = api.run(["--no-incremental", "--no-error-summary", str(_SOURCE)])
    return [(int(m.group(1)), _clean(m.group(2)))
            for m in re.finditer(rf"^.*?:(\d+): {severity}: (.*)$", stdout, re.MULTILINE)]


def _pyright(tmp: Path, severity: str) -> list[tuple[int, str]]:
    """pyright from a scratch dir, so no repo config applies. The site-packages holding
    tensor_annotations goes into `extraPaths`: under `uv run --with` it is an overlay that
    comes *after* the project venv, and pyright would otherwise stop at the real `jax`
    (which ships inline types) before ever seeing `jax-stubs`."""
    import tensor_annotations  # pyright: ignore[reportMissingImports]
    site = str(Path(tensor_annotations.__file__).parent.parent)
    (tmp / "pyrightconfig.json").write_text(json.dumps({"extraPaths": [site]}))
    run = subprocess.run(
        [sys.executable, "-m", "pyright", "--outputjson", "--pythonpath", sys.executable,
         str(_SOURCE)],
        cwd=tmp, capture_output=True, text=True,
    )
    return [(d["range"]["start"]["line"] + 1, d["message"].splitlines()[0])
            for d in json.loads(run.stdout)["generalDiagnostics"] if d["severity"] == severity]


_BUGGY = "logits = model(idx_cond, emb, w_out)"
_FIX = "logits = model(idx_cond.reshape(1, -1), emb, w_out)"


def test_mypy():
    assert _mypy("error") == [
        (_line_of(_BUGGY),
         'Argument 1 to "model" has incompatible type "Array1[int32, Seq] | Any"; '
         'expected "Array2[int32, Batch, Seq]"  [arg-type]'),
        (_line_of(_FIX), 'Item "Array1[int32, Seq]" of "Array1[int32, Seq] | Any" has no '
                         'attribute "reshape"  [union-attr]'),
    ]
    assert [m for _, m in _mypy("note")] == [
        'Revealed type is "Array1[int32, Seq] | Any"',
        'Revealed type is "Any"',
        'Revealed type is "Any"',
    ]


def test_pyright(tmp_path):
    assert _pyright(tmp_path, "error") == [
        (_line_of(_BUGGY),
         'Argument of type "TokenVector | Any" cannot be assigned to parameter "idx" of type '
         '"Tokens" in function "model"'),
        (_line_of(_FIX), 'Cannot access attribute "reshape" for class "TokenVector"'),
    ]
    # compare only the inferred type: pyright may re-render the expression text
    assert [m.rsplit(" is ", 1)[1] for _, m in _pyright(tmp_path, "information")] == [
        '"Array1[int32, Seq] | Any"',
        '"Any"',
        '"Any"',
    ]


IDX = jnp.array([1, 3, 5, 2])


def test_run_time_the_unbatched_call_runs_and_returns_the_wrong_thing():
    """No run-time check: the model accepts the 1-D vector; `logits[-1]` is then the last
    *position's* row only because the leading axis happens to be the sequence."""
    out = next_token_logits_buggy(IDX, 8, EMB, W_OUT)
    assert out.shape == (VOCAB,)
    batched = model(IDX.reshape(1, -1), EMB, W_OUT)[0, -1]
    assert not np.allclose(np.asarray(out), np.asarray(batched))


def test_run_time_the_fixes_agree():
    a = next_token_logits_fixed(IDX, 8, EMB, W_OUT)
    b = next_token_logits_fixed_fn(IDX, 8, EMB, W_OUT)
    assert a.shape == (VOCAB,)
    assert np.allclose(np.asarray(a), np.asarray(b))
