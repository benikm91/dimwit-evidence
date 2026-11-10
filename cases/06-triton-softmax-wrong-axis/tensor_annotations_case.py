"""Case 06 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

`keep_dims=False` drops the reduced axis, and the rank-reduced result re-attaches to the
LAST axis when it broadcasts back. tensor_annotations' stubs broadcast trailing axes *by
name*, which is exactly the check this defect needs -- when the types get that far.

A. as written upstream (method form, `dim: int`, `keep_dims: bool`)  -> no verdict
   `Array2` has no `max`/`sum` methods in the stubs. Buggy AND fixed are rejected with the
   same "no attribute max" error: a stub gap, not a detection. (`num.sum` is not flagged:
   after the failed `x.max`, `num` is already `Any`.)
B. function form, `jnp.max/jnp.sum(x, axis=dim, keepdims=keep_dims)`  -> MISSED
   A run-time `dim: int` / `keep_dims: bool` matches only the untyped overloads: `Any`.
C. most favourable: literal `axis=1`, intermediate annotated           -> REJECTED
   `jnp.max` is untyped (`max: Any`), and `x - Any` is `Any`, so the first half of
   softmax cannot be checked at all; annotating `num: Tile` restores the type. Then
   `jnp.sum(num, axis=1)` is `Array1[Row]`, and `num / den` needs the trailing axis to be
   `Col` -> rejected, on the square 2x2 tile from the issue too. The fix
   (`keepdims=True`) is accepted -- but only because it is untyped as well ("keepdims =
   True: yet be to be typed" in the stubs); `axis=0` (accidentally correct) is accepted
   and typed.
D. minimal scope: softmax over a vector                                 -> scope enforced
   Handing it a tile is rejected; lifting it with `jax.vmap` is `Any` (`vmap: Any`).

Expected to FAIL both checkers with exactly the errors in `test_mypy` / `test_pyright`.
Run with `./scripts/check_tensor_annotations.sh` (not collected by the default pytest run).
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NewType, reveal_type

import jax
import jax.numpy as jnp
import numpy as np
from tensor_annotations import axes
from tensor_annotations.jax import Array1, Array2, float32

Row = NewType("Row", axes.Axis)
Col = NewType("Col", axes.Axis)
N = NewType("N", axes.Axis)

Tile = Array2[float32, Row, Col]
Vector = Array1[float32, N]

REPORTED = jnp.array([[0.0, 2.0], [3.0, 1.0]])   # the 2x2 tile from the issue


# ------------------------------------------------- A. as written upstream: method form

def softmax_buggy(x: Tile, dim: int, keep_dims: bool = False) -> Tile:
    """As shipped: `keep_dims` reaches the two internal reductions."""
    z = x - x.max(axis=dim, keepdims=keep_dims)
    num = jnp.exp(z)
    den = num.sum(axis=dim, keepdims=keep_dims)
    return num / den


def softmax_fixed(x: Tile, dim: int) -> Tile:
    """After PR #11409: `keep_dims=True` in both reductions, and no parameter."""
    z = x - x.max(axis=dim, keepdims=True)
    num = jnp.exp(z)
    return num / num.sum(axis=dim, keepdims=True)


# ------------------------------------------- B. function form, run-time dim and flag

def softmax_fn_buggy(x: Tile, dim: int, keep_dims: bool = False) -> Tile:
    z = x - jnp.max(x, axis=dim, keepdims=keep_dims)
    num = jnp.exp(z)
    return num / jnp.sum(num, axis=dim, keepdims=keep_dims)


# --------------------------------- C. most favourable: literal axis, annotated intermediate

def softmax_dim1_buggy(x: Tile) -> Tile:
    """dim=1 with the default keep_dims=False. `den` is [Row]; `num / den` aligns it
    with the trailing axis, Col -> REJECTED."""
    num: Tile = jnp.exp(x - jnp.max(x, axis=1))   # first half untyped: max is Any
    den = jnp.sum(num, axis=1)                     # Array1[Row]
    return num / den                               # [Row, Col] / [Row] -> REJECTED


def softmax_dim1_fixed(x: Tile) -> Tile:
    """keepdims=True: correct, and accepted -- because it is untyped (Any)."""
    num: Tile = jnp.exp(x - jnp.max(x, axis=1, keepdims=True))
    den = jnp.sum(num, axis=1, keepdims=True)
    return num / den


def softmax_dim0_default(x: Tile) -> Tile:
    """dim=0 with keep_dims=False: accidentally correct upstream, and typed + accepted
    here -- `den` is [Col], which is the trailing axis."""
    num: Tile = jnp.exp(x - jnp.max(x, axis=0))
    den = jnp.sum(num, axis=0)                     # Array1[Col]
    return num / den


# ------------------------------------------------------------- D. minimal scope: a vector

def softmax_vector(v: Vector) -> Vector:
    z: Vector = v - jnp.max(v)
    num = jnp.exp(z)
    return num / jnp.sum(num)


def softmax_vector_given_a_tile(x: Tile) -> Tile:
    """The scope is enforced: a tile does not fit `Array1[N]` -> REJECTED."""
    return softmax_vector(x)


def softmax_lifted(x: Tile) -> Tile:
    """The tile version as the vector version lifted -- `jax.vmap` is `Any` in the stubs,
    so the lifted result is unchecked."""
    return jax.vmap(softmax_vector)(x)


def _control(x: Tile, v: Vector, dim: int) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(jnp.max(x, axis=1))                     # Any: max is untyped
    reveal_type(jnp.sum(x, axis=dim))                   # Any: run-time axis
    reveal_type(jnp.sum(x, axis=1))                     # Array1[Row]
    reveal_type(jnp.sum(x, axis=1, keepdims=True))      # Any: keepdims=True untyped
    reveal_type(jnp.sum(x, axis=0))                     # Array1[Col]
    reveal_type(jax.vmap(softmax_vector)(x))            # Any


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


_A_BUGGY_MAX = "z = x - x.max(axis=dim, keepdims=keep_dims)"
_A_FIXED_MAX = "z = x - x.max(axis=dim, keepdims=True)"
_C_BUGGY = "return num / den                               # [Row, Col] / [Row]"
_D_MISUSE = "return softmax_vector(x)"


def test_mypy():
    """A: buggy and fixed rejected alike (stub gap). C: only the buggy literal-axis version.
    D: the tile handed to the vector function."""
    assert _mypy("error") == [
        (_line_of(_A_BUGGY_MAX), '"Array2[float32, Row, Col]" has no attribute "max"  [attr-defined]'),
        (_line_of(_A_FIXED_MAX), '"Array2[float32, Row, Col]" has no attribute "max"  [attr-defined]'),
        (_line_of(_C_BUGGY), 'Unsupported operand types for / ("Array2[float32, Row, Col]" and '
                 '"Array1[float32, Row]")  [operator]'),
        (_line_of(_D_MISUSE), 'Incompatible return value type (got "Array1[float32, N]", '
                              'expected "Array2[float32, Row, Col]")  [return-value]'),
        (_line_of(_D_MISUSE), 'Argument 1 to "softmax_vector" has incompatible type '
                              '"Array2[float32, Row, Col]"; expected "Array1[float32, N]"  [arg-type]'),
    ]
    assert [m for _, m in _mypy("note")] == [
        'Revealed type is "Any"',
        'Revealed type is "Any"',
        'Revealed type is "Array1[float32, Row]"',
        'Revealed type is "Any"',
        'Revealed type is "Array1[float32, Col]"',
        'Revealed type is "Any"',
    ]


def test_pyright(tmp_path):
    """Same lines as mypy. pyright types `num` as "Unknown | Tile" (declared Tile, assigned
    from an untyped expression) and still rejects the division."""
    assert _pyright(tmp_path, "error") == [
        (_line_of(_A_BUGGY_MAX), 'Cannot access attribute "max" for class "Tile"'),
        (_line_of(_A_FIXED_MAX), 'Cannot access attribute "max" for class "Tile"'),
        (_line_of(_C_BUGGY), 'Operator "/" not supported for types "Unknown | Tile" and '
                             '"Array1[float32, Row]"'),
        (_line_of(_D_MISUSE), 'Type "Vector" is not assignable to return type "Tile"'),
        (_line_of(_D_MISUSE), 'Argument of type "Tile" cannot be assigned to parameter "v" '
                              'of type "Vector" in function "softmax_vector"'),
    ]
    assert [m for _, m in _pyright(tmp_path, "information")] == [
        'Type of "jnp.max(x, axis=1)" is "Any"',
        'Type of "jnp.sum(x, axis=dim)" is "Any"',
        'Type of "jnp.sum(x, axis=1)" is "Array1[float32, Row]"',
        'Type of "jnp.sum(x, axis=1, keepdims=True)" is "Any"',
        'Type of "jnp.sum(x, axis=0)" is "Array1[float32, Col]"',
        'Type of "jax.vmap(softmax_vector)(x)" is "Any"',
    ]


def test_run_time_the_reported_numbers():
    """No run-time check: every buggy variant reproduces the issue's row sums."""
    for out in (softmax_buggy(REPORTED, dim=1), softmax_fn_buggy(REPORTED, dim=1),
                softmax_dim1_buggy(REPORTED)):
        assert np.allclose(np.asarray(out).sum(axis=1), [0.3979, 5.4493], atol=1e-4)


def test_run_time_the_fixes_return_distributions():
    for out in (softmax_fixed(REPORTED, dim=1), softmax_dim1_fixed(REPORTED),
                softmax_lifted(REPORTED)):
        assert np.allclose(np.asarray(out).sum(axis=1), 1.0)
    assert np.allclose(np.asarray(softmax_dim0_default(REPORTED)).sum(axis=0), 1.0)
