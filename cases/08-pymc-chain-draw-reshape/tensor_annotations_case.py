"""Case 08 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

pymc moves `chain, draw` to the front and flattens them, but read the sizes for `reshape`
from the array *before* the transpose.

*  the transpose permutation is computed from the dims at run time (`list[int]`). On a
   typed array that matches NO `transpose` overload: the typed ones want a literal tuple,
   the untyped fallback wants a plain `jax.Array`. Both checkers reject the call -- in the
   helper that buggy and fixed share, so it is not a detection -- and the result is `Any`.
*  the flatten is `jnp.reshape`, declared `reshape: Any` in the stubs -- and fed a tuple of
   ints, which carries no axes in any case. Even a literal (typed) permutation would end
   here.
*  so the declared point-list type, `Array2[ChainDraw, Team]`, is never compared with what
   the body computes, and nothing distinguishes buggy from fixed.            -> MISSED

Expected to FAIL both checkers, but only on the shared transpose (and the control line
repeating it); the tests assert exactly that, plus a control of the inferred types.
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
from tensor_annotations.jax import Array2, Array3, float32

Team = NewType("Team", axes.Axis)
Draw = NewType("Draw", axes.Axis)
Chain = NewType("Chain", axes.Axis)
ChainDraw = NewType("ChainDraw", axes.Axis)     # the flattened sampling axis

Dataset = Array3[float32, Team, Draw, Chain]
PointList = Array2[float32, ChainDraw, Team]

SAMPLE_DIMS = ("chain", "draw")
DIMS = ("team", "draw", "chain")


def _transpose_to_front(data: Dataset, dims: tuple[str, ...], sample_dims: tuple[str, ...]):
    order = list(sample_dims) + [d for d in dims if d not in sample_dims]
    return jnp.transpose(data, [dims.index(d) for d in order])   # untypeable: list[int]


def to_point_list_buggy(data: Dataset, dims: tuple[str, ...],
                        sample_dims: tuple[str, ...] = SAMPLE_DIMS) -> PointList:
    """As shipped: `rest` is read from the ORIGINAL array, not the transposed one."""
    moved = _transpose_to_front(data, dims, sample_dims)
    rest = data.shape[len(sample_dims):]          # <- the defect
    return jnp.reshape(moved, (-1, *rest))


def to_point_list_fixed(data: Dataset, dims: tuple[str, ...],
                        sample_dims: tuple[str, ...] = SAMPLE_DIMS) -> PointList:
    """After PR #7180: transpose first, then read the shape."""
    moved = _transpose_to_front(data, dims, sample_dims)
    rest = moved.shape[len(sample_dims):]
    return jnp.reshape(moved, (-1, *rest))


def _control(data: Dataset) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(jnp.transpose(data, [2, 1, 0]))       # run-time permutation: untyped
    reveal_type(jnp.transpose(data, (2, 1, 0)))       # literal tuple: typed
    reveal_type(jnp.reshape(data, (-1, 5)))           # reshape: Any
    reveal_type(data.shape[1:])                       # ints only


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)


def _clean(msg: str) -> str:
    return (msg.replace("tensor_annotations.jax.", "").replace("tensor_annotations.axes.", "")
               .replace("tensor_annotations_case.", ""))


def _line_of(fragment: str) -> int:
    return next(i for i, line in enumerate(_SOURCE.read_text().splitlines(), 1)
                if fragment in line)


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


_HELPER = "return jnp.transpose(data, [dims.index(d) for d in order])"
_CONTROL = "reveal_type(jnp.transpose(data, [2, 1, 0]))"


_NO_OVERLOAD_MYPY = ('No overload variant of "transpose" matches argument types '
                     '"Array3[float32, Team, Draw, Chain]", "list[int]"  [call-overload]')


def test_mypy():
    """Only the run-time permutation is rejected -- shared by buggy and fixed alike."""
    assert _mypy("error") == [(_line_of(_HELPER), _NO_OVERLOAD_MYPY),
                              (_line_of(_CONTROL), _NO_OVERLOAD_MYPY)]
    assert [m for _, m in _mypy("note") if m.startswith("Revealed")] == [
        'Revealed type is "Any"',
        'Revealed type is "Array3[float32, Chain, Draw, Team]"',
        'Revealed type is "Any"',
        'Revealed type is "tuple[int, int]"',
    ]


def test_pyright(tmp_path):
    no_overload = 'No overloads for "transpose" match the provided arguments'
    bad_arg = ('Argument of type "list[int]" cannot be assigned to parameter "axes" of type '
               '"Tuple[L2, L1, L0]" in function "transpose"')
    helper, control = _line_of(_HELPER), _line_of(_CONTROL)
    assert _pyright(tmp_path, "error") == [(helper, no_overload), (helper, bad_arg),
                                           (control, no_overload), (control, bad_arg)]
    assert [m for _, m in _pyright(tmp_path, "information")] == [
        'Type of "jnp.transpose(data, [2, 1, 0])" is "Unknown"',
        'Type of "jnp.transpose(data, (2, 1, 0))" is "Array3[float32, Chain, Draw, Team]"',
        'Type of "jnp.reshape(data, (-1, 5))" is "Any"',
        'Type of "data.shape[1:]" is "Tuple[int, int]"',
    ]


def test_run_time_scrambled_point_list():
    """No run-time check: team=5, draw=2, chain=3 gives a (10, 3) list instead of (6, 5)."""
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    assert to_point_list_buggy(data, DIMS).shape == (10, 3)
    assert to_point_list_fixed(data, DIMS).shape == (6, 5)


def test_run_time_equal_rectangles_are_harmless():
    """team == chain: `data.shape[2:]` equals `moved.shape[2:]`, so the buggy reshape is the
    fixed one -- identical output, not just the same shape."""
    data = jnp.arange(3 * 2 * 3, dtype=jnp.float32).reshape(3, 2, 3)
    buggy, fixed = to_point_list_buggy(data, DIMS), to_point_list_fixed(data, DIMS)
    assert np.allclose(np.asarray(buggy), np.asarray(fixed))
