"""Case 09 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

`ft_ao(Gv)` assumes a grid `(N, 3)`; a single G-vector `(3,)` loses a rank silently.
tensor_annotations puts the rank into the class (`Array1` vs `Array2`), so with the
signature `Gv: Array2[GPoint, Component]` a rank-1 argument is a type error -- *if* its
class is known at the call:

*  the vector as a typed parameter `g: Array1[Component]` (as DimWit's
   `Buggy.scala` and case 10 write it) -- the verdict                       -> REJECTED
*  `g = GRID[1]` (as in plain.py): indexing is `-> Any`                     -> missed
*  `jnp.array([...])` / `jnp.asarray(...)`: `array: Any`, `asarray: Any`   -> missed
*  `jnp.zeros((3,))`: typed by rank only, `Array1[float32, Any]` -- the
   `Any` axis does not rescue it, rank 1 is still not `Array2`              -> REJECTED

The body is untyped either way (`jnp.asarray` is `Any`). The caller-side fix on a typed
vector, `g.reshape(1, 3)`, is itself rejected: `Array1` has no `reshape` method in the
stubs (a stub gap, as `x.max` in case 06).

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
from tensor_annotations.jax import Array1, Array2, float32

GPoint = NewType("GPoint", axes.Axis)
Component = NewType("Component", axes.Axis)     # x, y, z
Center = NewType("Center", axes.Axis)

Grid = Array2[float32, GPoint, Component]
Vector = Array1[float32, Component]
Centers = Array1[float32, Center]
Result = Array2[float32, GPoint, Center]

CENTERS = jnp.array([0.0, 0.5, 1.0, 1.5])
GRID = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])


# ------------------------------------------------------------------------- the routine

def ft_ao_buggy(Gv: Grid, centers: Centers) -> Result:
    """As shipped. The body assumes `Gv` is (N, 3) and never checks that it is."""
    G = jnp.asarray(Gv, dtype=jnp.float32)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


def ft_ao_fixed(Gv: Grid, centers: Centers) -> Result:
    """After PR #3340: one line at the top, restoring the assumption the body makes."""
    G = jnp.asarray(Gv, dtype=jnp.float32).reshape(-1, 3)
    phase = jnp.exp(-0.5 * (G**2).sum(axis=-1))
    return phase[..., None] * centers


# ---------------------------------------------------------------------------- callers

def call_as_reconstructed(grid: Grid, centers: Centers) -> Result:
    """One G-vector taken from a grid, as in plain.py: `grid[1]` is Any -> accepted."""
    g = grid[1]
    return ft_ao_buggy(g, centers)


def call_with_a_literal(centers: Centers) -> Result:
    """A vector written out: `jnp.array` is Any -> accepted."""
    g = jnp.array([1.0, 0.0, 0.0])
    return ft_ao_buggy(g, centers)


def call_with_a_typed_vector(g: Vector, centers: Centers) -> Result:
    """The vector's type is declared: rank 1 for rank 2 -> REJECTED."""
    return ft_ao_buggy(g, centers)


def call_with_zeros(centers: Centers) -> Result:
    """`jnp.zeros((3,))` is Array1[float32, Any]: still rank 1 -> REJECTED."""
    return ft_ao_buggy(jnp.zeros((3,)), centers)


def call_fixed(g: Vector, centers: Centers) -> Result:
    """The caller-side fix: add the grid axis. `Array1.reshape` does not exist in the
    stubs -> rejected as well (stub gap, not a detection)."""
    return ft_ao_buggy(g.reshape(1, 3), centers)


def _control(grid: Grid) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(grid[1])                        # Any: indexing
    reveal_type(jnp.array([1.0, 0.0, 0.0]))     # Any
    reveal_type(jnp.asarray(grid))              # Any
    reveal_type(jnp.zeros((3,)))                # Array1[float32, Any]


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


_TYPED = '"""The vector\'s type is declared: rank 1 for rank 2 -> REJECTED."""'
_ZEROS = "return ft_ao_buggy(jnp.zeros((3,)), centers)"
_FIX = "return ft_ao_buggy(g.reshape(1, 3), centers)"


def _call_line_after(docstring: str) -> int:
    return _line_of(docstring) + 1


def test_mypy():
    """The two calls whose argument has a known class are rejected; so is the fix (stub gap)."""
    assert _mypy("error") == [
        (_call_line_after(_TYPED),
         'Argument 1 to "ft_ao_buggy" has incompatible type "Array1[float32, Component]"; '
         'expected "Array2[float32, GPoint, Component]"  [arg-type]'),
        (_line_of(_ZEROS),
         'Argument 1 to "ft_ao_buggy" has incompatible type "Array1[float32, Any]"; '
         'expected "Array2[float32, GPoint, Component]"  [arg-type]'),
        (_line_of(_FIX),
         '"Array1[float32, Component]" has no attribute "reshape"; maybe "shape"?  [attr-defined]'),
    ]
    assert [m for _, m in _mypy("note")] == [
        'Revealed type is "Any"',
        'Revealed type is "Any"',
        'Revealed type is "Any"',
        'Revealed type is "Array1[float32, Any]"',
    ]


def test_pyright(tmp_path):
    assert _pyright(tmp_path, "error") == [
        (_call_line_after(_TYPED),
         'Argument of type "Vector" cannot be assigned to parameter "Gv" of type "Grid" '
         'in function "ft_ao_buggy"'),
        (_line_of(_ZEROS),
         'Argument of type "Array1[float32, Any]" cannot be assigned to parameter "Gv" of '
         'type "Grid" in function "ft_ao_buggy"'),
        (_line_of(_FIX), 'Cannot access attribute "reshape" for class "Vector"'),
    ]
    # compare only the inferred type: pyright re-renders the expression text ("(3, )")
    assert [m.rsplit(" is ", 1)[1] for _, m in _pyright(tmp_path, "information")] == [
        '"Any"',                    # grid[1]
        '"Any"',                    # jnp.array([...])
        '"Any"',                    # jnp.asarray(grid)
        '"Array1[float32, Any]"',   # jnp.zeros((3,))
    ]


def test_run_time_a_single_vector_silently_loses_a_rank():
    """No run-time check: the accepted calls return rank 1 instead of rank 2."""
    assert call_as_reconstructed(GRID, CENTERS).shape == (4,)
    assert call_with_a_literal(CENTERS).shape == (4,)
    assert call_with_a_typed_vector(GRID[1], CENTERS).shape == (4,)


def test_run_time_the_fixes_agree():
    g = GRID[1]
    assert call_fixed(g, CENTERS).shape == (1, 4)
    assert np.allclose(np.asarray(ft_ao_fixed(g, CENTERS)), np.asarray(call_fixed(g, CENTERS)))
