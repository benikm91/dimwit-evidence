"""Case 03 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

Axes are types here, so unlike jaxtyping the batch axis and the vector axis *are*
distinguishable even though both have extent 3. Whether that helps depends on the scope of
the interface -- the same distinction DimWit's verdict draws.

A. upstream scope: `cross` over a whole array, with an axis argument       -> MISSED
   `jnp.cross` is declared `cross: Any`; its result is `Any` whatever axis is chosen.
   Buggy (first axis of size 3) and fixed (`axis=-1`) are both accepted, and so is a
   deliberately swapped return annotation. The axis search is typed (`a.shape` is
   `Tuple[int, int]`), but yields an `int`, which carries no axis identity.

B. minimal scope, as DimWit's `Fixed.scala` defines it: the cross product of two
   3-vectors on ONE axis, `cross(v1: Array1[DT, L], v2: Array1[DT, L]) -> Array1[DT, L]`,
   batched by mapping a per-sample function                                 -> REJECTED
   There is no axis argument left to default. Crossing a `Spatial` vector with an `Rgb`
   one -- DimWit's `Buggy.scala` -- fails because both arguments share the type variable
   `L`. The body (`jnp.cross` on two vectors) is untyped, but at this scope it has no axis
   to choose. Limit: the batching lift, `jax.vmap`, is `Any` (`vmap: Any`), so the lift
   itself is unchecked -- mapping over the wrong axis (`in_axes=1`) is accepted, where
   DimWit's `zipvmap(Axis[Sample])` names the axis it maps over.

Expected to FAIL both checkers with exactly the one error in `test_mypy` / `test_pyright`.
Run with `./scripts/check_tensor_annotations.sh` (not collected by the default pytest run).
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NewType, TypeVar, reveal_type

import jax
import jax.numpy as jnp
import numpy as np
from tensor_annotations import axes
from tensor_annotations.jax import Array1, Array2, DType, float32

Batch = axes.Batch
Sample = NewType("Sample", axes.Axis)      # rows of the batch (extent 3 in the repro)
Spatial = NewType("Spatial", axes.Axis)    # the 3-vector components
Rgb = NewType("Rgb", axes.Axis)            # another axis of extent 3

DT = TypeVar("DT", bound=DType)
L = TypeVar("L", bound=axes.Axis)

Vectors = Array2[float32, Batch, Spatial]


# ------------------------------------------------- A. upstream scope: an axis argument

def cross_buggy(a: Vectors, b: Vectors) -> Vectors:
    """keras' torch backend: torch.cross picks the first axis of size 3."""
    axis = next(i for i, s in enumerate(a.shape) if s == 3)   # int: no axis identity
    return jnp.cross(a, b, axisa=axis, axisb=axis, axisc=axis)  # Any -> accepted


def cross_fixed(a: Vectors, b: Vectors) -> Vectors:
    return jnp.cross(a, b, axisa=-1, axisb=-1, axisc=-1)


def cross_buggy_annotated(a: Vectors, b: Vectors) -> Vectors:
    """Annotating the result does not help: `Any` may be assigned to any type."""
    out: Vectors = jnp.cross(a, b, axisa=0, axisb=0, axisc=0)
    return out


def cross_swapped_return(a: Vectors, b: Vectors) -> Array2[float32, Spatial, Batch]:
    """A deliberately WRONG return annotation ([Spatial, Batch] for a [Batch, Spatial]
    result), also accepted: the result of `jnp.cross` carries no axes to compare."""
    return jnp.cross(a, b, axisa=-1, axisb=-1, axisc=-1)


# -------------------------------------- B. minimal scope: two 3-vectors on one axis

def cross(v1: Array1[DT, L], v2: Array1[DT, L]) -> Array1[DT, L]:
    """The concept at its scope: no axis to choose, and the axis survives into the result."""
    return jnp.cross(v1, v2)


def cross_batch_fixed(a: Array2[float32, Sample, Spatial],
                      b: Array2[float32, Sample, Spatial]) -> Array2[float32, Sample, Spatial]:
    def per_sample(x: Array1[float32, Spatial], y: Array1[float32, Spatial]) -> Array1[float32, Spatial]:
        return cross(x, y)
    return jax.vmap(per_sample)(a, b)


def cross_batch_buggy(a: Array2[float32, Sample, Spatial],
                      b: Array2[float32, Sample, Rgb]) -> Array2[float32, Sample, Spatial]:
    """DimWit's Buggy.scala: a spatial vector crossed with a colour triple. Every axis has
    extent 3, so torch.cross would return a number; here `L` cannot be both -> REJECTED."""
    def per_sample(x: Array1[float32, Spatial], y: Array1[float32, Rgb]) -> Array1[float32, Spatial]:
        return cross(x, y)
    return jax.vmap(per_sample)(a, b)


def cross_batch_wrong_lift(a: Array2[float32, Sample, Spatial],
                           b: Array2[float32, Sample, Spatial]) -> Array2[float32, Sample, Spatial]:
    """The limit: mapping over Spatial instead of Sample. `jax.vmap` is Any -> accepted."""
    def per_sample(x: Array1[float32, Spatial], y: Array1[float32, Spatial]) -> Array1[float32, Spatial]:
        return cross(x, y)
    return jax.vmap(per_sample, in_axes=1, out_axes=1)(a, b)


def _control(a: Vectors) -> None:
    """Never called. Shows the checkers see the stubs (`a` is typed) and where `Any` arises."""
    reveal_type(a)
    reveal_type(jnp.cross(a, a, axisa=-1, axisb=-1, axisc=-1))   # Any: cross untyped
    reveal_type(jax.vmap(cross))                                  # Any: vmap untyped


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)


def _buggy_minimal_line() -> int:
    """The `return cross(x, y)` inside `cross_batch_buggy` (three identical lines exist)."""
    lines = _SOURCE.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("def cross_batch_buggy"))
    return next(i for i in range(start, len(lines)) if lines[i].strip() == "return cross(x, y)") + 1


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


def test_mypy():
    """A: everything accepted (MISSED). B: only the Spatial x Rgb cross is rejected."""
    assert _mypy("error") == [
        (_buggy_minimal_line(),
         'Argument 2 to "cross" has incompatible type "Array1[float32, Rgb]"; '
         'expected "Array1[float32, Spatial]"  [arg-type]'),
    ]
    assert [m for _, m in _mypy("note")] == [
        'Revealed type is "Array2[float32, Batch, Spatial]"',
        'Revealed type is "Any"',
        'Revealed type is "Any"',
    ]


def test_pyright(tmp_path):
    assert _pyright(tmp_path, "error") == [
        (_buggy_minimal_line(),
         'Argument of type "Array1[float32, Rgb]" cannot be assigned to parameter "v2" of '
         'type "Array1[DT@cross, L@cross]" in function "cross"'),
    ]
    assert [m.rsplit(" is ", 1)[1] for _, m in _pyright(tmp_path, "information")] == [
        '"Array2[float32, Batch, Spatial]"',
        '"Any"',
        '"Any"',
    ]


def _example():
    a = jnp.arange(9.0).reshape(3, 3)
    b = jnp.arange(9.0)[::-1].reshape(3, 3)
    return a, b


def test_run_time_upstream_scope_same_shape_different_numbers():
    """Run time: no check either. The buggy version returns a [3, 3] array of wrong values."""
    a, b = _example()
    assert cross_buggy(a, b).shape == cross_fixed(a, b).shape == (3, 3)
    assert not np.allclose(np.asarray(cross_buggy(a, b)), np.asarray(cross_fixed(a, b)))
    assert np.allclose(np.asarray(cross_buggy_annotated(a, b)), np.asarray(cross_buggy(a, b)))


def test_run_time_minimal_scope_agrees_with_numpy():
    a, b = _example()
    expected = np.cross(np.asarray(a), np.asarray(b))
    assert np.allclose(np.asarray(cross_fixed(a, b)), expected)
    assert np.allclose(np.asarray(cross_batch_fixed(a, b)), expected)


def test_run_time_the_wrong_lift_runs_and_is_wrong():
    """The unchecked lift: crossing along Sample instead of Spatial, silently."""
    a, b = _example()
    assert not np.allclose(np.asarray(cross_batch_wrong_lift(a, b)), np.asarray(cross_batch_fixed(a, b)))
