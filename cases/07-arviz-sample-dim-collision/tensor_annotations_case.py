"""Case 07 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

ArviZ stacks `chain` and `draw` into one dimension and called it `sample`, colliding with
a user dimension of the same name. tensor_annotations replaces string dims with axis
*types*, so the collision becomes: the stacked axis is given the same type as the user's
`Sample` axis.

*  stacking is untyped: flattening two axes needs `jnp.reshape`, and the stubs declare
   `reshape: Any`. The stacked axis is therefore never derived from `Chain` and `Draw` --
   whatever the stack function declares as its return type is accepted unchecked.
*  so the buggy declaration, `-> Array2[Sample, A]` (stacked axis typed like the user's
   `Sample`), is accepted, and with a user `Sample` axis the result is
   `Array2[Sample, Sample]`: two axes of one type. Averaging over "the first `Sample`" --
   upstream's selection-by-name picking the stacked axis -- type-checks.        -> MISSED

The verbatim string-dims API (`NamedArray` in `jaxtyping_case.py`) has no tensor_annotations
counterpart; this file uses the library the way it is meant to be used, with typed arrays.

Expected to PASS both checkers with zero errors; the tests assert that, plus a control of
the inferred types. Run with `./scripts/check_tensor_annotations.sh`.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NewType, TypeVar, reveal_type

import jax.numpy as jnp
import numpy as np
from tensor_annotations import axes
from tensor_annotations.jax import Array1, Array2, Array3, float32

Chain = NewType("Chain", axes.Axis)
Draw = NewType("Draw", axes.Axis)
Sample = NewType("Sample", axes.Axis)                 # the user's own dimension
StackedSample = NewType("StackedSample", axes.Axis)   # "__sample__" after the fix
A = TypeVar("A", bound=axes.Axis)

Posterior = Array3[float32, Chain, Draw, Sample]


# ---------------------------------------------------------------------- arviz: stacking

def stack_buggy(posterior: Array3[float32, Chain, Draw, A]) -> Array2[float32, Sample, A]:
    """As shipped: the stacked dimension is `sample`. `reshape` is Any, so the declared
    return type is never checked against the body."""
    chain, draw, rest = posterior.shape
    return jnp.reshape(posterior, (chain * draw, rest))


def stack_fixed(posterior: Array3[float32, Chain, Draw, A]) -> Array2[float32, StackedSample, A]:
    """After PR #1647: a distinct dimension for the stacked draws."""
    chain, draw, rest = posterior.shape
    return jnp.reshape(posterior, (chain * draw, rest))


# ------------------------------------------------------------------ user: downstream code

def mean_over_user_samples_buggy(stacked: Array2[float32, Sample, Sample]) -> Array1[float32, Sample]:
    """Meant: average over the USER's samples. Upstream, selection by name picked the first
    `sample` -- the stacked one. Both axes have the same type, so axis 0 type-checks."""
    return jnp.mean(stacked, axis=0)


def mean_over_user_samples_fixed(stacked: Array2[float32, StackedSample, Sample]) -> Array1[float32, StackedSample]:
    return jnp.mean(stacked, axis=1)


def _control(posterior: Posterior) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(jnp.reshape(posterior, (12, 5)))      # Any: stacking is untyped
    reveal_type(stack_buggy(posterior))               # Array2[Sample, Sample]: as declared
    reveal_type(stack_fixed(posterior))               # Array2[StackedSample, Sample]


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)


def _clean(msg: str) -> str:
    return (msg.replace("tensor_annotations.jax.", "").replace("tensor_annotations.axes.", "")
               .replace("tensor_annotations_case.", ""))


def _mypy(severity: str) -> list[str]:
    from mypy import api  # pyright: ignore[reportMissingImports]  (only in the check env)
    stdout, _, _ = api.run(["--no-incremental", "--no-error-summary", str(_SOURCE)])
    return [_clean(m) for m in re.findall(rf"^.*?:\d+: {severity}: (.*)$", stdout, re.MULTILINE)]


def _pyright(tmp: Path, severity: str) -> list[str]:
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
    return [d["message"] for d in json.loads(run.stdout)["generalDiagnostics"]
            if d["severity"] == severity]


def test_mypy_accepts_everything():
    assert _mypy("error") == []
    assert _mypy("note") == [
        'Revealed type is "Any"',
        'Revealed type is "Array2[float32, Sample, Sample]"',
        'Revealed type is "Array2[float32, StackedSample, Sample]"',
    ]


def test_pyright_accepts_everything(tmp_path):
    assert _pyright(tmp_path, "error") == []
    assert _pyright(tmp_path, "information") == [
        'Type of "jnp.reshape(posterior, (12, 5))" is "Any"',
        'Type of "stack_buggy(posterior)" is "Array2[float32, Sample, Sample]"',
        'Type of "stack_fixed(posterior)" is "Array2[float32, StackedSample, Sample]"',
    ]


def _posterior():
    return jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5)   # chain, draw, sample


def test_run_time_the_buggy_routine_averages_the_wrong_axis():
    """One value per user sample (5) instead of one per stacked draw (12)."""
    assert mean_over_user_samples_buggy(stack_buggy(_posterior())).shape == (5,)


def test_run_time_the_fixed_routine_averages_the_user_samples():
    out = mean_over_user_samples_fixed(stack_fixed(_posterior()))
    assert out.shape == (12,)
    assert np.allclose(np.asarray(out), np.asarray(_posterior()).reshape(12, 5).mean(axis=1))
