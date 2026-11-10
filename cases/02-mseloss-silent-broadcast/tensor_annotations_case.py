"""Case 02 under tensor_annotations (google-deepmind/tensor_annotations) + mypy.

Static checking, axes as types. The model is typed too: `x @ w` with
`x: [Batch, Features]` and `w: [Features, Out]` is inferred as `Array2[Batch, Out]`, so
the `[n, 1]` prediction carries its `Out` axis by type all the way to the loss.

*  strict loss (`pred: [Batch]`), called with the model output   -> REJECTED at the call
*  loss as written upstream (`pred: [Batch, Out]`)                -> REJECTED in the body
   `pred - target` is a trailing-axis broadcast, which the stubs do check by name:
   `Array2[A1, A2] - Array1[A2]` needs `A2 == Out`, but the target is `Array1[Batch]`.
*  fix: drop the `Out` axis before the loss
   - `jnp.squeeze(pred, axis=1)` / `pred[:, 0]`  -> `Any`: accepted, but unchecked
   - `jnp.mean(pred, axis=1)`                    -> `Array1[Batch]`: typed and checked;
     reducing the wrong axis gives `Array1[Out]` -- but here the checkers DISAGREE:
     pyright rejects it; mypy accepts it when the call is written inline as an argument.
     With an expected type in context mypy resolves `mean` to its untyped fallback
     overload (`-> Any`); bound to a variable first, it infers `Array1[Out]` and rejects.

The file is therefore expected to FAIL both checkers, with the exact errors listed in
`test_mypy_errors` (2) and `test_pyright_errors` (3). There is no run-time component:
every rejected line still runs and computes the wrong loss, which the remaining tests show.

Run with `./scripts/check_tensor_annotations.sh` (not collected by the default pytest run).
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NewType

import jax.numpy as jnp
from tensor_annotations import axes
from tensor_annotations.jax import Array0, Array1, Array2, float32

Batch = axes.Batch
Features = axes.Features
Out = NewType("Out", axes.Axis)          # nn.Linear(2, 1): the output axis, of size 1

Inputs = Array2[float32, Batch, Features]
Weights = Array2[float32, Features, Out]
Prediction = Array2[float32, Batch, Out]
Target = Array1[float32, Batch]


def linear(x: Inputs, w: Weights) -> Prediction:
    return x @ w                                         # inferred Array2[Batch, Out]


# ------------------------------------------------------------------------- the losses

def mse_strict(pred: Array1[float32, Batch], target: Target) -> Array0[float32]:
    return jnp.mean((pred - target) ** 2)


def mse_as_written_upstream(pred: Prediction, target: Target) -> Array0[float32]:
    """The signature the buggy model actually had: the annotation is *true*."""
    return jnp.mean((pred - target) ** 2)                # [Batch, Out] - [Batch] -> REJECTED


# ---------------------------------------------------------------------- the call sites

def loss_buggy(x: Inputs, w: Weights, t: Target) -> Array0[float32]:
    return mse_strict(linear(x, w), t)                   # [Batch, Out] for [Batch] -> REJECTED


def loss_fixed_squeeze(x: Inputs, w: Weights, t: Target) -> Array0[float32]:
    """The usual fix. `squeeze` is untyped in the stubs (`squeeze: Any`): accepted, unchecked."""
    return mse_strict(jnp.squeeze(linear(x, w), axis=1), t)


def loss_fixed_typed(x: Inputs, w: Weights, t: Target) -> Array0[float32]:
    """Drops `Out` (size 1) with a typed reduction: `mean(axis=1)` -> Array1[Batch]."""
    return mse_strict(jnp.mean(linear(x, w), axis=1), t)


def loss_fixed_typed_wrong_axis(x: Inputs, w: Weights, t: Target) -> Array0[float32]:
    """Reducing `Batch` instead leaves Array1[Out]. pyright rejects; mypy (inline) does not."""
    return mse_strict(jnp.mean(linear(x, w), axis=0), t)  # [Out] for [Batch] -> pyright only


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


def _pyright_errors(tmp: Path) -> list[tuple[int, str]]:
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
    diags = json.loads(run.stdout)["generalDiagnostics"]
    return [(d["range"]["start"]["line"] + 1, d["message"].splitlines()[0])
            for d in diags if d["severity"] == "error"]


_UPSTREAM_BODY = "return jnp.mean((pred - target) ** 2)                # [Batch, Out]"
_STRICT_CALL = "return mse_strict(linear(x, w), t)"
_WRONG_AXIS = "return mse_strict(jnp.mean(linear(x, w), axis=0), t)"


def test_mypy_errors():
    """Both buggy variants are rejected. The wrong-axis typed fix is NOT (see docstring);
    `loss_fixed_squeeze` and `loss_fixed_typed` are accepted."""
    assert _mypy_errors() == [
        (_line_of(_UPSTREAM_BODY),
         'Unsupported operand types for - ("Array2[float32, Batch, Out]" and '
         '"Array1[float32, Batch]")  [operator]'),
        (_line_of(_STRICT_CALL),
         'Argument 1 to "mse_strict" has incompatible type "Array2[float32, Batch, Out]"; '
         'expected "Array1[float32, Batch]"  [arg-type]'),
    ]


def test_pyright_errors(tmp_path):
    """Both buggy variants and the wrong-axis typed fix are rejected."""
    assert _pyright_errors(tmp_path) == [
        (_line_of(_UPSTREAM_BODY),
         'Operator "-" not supported for types "Prediction" and "Target"'),   # = the aliases
        (_line_of(_STRICT_CALL),
         'Argument of type "Prediction" cannot be assigned to parameter '
         '"pred" of type "Array1[float32, Batch]" in function "mse_strict"'),
        (_line_of(_WRONG_AXIS),
         'Argument of type "Array1[float32, Out]" cannot be assigned to parameter '
         '"pred" of type "Array1[float32, Batch]" in function "mse_strict"'),
    ]


def _example():
    x = jnp.stack([jnp.linspace(0, 1, 100), jnp.linspace(1, 2, 100)], axis=1)  # [100, 2]
    w = jnp.array([[2.0], [-1.0]])                                              # [2, 1]
    t = x @ jnp.array([2.0, -1.0])                                              # [100], exact fit
    return x, w, t


def test_annotations_are_not_enforced_at_run_time():
    """Every statically rejected function still runs and returns a wrong, positive loss,
    although the weights are an exact fit."""
    x, w, t = _example()
    assert float(loss_buggy(x, w, t)) > 1e-2
    assert float(mse_as_written_upstream(linear(x, w), t)) > 1e-2
    assert float(loss_fixed_typed_wrong_axis(x, w, t)) > 1e-2


def test_the_fixes_reach_the_optimum():
    x, w, t = _example()
    assert float(loss_fixed_squeeze(x, w, t)) < 1e-10
    assert float(loss_fixed_typed(x, w, t)) < 1e-10
