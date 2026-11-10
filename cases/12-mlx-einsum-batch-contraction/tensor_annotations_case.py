"""Case 12 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

A batched contraction `a[batch, row, inner] . b[inner, col]` performed over `row` instead of
`inner`. Both operations the upstream code uses are untyped in the stubs:

*  `jnp.tensordot` is declared `tensordot: Any`                              -> MISSED
*  `jnp.einsum` is declared `einsum: Any`                                    -> MISSED

so the return annotation `Array3[Batch, Row, Col]` is satisfied by whatever axes are
contracted, also for rectangular inputs (where jaxtyping fires at run time).

(`a @ b` is typed -- `Array3[Batch, Row, Col]` -- but has no axis argument, so the upstream
mistake cannot be written with it; rewriting to it would measure a different program.)

Expected to PASS both checkers with zero errors; the tests assert that, plus a control of
the inferred types. Run with `./scripts/check_tensor_annotations.sh`.
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

Batch = axes.Batch
Row = NewType("Row", axes.Axis)
Inner = NewType("Inner", axes.Axis)
Col = NewType("Col", axes.Axis)

Lhs = Array3[float32, Batch, Row, Inner]
Rhs = Array2[float32, Inner, Col]
Out = Array3[float32, Batch, Row, Col]


def batched_matmul_buggy(a: Lhs, b: Rhs) -> Out:
    """Contracts `row` (axis 1) instead of `inner` (axis 2)."""
    return jnp.tensordot(a, b, axes=([1], [0]))


def batched_matmul_fixed(a: Lhs, b: Rhs) -> Out:
    return jnp.tensordot(a, b, axes=([2], [0]))


def batched_matmul_einsum_buggy(a: Lhs, b: Rhs) -> Out:
    """The same mistake in einsum letters: `r` is summed against `b`'s first axis."""
    return jnp.einsum("bri,rc->bic", a, b)


def batched_matmul_einsum_fixed(a: Lhs, b: Rhs) -> Out:
    return jnp.einsum("bri,ic->brc", a, b)


def _control(a: Lhs, b: Rhs) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(a)                                     # typed operands ...
    reveal_type(jnp.tensordot(a, b, axes=([1], [0])))  # ... Any result
    reveal_type(jnp.einsum("bri,ic->brc", a, b))       # Any


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
        'Revealed type is "Array3[float32, Batch, Row, Inner]"',
        'Revealed type is "Any"',
        'Revealed type is "Any"',
    ]


def test_pyright_accepts_everything(tmp_path):
    assert _pyright(tmp_path, "error") == []
    assert [m.rsplit(" is ", 1)[1] for m in _pyright(tmp_path, "information")] == [
        '"Array3[float32, Batch, Row, Inner]"',
        '"Any"',
        '"Any"',
    ]


def test_run_time_square_case_wrong_values():
    """No run-time check: batch == row == inner == 3 runs and returns wrong numbers."""
    rng = np.random.default_rng(12)
    a = jnp.asarray(rng.normal(size=(3, 3, 3)), dtype=jnp.float32)
    b = jnp.asarray(rng.normal(size=(3, 4)), dtype=jnp.float32)
    for buggy, fixed in ((batched_matmul_buggy, batched_matmul_fixed),
                         (batched_matmul_einsum_buggy, batched_matmul_einsum_fixed)):
        assert buggy(a, b).shape == fixed(a, b).shape == (3, 3, 4)
        assert not np.allclose(np.asarray(buggy(a, b)), np.asarray(fixed(a, b)))


def test_run_time_rectangular_case_is_not_caught_by_the_annotation():
    """(2, 5, 3): the buggy tensordot raises inside jnp (row 5 != inner 3), not at any
    annotation; the fixed one matches a @ b."""
    a, b = jnp.ones((2, 5, 3)), jnp.ones((3, 4))
    try:
        batched_matmul_buggy(a, b)
        raised = False
    except Exception:
        raised = True
    assert raised
    assert np.allclose(np.asarray(batched_matmul_fixed(a, b)), np.asarray(a @ b))
