"""Case 13 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

`cv2.resize` takes `dsize` as (width, height) and returns (height, width); the wrapper
forwarded a (height, width) pair unflipped.

Most favourable annotations, mirroring DimWit's `innerResize`: the resize is generic over
the frame's own axes and keeps them (`Array2[A1, A2] -> Array2[A1, A2]`), since a resize
resizes an axis rather than renaming it. It does not help:

*  the extents are `tuple[int, int]`. tensor_annotations has no type for a single extent
   (DimWit's `AxisExtent[W]`), so nothing ties `dsize[0]` to the width axis, and the
   unflipped pair is as well typed as the flipped one.                      -> MISSED
*  the body is untyped anyway: `jnp.arange` and indexing (`frame[rows][:, cols]`) are
   `Any`, so the declared `Array2[A1, A2]` result is never compared with what is built.

Expected to PASS both checkers with zero errors; the tests assert that, plus a control of
the inferred types. Run with `./scripts/check_tensor_annotations.sh`.
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import TypeVar, reveal_type

import jax.numpy as jnp
import numpy as np
from tensor_annotations import axes
from tensor_annotations.jax import Array2, float32

A1 = TypeVar("A1", bound=axes.Axis)
A2 = TypeVar("A2", bound=axes.Axis)
Height = axes.Height
Width = axes.Width

Frame = Array2[float32, Height, Width]
ATARI_FRAME = (210, 160)


def inner_resize(frame: Array2[float32, A1, A2], dsize: tuple[int, int]) -> Array2[float32, A1, A2]:
    """The `cv2.resize` half: `dsize` is (width, height); the result is (height, width)."""
    out_w, out_h = dsize
    in_h, in_w = frame.shape
    rows = jnp.clip(jnp.arange(out_h) * in_h // out_h, 0, in_h - 1)
    cols = jnp.clip(jnp.arange(out_w) * in_w // out_w, 0, in_w - 1)
    return frame[rows][:, cols]


def resize_buggy(frame: Array2[float32, A1, A2], size_hw: tuple[int, int]) -> Array2[float32, A1, A2]:
    """As shipped: advertises (height, width) and forwards it unflipped."""
    return inner_resize(frame, size_hw)


def resize_fixed(frame: Array2[float32, A1, A2], size_hw: tuple[int, int]) -> Array2[float32, A1, A2]:
    """After PR #1312: the extents are flipped at the boundary."""
    height, width = size_hw
    return inner_resize(frame, (width, height))


def _control(frame: Frame, rows: Array2[float32, Height, Width]) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(frame.shape)                    # Tuple[int, int]: no axis link
    reveal_type(jnp.arange(84))                 # untyped
    reveal_type(frame[rows])                    # Any: indexing
    reveal_type(resize_buggy(frame, (84, 64)))  # Array2[Height, Width]: as declared


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
        'Revealed type is "tuple[int, int]"',
        'Revealed type is "Any"',
        'Revealed type is "Any"',
        'Revealed type is "Array2[float32, Height, Width]"',
    ]


def test_pyright_accepts_everything(tmp_path):
    assert _pyright(tmp_path, "error") == []
    assert [m.rsplit(" is ", 1)[1] for m in _pyright(tmp_path, "information")] == [
        '"Tuple[int, int]"',
        '"Any"',
        '"Any"',
        '"Array2[float32, Height, Width]"',
    ]


def _frame():
    return jnp.asarray(
        np.arange(ATARI_FRAME[0] * ATARI_FRAME[1], dtype=np.float32).reshape(ATARI_FRAME))


def test_run_time_the_swap_produces_a_transposed_frame():
    """No run-time check: (84, 64) requested as (H, W) comes back 64 high and 84 wide."""
    assert resize_buggy(_frame(), (84, 64)).shape == (64, 84)
    assert resize_fixed(_frame(), (84, 64)).shape == (84, 64)


def test_run_time_square_targets_hide_it():
    assert resize_buggy(_frame(), (84, 84)).shape == resize_fixed(_frame(), (84, 84)).shape
