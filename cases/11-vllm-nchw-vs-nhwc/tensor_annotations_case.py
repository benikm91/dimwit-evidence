"""Case 11 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

One video backend returns NCHW, the pipeline was written for NHWC. tensor_annotations writes
each layout as a type, and compares axes by identity, not size:

*  the miswiring, `per_channel_mean_nhwc(decoded)` with `decoded: Nchw`    -> REJECTED
   independent of the extents -- also for the square `[2, 3, 3, 3]` frames that shipped,
   where jaxtyping's size-bound names cannot tell the layouts apart.
*  the fix, `jnp.transpose(x, (0, 2, 3, 1))` with a literal permutation     -> typed, accepted;
   a wrong permutation, (0, 3, 2, 1), gives [Frame, Width, Height, Channels] -> REJECTED

Two limits, both recorded in the tests:

*  the consumer's reduction as `plain_jax.py` writes it, `frames.mean(...)`, is a method the
   stubs do not have (`Array4` has no `mean`): rejected in the consumer that buggy and fixed
   share -- a stub gap, not a detection. (`jnp.mean(frames, axis=(0, 1, 2))` would be typed.)
*  the backend's body is untyped (`jnp.asarray` is `Any`), so the layout a backend
   *declares* is never checked against the array it builds.

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
from tensor_annotations.jax import Array1, Array4, float32

Frame = NewType("Frame", axes.Axis)
Channels = axes.Channels
Height = axes.Height
Width = axes.Width

Nchw = Array4[float32, Frame, Channels, Height, Width]
Nhwc = Array4[float32, Frame, Height, Width, Channels]
PerChannel = Array1[float32, Channels]

RNG = np.random.default_rng(11)


def decode_nchw(n_frames: int = 2, channels: int = 3, height: int = 3, width: int = 3) -> Nchw:
    """The PyNvVideoCodec backend: [frames, channels, height, width]. Body is Any."""
    return jnp.asarray(RNG.normal(size=(n_frames, channels, height, width)), dtype=jnp.float32)


def per_channel_mean_nhwc(frames: Nhwc) -> PerChannel:
    """Written for NHWC. `Array4.mean` is not in the stubs -> rejected (stub gap)."""
    return frames.mean(axis=(0, 1, 2))


def to_nhwc(frames_nchw: Nchw) -> Nhwc:
    """The fix: transpose the backend's output into the layout the pipeline expects."""
    return jnp.transpose(frames_nchw, (0, 2, 3, 1))


def to_nhwc_wrong(frames_nchw: Nchw) -> Nhwc:
    """The fix with a wrong permutation: [Frame, Width, Height, Channels] -> REJECTED."""
    return jnp.transpose(frames_nchw, (0, 3, 2, 1))


def normalise_buggy(decoded: Nchw) -> PerChannel:
    """The backend was swapped and nobody transposed -> REJECTED."""
    return per_channel_mean_nhwc(decoded)


def normalise_fixed(decoded: Nchw) -> PerChannel:
    return per_channel_mean_nhwc(to_nhwc(decoded))


def _control(frames: Nhwc) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(jnp.asarray(np.zeros((2, 3, 3, 3))))      # Any: backend bodies are untyped
    reveal_type(jnp.mean(frames, axis=(0, 1, 2)))         # Array1[Channels]: function form typed


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


_MEAN = "return frames.mean(axis=(0, 1, 2))"
_WRONG = "return jnp.transpose(frames_nchw, (0, 3, 2, 1))"
_BUGGY = "return per_channel_mean_nhwc(decoded)"


def test_mypy():
    """mypy reports the wrong permutation against the transpose *arguments* (it infers the
    call from the expected return type), as in case 05."""
    assert _mypy("error") == [
        (_line_of(_MEAN), '"Array4[float32, Frame, Height, Width, Channels]" has no attribute '
                          '"mean"  [attr-defined]'),
        (_line_of(_WRONG), 'Argument 1 to "transpose" has incompatible type '
                           '"Array4[float32, Frame, Channels, Height, Width]"; expected '
                           '"Array4[float32, Frame, Height, Width, Channels]"  [arg-type]'),
        (_line_of(_WRONG), 'Argument 2 to "transpose" has incompatible type '
                           '"tuple[Literal[0], Literal[3], Literal[2], Literal[1]]"; expected '
                           '"tuple[Literal[0], Literal[1], Literal[2], Literal[3]]"  [arg-type]'),
        (_line_of(_BUGGY), 'Argument 1 to "per_channel_mean_nhwc" has incompatible type '
                           '"Array4[float32, Frame, Channels, Height, Width]"; expected '
                           '"Array4[float32, Frame, Height, Width, Channels]"  [arg-type]'),
    ]
    assert [m for _, m in _mypy("note")] == [
        'Revealed type is "Any"',
        'Revealed type is "Array1[float32, Channels]"',
    ]


def test_pyright(tmp_path):
    assert _pyright(tmp_path, "error") == [
        (_line_of(_MEAN), 'Cannot access attribute "mean" for class "Nhwc"'),
        (_line_of(_WRONG), 'Type "Array4[float32, Frame, Width, Height, Channels]" is not '
                           'assignable to return type "Nhwc"'),
        (_line_of(_BUGGY), 'Argument of type "Nchw" cannot be assigned to parameter "frames" '
                           'of type "Nhwc" in function "per_channel_mean_nhwc"'),
    ]
    assert [m.rsplit(" is ", 1)[1] for _, m in _pyright(tmp_path, "information")] == [
        '"Any"',
        '"Array1[float32, Channels]"',
    ]


def test_run_time_square_rgb_frames_give_plausible_wrong_statistics():
    """No run-time check: [2, 3, 3, 3] runs under either layout, with different numbers."""
    nchw = decode_nchw()
    wrong, right = normalise_buggy(nchw), normalise_fixed(nchw)
    assert wrong.shape == right.shape == (3,)
    assert not np.allclose(np.asarray(wrong), np.asarray(right))


def test_run_time_the_fix_matches_numpy():
    nchw = decode_nchw()
    expected = np.asarray(nchw).mean(axis=(0, 2, 3))
    assert np.allclose(np.asarray(normalise_fixed(nchw)), expected, atol=1e-6)
