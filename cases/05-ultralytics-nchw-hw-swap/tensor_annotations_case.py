"""Case 05 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

The defect: `input_width = input_shape[2]; input_height = input_shape[3]` on an NCHW shape,
then a letterbox buffer built from the swapped extents.

*  as written upstream (`onnx_input_shape: tuple[int, int, int, int]`)    -> MISSED
   `jnp.zeros((h, w, c))` is typed by rank only: `Array3[float32, Any, Any, Any]`. Axes
   that are `Any` match every annotation, and `[None, ...]` (adding the batch axis) is
   `Any` altogether, so the `ModelInput` return annotation is never compared to anything.
*  the buffer annotated `Letterbox = Array3[Height, Width, Channels]`       -> MISSED
   (assigned from `Array3[Any, Any, Any]`).
*  the layout carried by a typed template array, extents read from `.shape` -> MISSED
   `template.shape` is `Tuple[int, int, int]`; the entries carry no axis.
*  typed rewrite: build the buffer from the template with `zeros_like` + `transpose`
                                                                          -> REJECTED
   The swap becomes a wrong permutation, the result is `[Width, Height, Channels]`, and
   both checkers reject it -- independent of the sizes, so also for a square 640x640
   model. Like the typed rewrites in cases 01 and 02, this catches a re-expressed program,
   not the code that was shipped.

Expected to FAIL both checkers with exactly the errors in `test_mypy` / `test_pyright`.
Run with `./scripts/check_tensor_annotations.sh` (not collected by the default pytest run).
"""

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NewType, reveal_type

import jax.numpy as jnp
from tensor_annotations import axes
from tensor_annotations.jax import Array3, Array4, float32

Batch = axes.Batch
Channels = axes.Channels
Height = axes.Height          # model input height
Width = axes.Width            # model input width
SrcHeight = NewType("SrcHeight", axes.Axis)
SrcWidth = NewType("SrcWidth", axes.Axis)

Image = Array3[float32, SrcHeight, SrcWidth, Channels]
ModelInput = Array4[float32, Batch, Channels, Height, Width]
Letterbox = Array3[float32, Height, Width, Channels]
Template = Array3[float32, Channels, Height, Width]

ONNX_SHAPE_RECT = (1, 3, 640, 480)     # N, C, H = 640, W = 480
ONNX_SHAPE_SQUARE = (1, 3, 640, 640)   # YOLO's default export
IMAGE = jnp.zeros((1080, 1920, 3), dtype=jnp.float32)


# ------------------------------------------------------------------ as written upstream

def preprocess_buggy(image: Image, onnx_input_shape: tuple[int, int, int, int]) -> ModelInput:
    """As shipped. `input_shape` is (N, C, H, W); the two spatial entries are swapped."""
    input_width = onnx_input_shape[2]   # actually the height
    input_height = onnx_input_shape[3]  # actually the width
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


def preprocess_fixed(image: Image, onnx_input_shape: tuple[int, int, int, int]) -> ModelInput:
    """After PR #23402."""
    input_height = onnx_input_shape[2]
    input_width = onnx_input_shape[3]
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


def preprocess_buggy_annotated(image: Image, onnx_input_shape: tuple[int, int, int, int]) -> ModelInput:
    """Annotating the buffer does not help: `Array3[Any, Any, Any]` fits `Letterbox`."""
    input_width = onnx_input_shape[2]
    input_height = onnx_input_shape[3]
    letterboxed: Letterbox = jnp.zeros((input_height, input_width, image.shape[2]))
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


# ------------------------------------------- layout carried by a typed template array

def preprocess_from_template_buggy(image: Image, template: Template) -> ModelInput:
    """The axes of `template` are typed -- its `.shape` is `Tuple[int, int, int]`."""
    input_width, input_height = template.shape[1], template.shape[2]
    letterboxed: Letterbox = jnp.zeros((input_height, input_width, image.shape[2]))
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


# ----------------------------------------- typed rewrite: no extents, only typed ops

def letterbox_typed_buggy(template: Template) -> Letterbox:
    """[C, H, W] -> [W, H, C]: the swap re-expressed as a permutation. REJECTED."""
    return jnp.transpose(jnp.zeros_like(template), (2, 1, 0))


def letterbox_typed_fixed(template: Template) -> Letterbox:
    """[C, H, W] -> [H, W, C]."""
    return jnp.transpose(jnp.zeros_like(template), (1, 2, 0))


def _control(template: Template) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(jnp.zeros((640, 480, 3)))             # rank only: Array3[Any, Any, Any]
    reveal_type(jnp.zeros((640, 480, 3))[None, ...])  # Any
    reveal_type(template.shape)                       # Tuple[int, int, int]
    reveal_type(jnp.zeros_like(template))             # Array3[Channels, Height, Width]


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)
_TYPED_BUGGY = "return jnp.transpose(jnp.zeros_like(template), (2, 1, 0))"


def _line_of(fragment: str) -> int:
    return next(i for i, line in enumerate(_SOURCE.read_text().splitlines(), 1)
                if fragment in line)


def _clean(msg: str) -> str:
    return msg.replace("tensor_annotations.jax.", "").replace("tensor_annotations.axes.", "")


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
    """Only the typed rewrite is rejected. mypy reports it against the *arguments*: it
    infers the nested calls from the expected return type (`Letterbox`), so it expects
    `zeros_like` to produce [Height, Width, Channels] and the permutation to be (0, 1, 2)."""
    line = _line_of(_TYPED_BUGGY)
    assert _mypy("error") == [
        (line, 'Argument 1 to "zeros_like" has incompatible type '
               '"Array3[float32, Channels, Height, Width]"; expected '
               '"Array3[float32, Height, Width, Channels]"  [arg-type]'),
        (line, 'Argument 2 to "transpose" has incompatible type '
               '"tuple[Literal[2], Literal[1], Literal[0]]"; expected '
               '"tuple[Literal[0], Literal[1], Literal[2]]"  [arg-type]'),
    ]
    assert [m for _, m in _mypy("note")] == [
        'Revealed type is "Array3[float32, Any, Any, Any]"',
        'Revealed type is "Any"',
        'Revealed type is "tuple[int, int, int]"',
        'Revealed type is "Array3[float32, Channels, Height, Width]"',
    ]


def test_pyright(tmp_path):
    assert _pyright(tmp_path, "error") == [
        (_line_of(_TYPED_BUGGY),
         'Type "Array3[float32, Width, Height, Channels]" is not assignable to return type '
         '"Letterbox"'),
    ]
    assert [m for _, m in _pyright(tmp_path, "information")] == [
        'Type of "jnp.zeros((640, 480, 3))" is "Array3[float32, Any, Any, Any]"',
        'Type of "jnp.zeros((640, 480, 3))[None, ...]" is "Any"',
        'Type of "template.shape" is "Tuple[int, int, int]"',
        'Type of "jnp.zeros_like(template)" is "Array3[float32, Channels, Height, Width]"',
    ]


def test_run_time_rectangular_model_gets_a_transposed_input():
    """No run-time check: the buggy input is (1, 3, 480, 640) for a 640x480 (HxW) model."""
    assert preprocess_buggy(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 480, 640)
    assert preprocess_buggy_annotated(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 480, 640)
    assert preprocess_fixed(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 640, 480)
    template = jnp.zeros((3, 640, 480))
    assert preprocess_from_template_buggy(IMAGE, template).shape == (1, 3, 480, 640)
    assert letterbox_typed_buggy(template).shape == (480, 640, 3)
    assert letterbox_typed_fixed(template).shape == (640, 480, 3)


def test_run_time_square_model_hides_it():
    assert (preprocess_buggy(IMAGE, ONNX_SHAPE_SQUARE).shape
            == preprocess_fixed(IMAGE, ONNX_SHAPE_SQUARE).shape)
