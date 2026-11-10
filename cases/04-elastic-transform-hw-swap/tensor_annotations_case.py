"""Case 04 under tensor_annotations (google-deepmind/tensor_annotations) + mypy / pyright.

The defect is a scalar divisor: `dx * alpha[0] / size[0]` with `size = (H, W)`.
tensor_annotations types arrays by their axes; it has no type for a single extent.

*  as written upstream (`size: tuple[int, int]`)                 -> MISSED
   `jax.random.uniform` and `jnp.stack` are `Any` in the stubs, so not even the return
   annotation is checked against anything.
*  most favourable typing: `dx` annotated `Array2[float32, Height, Width]` and `size` read
   from a typed image                                             -> MISSED
   `image.shape` is `Tuple[int, int]` -- the stubs do not link shape entries to axes -- and
   `Array2[Height, Width] / int` is `Array2[Height, Width]` whichever int it is. The type
   of the result cannot depend on which number was divided by.

Expected to PASS both checkers with zero errors; the tests assert exactly that, plus a
control showing what the checkers infer (so the result cannot come from a broken setup).
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
from tensor_annotations.jax import Array2, Array3, float32

Height = axes.Height
Width = axes.Width
Components = NewType("Components", axes.Axis)     # (dx, dy)

Image = Array2[float32, Height, Width]
Displacement = Array3[float32, Height, Width, Components]
Field = Array2[float32, Height, Width]

KEY = jax.random.key(7)


# ------------------------------------------------------------------ as written upstream

def displacement_buggy(alpha: tuple[float, float], size: tuple[int, int]) -> Displacement:
    """As shipped. `size` is (height, width); dx is divided by size[0] = height."""
    height, width = size
    dx = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dx = dx * alpha[0] / size[0]  # accidentally normalised by height
    dy = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dy = dy * alpha[1] / size[1]  # accidentally normalised by width
    return jnp.stack([dx, dy], axis=-1)


def displacement_fixed(alpha: tuple[float, float], size: tuple[int, int]) -> Displacement:
    """After PR #9300: horizontal by width, vertical by height."""
    height, width = size
    dx = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dx = dx * alpha[0] / size[1]
    dy = jax.random.uniform(KEY, (height, width), minval=-1.0, maxval=1.0)
    dy = dy * alpha[1] / size[0]
    return jnp.stack([dx, dy], axis=-1)


# ------------------------------------------- most favourable: typed fields, typed image

def displacement_from_image_buggy(alpha: tuple[float, float], image: Image) -> Displacement:
    """`size` comes from an image whose axes ARE typed -- but `.shape` is Tuple[int, int]."""
    size = image.shape
    dx: Field = jax.random.uniform(KEY, size, minval=-1.0, maxval=1.0)
    dy: Field = jax.random.uniform(KEY, size, minval=-1.0, maxval=1.0)
    dx = dx * alpha[0] / size[0]  # still Field: the divisor is an int either way
    dy = dy * alpha[1] / size[1]
    return jnp.stack([dx, dy], axis=-1)


def displacement_from_image_fixed(alpha: tuple[float, float], image: Image) -> Displacement:
    size = image.shape
    dx: Field = jax.random.uniform(KEY, size, minval=-1.0, maxval=1.0)
    dy: Field = jax.random.uniform(KEY, size, minval=-1.0, maxval=1.0)
    dx = dx * alpha[0] / size[1]
    dy = dy * alpha[1] / size[0]
    return jnp.stack([dx, dy], axis=-1)


def _control(image: Image, dx: Field) -> None:
    """Never called. What the checkers infer at the points that matter."""
    reveal_type(image.shape)                          # Tuple[int, int]: no axis link
    reveal_type(dx / image.shape[0])                  # Array2[Height, Width] ...
    reveal_type(dx / image.shape[1])                  # ... the same type
    reveal_type(jax.random.uniform(KEY, (2, 3)))      # Any
    reveal_type(jnp.stack([dx, dx], axis=-1))         # Any


# --------------------------------------------------------------------------- tests

_SOURCE = Path(__file__)


def _mypy(severity: str) -> list[str]:
    from mypy import api  # pyright: ignore[reportMissingImports]  (only in the check env)
    stdout, _, _ = api.run(["--no-incremental", "--no-error-summary", str(_SOURCE)])
    found = re.findall(rf"^.*?:\d+: {severity}: (.*)$", stdout, re.MULTILINE)
    return [m.replace("tensor_annotations.jax.", "").replace("tensor_annotations.axes.", "")
            for m in found]


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
    """MISSED statically, in both the verbatim and the most favourable typing."""
    assert _mypy("error") == []
    assert _mypy("note") == [
        'Revealed type is "tuple[int, int]"',
        'Revealed type is "Array2[Any, Height, Width]"',   # dtype lost by `/ int`; axes kept
        'Revealed type is "Array2[Any, Height, Width]"',
        'Revealed type is "Any"',
        'Revealed type is "Any"',
    ]


def test_pyright_accepts_everything(tmp_path):
    assert _pyright(tmp_path, "error") == []
    assert _pyright(tmp_path, "information") == [
        'Type of "image.shape" is "Tuple[int, int]"',
        'Type of "dx / image.shape[0]" is "Array2[Any, Height, Width]"',
        'Type of "dx / image.shape[1]" is "Array2[Any, Height, Width]"',
        'Type of "jax.random.uniform(KEY, (2, 3))" is "Any"',
        'Type of "jnp.stack([dx, dx], axis=-1)" is "Any"',
    ]


def test_same_shapes_wrong_values_on_non_square_images():
    """Run time: no check. 100x800: horizontal displacement eight times too large."""
    alpha, size = (50.0, 50.0), (100, 800)
    buggy, fixed = displacement_buggy(alpha, size), displacement_fixed(alpha, size)
    assert buggy.shape == fixed.shape == (100, 800, 2)
    assert np.allclose(np.asarray(buggy[..., 0]) / np.asarray(fixed[..., 0]), 8.0)
    assert np.allclose(np.asarray(buggy[..., 1]) / np.asarray(fixed[..., 1]), 1 / 8)


def test_from_image_variant_behaves_the_same():
    alpha, image = (50.0, 50.0), jnp.zeros((100, 800))
    assert np.allclose(np.asarray(displacement_from_image_buggy(alpha, image)),
                       np.asarray(displacement_buggy(alpha, (100, 800))))
    assert np.allclose(np.asarray(displacement_from_image_fixed(alpha, image)),
                       np.asarray(displacement_fixed(alpha, (100, 800))))


def test_invisible_on_square_images():
    alpha, size = (50.0, 50.0), (256, 256)
    assert np.allclose(np.asarray(displacement_buggy(alpha, size)),
                       np.asarray(displacement_fixed(alpha, size)))
