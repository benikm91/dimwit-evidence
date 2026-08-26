"""Case 05 under jaxtyping — `plain_jax.py` with as much typing as jaxtyping can express.

`onnx_input_shape` is a tuple of ints, and jaxtyping binds axis names only from *array*
annotations, so the two spatial entries are indistinguishable and the return annotation
constrains nothing. The extra pair at the bottom is not part of the ladder: it carries the
same layout in an *array* instead, which is the only way to make jaxtyping see it at all.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

ONNX_SHAPE_RECT = (1, 3, 640, 480)     # N, C, H = 640, W = 480
ONNX_SHAPE_SQUARE = (1, 3, 640, 640)   # YOLO's default export

IMAGE = jnp.zeros((1080, 1920, 3), dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
def preprocess_buggy(
    image: Float[Array, "orig_h orig_w channel"],
    onnx_input_shape: tuple[int, int, int, int],
) -> Float[Array, "1 channel height width"]:
    """As shipped. `input_shape` is (N, C, H, W); the two spatial entries are swapped."""
    input_width = onnx_input_shape[2]   # actually the height
    input_height = onnx_input_shape[3]  # actually the width
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


@jaxtyped(typechecker=beartype)
def preprocess_fixed(
    image: Float[Array, "orig_h orig_w channel"],
    onnx_input_shape: tuple[int, int, int, int],
) -> Float[Array, "1 channel height width"]:
    """After PR #23402."""
    input_height = onnx_input_shape[2]
    input_width = onnx_input_shape[3]
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


# --------------------------------------------------------------------------- tests

def test_the_annotation_is_satisfied_by_the_buggy_version():
    """MISSED, even for a rectangular model. `height` and `width` are free variables in the
    return annotation, so they bind to whatever comes back."""
    assert preprocess_buggy(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 480, 640)


def test_the_shape_tuple_carries_no_axis_identity():
    """`tuple[int, int, int, int]` says nothing about which entry is the height."""
    assert preprocess_fixed(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 640, 480)


def test_the_bug_is_invisible_on_a_square_model_input():
    assert (
        preprocess_buggy(IMAGE, ONNX_SHAPE_SQUARE).shape
        == preprocess_fixed(IMAGE, ONNX_SHAPE_SQUARE).shape
    )


# ----------------------------------------------- not the ladder: the layout as an array
#
# The only way jaxtyping can see this defect is if the model's declared layout arrives as
# an annotated array, so that `height` and `width` are *bound* rather than free.

TEMPLATE_RECT = jnp.zeros((3, 640, 480), dtype=jnp.float32)
TEMPLATE_SQUARE = jnp.zeros((3, 640, 640), dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
def preprocess_from_template_buggy(
    image: Float[Array, "orig_h orig_w channel"],
    template: Float[Array, "channel height width"],
) -> Float[Array, "1 channel height width"]:
    input_width, input_height = template.shape[1], template.shape[2]
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


def test_a_template_array_does_bind_them_and_the_swap_is_caught():
    """RUN-TIME detection — but only once the layout is carried by an array, not four ints."""
    with pytest.raises(Exception):
        preprocess_from_template_buggy(IMAGE, TEMPLATE_RECT)


def test_even_with_a_template_a_square_model_hides_it():
    """MISSED for the default 640x640 export: `height` and `width` bind to the same number."""
    assert preprocess_from_template_buggy(IMAGE, TEMPLATE_SQUARE).shape == (1, 3, 640, 640)
