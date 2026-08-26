"""Case 05 in JAX — the same program as `plain.py`.

Transposing the spatial axes of an image is a legal array operation here too.
"""

import jax.numpy as jnp

ONNX_SHAPE_RECT = (1, 3, 640, 480)     # N, C, H = 640, W = 480
ONNX_SHAPE_SQUARE = (1, 3, 640, 640)   # YOLO's default export

IMAGE = jnp.zeros((1080, 1920, 3), dtype=jnp.float32)


def preprocess_buggy(image, onnx_input_shape):
    """As shipped. `input_shape` is (N, C, H, W); the two spatial entries are swapped."""
    input_width = onnx_input_shape[2]   # actually the height
    input_height = onnx_input_shape[3]  # actually the width
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


def preprocess_fixed(image, onnx_input_shape):
    """After PR #23402."""
    input_height = onnx_input_shape[2]
    input_width = onnx_input_shape[3]
    letterboxed = jnp.zeros((input_height, input_width, image.shape[2]), dtype=image.dtype)
    return jnp.transpose(letterboxed, (2, 0, 1))[None, ...]


# --------------------------------------------------------------------------- tests

def test_both_produce_a_valid_nchw_batch():
    """The whole problem: nothing about the result says which one is right."""
    assert preprocess_buggy(IMAGE, ONNX_SHAPE_RECT).ndim == 4
    assert preprocess_fixed(IMAGE, ONNX_SHAPE_RECT).ndim == 4


def test_numpy_allocates_the_transposed_buffer_without_complaint():
    """No error: (480, 640, 3) is a perfectly good array."""
    assert preprocess_buggy(IMAGE, ONNX_SHAPE_RECT).size == 3 * 640 * 480


def test_the_spatial_axes_are_transposed():
    """Erroneous behaviour: (1, 3, 480, 640) where the model declared (1, 3, 640, 480)."""
    assert preprocess_buggy(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 480, 640)
    assert preprocess_fixed(IMAGE, ONNX_SHAPE_RECT).shape == (1, 3, 640, 480)


def test_the_bug_is_invisible_on_a_square_model_input():
    """YOLO's default imgsz is 640x640, so the defect never showed in the common path."""
    assert (
        preprocess_buggy(IMAGE, ONNX_SHAPE_SQUARE).shape
        == preprocess_fixed(IMAGE, ONNX_SHAPE_SQUARE).shape
    )


def test_the_aspect_ratio_of_the_model_input_is_inverted():
    """The concrete consequence: boxes are later rescaled with the axes swapped."""
    _, _, bh, bw = preprocess_buggy(IMAGE, ONNX_SHAPE_RECT).shape
    _, _, fh, fw = preprocess_fixed(IMAGE, ONNX_SHAPE_RECT).shape
    assert bw / bh == 640 / 480
    assert fw / fh == 480 / 640
