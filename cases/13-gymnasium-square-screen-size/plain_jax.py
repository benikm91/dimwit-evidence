"""Case 14 in JAX — the same boundary as `plain.py`.

Nothing changes. `jax.eval_shape` will tell you the output is (64, 84) with complete
confidence; it has no way to know that (84, 64) was what the caller asked for. Shape
inference checks that a program computes the shape it claims, not that the claim was right.
"""

import jax
import jax.numpy as jnp
import numpy as np

ATARI_FRAME = (210, 160)


def inner_resize(frame, dsize):
    """The `cv2.resize` half of the boundary: `dsize` is (width, height)."""
    out_w, out_h = dsize
    in_h, in_w = frame.shape
    rows = jnp.clip(jnp.arange(out_h) * in_h // out_h, 0, in_h - 1)
    cols = jnp.clip(jnp.arange(out_w) * in_w // out_w, 0, in_w - 1)
    return frame[rows][:, cols]


def resize_buggy(frame, size_hw):
    """As shipped: advertises (height, width) and forwards it unflipped."""
    return inner_resize(frame, size_hw)


def resize_fixed(frame, size_hw):
    """After PR #1312: the extents are flipped at the boundary."""
    height, width = size_hw
    return inner_resize(frame, (width, height))


def _frame():
    return jnp.asarray(
        np.arange(ATARI_FRAME[0] * ATARI_FRAME[1], dtype=np.float32).reshape(ATARI_FRAME)
    )


# --------------------------------------------------------------------------- tests

def test_the_swap_is_not_caught():
    """MISSED. Tracing completes; the result is simply the wrong way round."""
    assert resize_buggy(_frame(), (84, 64)).shape == (64, 84)


def test_the_fixed_version_honours_the_advertised_order():
    assert resize_fixed(_frame(), (84, 64)).shape == (84, 64)


def test_a_square_target_hides_the_defect_entirely():
    assert resize_buggy(_frame(), (84, 84)).shape == resize_fixed(_frame(), (84, 84)).shape


def test_shape_inference_confirms_the_transposed_output():
    """JAX-specific: `eval_shape` proves the program is self-consistent, and it is —
    consistently transposed. A checker cannot supply the intent the caller never wrote down.
    """
    traced = jax.eval_shape(lambda f: resize_buggy(f, (84, 64)), _frame())
    assert traced.shape == (64, 84)


def test_jit_compiles_the_defect_without_complaint():
    """The swap survives staging out: it is a legal program, just not the intended one."""
    compiled = jax.jit(lambda f: resize_buggy(f, (84, 64)))
    assert compiled(_frame()).shape == (64, 84)
