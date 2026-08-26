"""Case 14 under jaxtyping — `plain_jax.py` with the axis names written into annotations.

The names are present and still do not bind. jaxtyping binds a variable to an array's axis,
never to a Python `int`, so `width` and `height` — which arrive as plain integers inside a
tuple — carry no name to check against. `Float[Array, "out_h out_w"]` is satisfied by any
two-dimensional result, transposed or not.

Annotating harder does not rescue it. The one annotation that would flag `resize_buggy` is
`Float[Array, "h w"] -> Float[Array, "w h"]` on `inner_resize`, and that is a lie: the two
axes are genuinely independent, and every square target satisfies both readings at once.
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

ATARI_FRAME = (210, 160)


@jaxtyped(typechecker=beartype)
def inner_resize(
    frame: Float[Array, "in_h in_w"],
    dsize: tuple[int, int],
) -> Float[Array, "out_h out_w"]:
    """The `cv2.resize` half of the boundary: `dsize` is (width, height).

    Note what the return annotation can and cannot say. `out_h` and `out_w` are fresh
    variables bound by whatever comes back, so they record the result rather than constrain
    it. Nothing ties either of them to an element of `dsize`.
    """
    out_w, out_h = dsize
    in_h, in_w = frame.shape
    rows = jnp.clip(jnp.arange(out_h) * in_h // out_h, 0, in_h - 1)
    cols = jnp.clip(jnp.arange(out_w) * in_w // out_w, 0, in_w - 1)
    return frame[rows][:, cols]


@jaxtyped(typechecker=beartype)
def resize_buggy(
    frame: Float[Array, "in_h in_w"],
    size_hw: tuple[int, int],
) -> Float[Array, "out_h out_w"]:
    """As shipped: advertises (height, width) and forwards it unflipped."""
    return inner_resize(frame, size_hw)


@jaxtyped(typechecker=beartype)
def resize_fixed(
    frame: Float[Array, "in_h in_w"],
    size_hw: tuple[int, int],
) -> Float[Array, "out_h out_w"]:
    """After PR #1312: the extents are flipped at the boundary."""
    height, width = size_hw
    return inner_resize(frame, (width, height))


def _frame():
    return jnp.asarray(
        np.arange(ATARI_FRAME[0] * ATARI_FRAME[1], dtype=np.float32).reshape(ATARI_FRAME)
    )


# --------------------------------------------------------------------------- tests

def test_the_swap_is_not_caught():
    """MISSED. Both annotations are satisfied; `out_h`/`out_w` bind to 64/84 and agree."""
    assert resize_buggy(_frame(), (84, 64)).shape == (64, 84)


def test_the_fixed_version_honours_the_advertised_order():
    assert resize_fixed(_frame(), (84, 64)).shape == (84, 64)


def test_the_two_functions_have_identical_signatures():
    """The correct and the incorrect wrapper are indistinguishable to the type checker."""
    def sig(fn):
        return {k: str(v) for k, v in fn.__annotations__.items()}

    assert sig(resize_buggy) == sig(resize_fixed)


def test_names_do_not_bind_to_integers():
    """The root cause: the extents arrive as `int`, and jaxtyping has no name for an int.

    Both calls type-check, so the annotation cannot tell the two orderings apart.
    """
    assert inner_resize(_frame(), (84, 64)).shape == (64, 84)
    assert inner_resize(_frame(), (64, 84)).shape == (84, 64)


def test_a_square_target_hides_the_defect_entirely():
    assert resize_buggy(_frame(), (84, 84)).shape == resize_fixed(_frame(), (84, 84)).shape
