"""Case 11 under jaxtyping — `plain_jax.py` with the axis names written into annotations.

The annotation names the axes, which looks like exactly the right tool — and it is, right up
until height, width and channels all equal 3, which is the configuration that shipped.
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

RNG = np.random.default_rng(11)


def decode_nchw(n_frames=2, channels=3, height=3, width=3) -> Float[Array, "frame channel height width"]:
    """The PyNvVideoCodec backend: [frames, channels, height, width]."""
    return jnp.asarray(RNG.normal(size=(n_frames, channels, height, width)), dtype=jnp.float32)


def decode_nhwc(n_frames=2, channels=3, height=3, width=3) -> Float[Array, "frame height width channel"]:
    """The other backend: [frames, height, width, channels]."""
    return jnp.asarray(RNG.normal(size=(n_frames, height, width, channels)), dtype=jnp.float32)


@jaxtyped(typechecker=beartype)
def per_channel_mean_nhwc(
    frames: Float[Array, "frame height width channel"],
) -> Float[Array, "channel"]:
    """Written for NHWC: the channel axis is last, so reduce over frames, rows, columns."""
    return frames.mean(axis=(0, 1, 2))


@jaxtyped(typechecker=beartype)
def to_nhwc(
    frames_nchw: Float[Array, "frame channel height width"],
) -> Float[Array, "frame height width channel"]:
    """The fix: transpose the backend's output into the layout the pipeline expects."""
    return jnp.transpose(frames_nchw, (0, 2, 3, 1))


# --------------------------------------------------------------------------- tests

def test_square_rgb_frames_slip_through():
    """MISSED. [2, 3, 3, 3] satisfies "frame height width channel" under either layout."""
    assert per_channel_mean_nhwc(decode_nchw()).shape == (3,)


def test_the_annotation_cannot_state_which_axis_is_the_channel():
    """`channel` is a size variable. It binds to whatever is in that position."""
    assert per_channel_mean_nhwc(jnp.zeros((2, 3, 4, 5))).shape == (5,)
    assert per_channel_mean_nhwc(jnp.zeros((2, 5, 4, 3))).shape == (3,)


def test_the_statistics_still_differ():
    nchw = decode_nchw()
    assert not np.allclose(
        np.asarray(per_channel_mean_nhwc(nchw)), np.asarray(per_channel_mean_nhwc(to_nhwc(nchw)))
    )
