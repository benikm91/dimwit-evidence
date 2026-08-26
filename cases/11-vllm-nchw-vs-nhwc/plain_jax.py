"""Case 11 in JAX — the same program as `plain.py`."""

import jax
import jax.numpy as jnp
import numpy as np

RNG = np.random.default_rng(11)


def decode_nchw(n_frames=2, channels=3, height=3, width=3):
    """The PyNvVideoCodec backend: [frames, channels, height, width]."""
    return jnp.asarray(RNG.normal(size=(n_frames, channels, height, width)), dtype=jnp.float32)


def decode_nhwc(n_frames=2, channels=3, height=3, width=3):
    """The other backend: [frames, height, width, channels]."""
    return jnp.asarray(RNG.normal(size=(n_frames, height, width, channels)), dtype=jnp.float32)


def per_channel_mean_nhwc(frames):
    """Written for NHWC: the channel axis is last, so reduce over frames, rows, columns."""
    return frames.mean(axis=(0, 1, 2))


def to_nhwc(frames_nchw):
    """The fix: transpose the backend's output into the layout the pipeline expects."""
    return jnp.transpose(frames_nchw, (0, 2, 3, 1))


# --------------------------------------------------------------------------- tests

def test_the_two_layouts_are_indistinguishable_for_square_rgb_frames():
    """The reason it is silent: [2, 3, 3, 3] under either reading."""
    assert decode_nchw().shape == decode_nhwc().shape == (2, 3, 3, 3)


def test_per_channel_statistics_are_computed_over_the_wrong_axes():
    """Erroneous behaviour: three plausible numbers, none of them a channel mean."""
    nchw = decode_nchw()
    wrong = per_channel_mean_nhwc(nchw)
    right = per_channel_mean_nhwc(to_nhwc(nchw))
    assert wrong.shape == right.shape == (3,)
    assert not np.allclose(np.asarray(wrong), np.asarray(right))


def test_no_exception_is_raised_anywhere():
    assert per_channel_mean_nhwc(decode_nchw()).shape == (3,)


def test_a_non_square_frame_would_have_crashed_instead():
    """The defect is only silent while height == width == channels."""
    nchw = decode_nchw(height=4, width=5)
    assert nchw.shape == (2, 3, 4, 5)
    assert per_channel_mean_nhwc(nchw).shape == (5,)   # 5 "channels": obviously wrong
    assert per_channel_mean_nhwc(to_nhwc(nchw)).shape == (3,)


def test_shape_inference_accepts_the_wrong_layout():
    """JAX-specific: tracing proves well-formedness, and the layout is not part of that."""
    assert jax.eval_shape(per_channel_mean_nhwc, decode_nchw()).shape == (3,)
