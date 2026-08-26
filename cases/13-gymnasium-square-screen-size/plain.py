"""Case 14 — a (width, height) API meeting a (height, width) API (NumPy).

Original: Farama-Foundation/Gymnasium#1312, "Allow `AtariPreprocessing` non-square
observations".

`AtariPreprocessing` passes `screen_size` straight into `cv2.resize`, whose `dsize` is
ordered **(width, height)**; the array that comes back is ordered **(height, width)**. The
wrapper then declared its observation space in the first order while producing the second:

    _shape = self.screen_size + (1 if grayscale_obs else 3,)      # (W, H, C)  -- wrong
    _shape = (self.screen_size[1], self.screen_size[0], ...)      # (H, W, C)  -- the fix

Below, that boundary is distilled to its two halves: `inner_resize` speaks (width, height)
like `cv2.resize`, `resize_*` speaks (height, width) like the rest of the pipeline. The
defect is a pair of extents crossing between them without being flipped.

It shipped because `screen_size` had always been square, and (s, s) is the one input on
which the two conventions agree.
"""

import numpy as np

ATARI_FRAME = (210, 160)


def inner_resize(frame, dsize):
    """The `cv2.resize` half of the boundary.

    `dsize` is **(width, height)**, and the returned array is **(height, width)**. Both
    conventions are correct; they are simply not the same one.
    """
    out_w, out_h = dsize
    in_h, in_w = frame.shape
    rows = (np.arange(out_h) * in_h // out_h).clip(0, in_h - 1)
    cols = (np.arange(out_w) * in_w // out_w).clip(0, in_w - 1)
    return frame[np.ix_(rows, cols)]


def resize_buggy(frame, size_hw):
    """As shipped: advertises (height, width) and forwards it unflipped."""
    return inner_resize(frame, size_hw)


def resize_fixed(frame, size_hw):
    """After PR #1312: the extents are flipped at the boundary."""
    height, width = size_hw
    return inner_resize(frame, (width, height))


def _frame():
    return np.arange(ATARI_FRAME[0] * ATARI_FRAME[1], dtype=float).reshape(ATARI_FRAME)


# --------------------------------------------------------------------------- tests

def test_the_swap_is_not_caught():
    """MISSED. No exception, no warning: a well-formed frame of the wrong orientation."""
    assert resize_buggy(_frame(), (84, 64)).shape == (64, 84)


def test_the_fixed_version_honours_the_advertised_order():
    assert resize_fixed(_frame(), (84, 64)).shape == (84, 64)


def test_a_square_target_hides_the_defect_entirely():
    """Why it shipped: (s, s) is the fixed point of the two conventions."""
    square_buggy = resize_buggy(_frame(), (84, 84))
    square_fixed = resize_fixed(_frame(), (84, 84))
    assert square_buggy.shape == square_fixed.shape == (84, 84)
    assert np.array_equal(square_buggy, square_fixed)


def test_both_outputs_are_valid_2d_frames():
    """Nothing here is a structural error. Both results are usable arrays."""
    buggy, fixed = resize_buggy(_frame(), (84, 64)), resize_fixed(_frame(), (84, 64))
    assert buggy.ndim == fixed.ndim == 2
    assert buggy.size == fixed.size


def test_the_declared_shape_and_the_actual_array_disagree():
    """The upstream symptom: `observation_space.shape` promised what was never produced."""
    declared = (84, 64)
    actual = resize_buggy(_frame(), declared).shape
    assert actual != declared and actual == declared[::-1]
