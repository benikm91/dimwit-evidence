# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

`size[0]` and `size[1]` are both `int`. The expression `dx * alpha[0] / size[0]` is
well-typed, well-shaped and produces exactly the array the correct version produces —
scaled by the wrong constant.

Two facts from `plain.py` are worth reporting:

* `test_the_bug_is_invisible_on_square_images` — on any square input the two versions are
  bit-identical. Square test fixtures are the norm in vision code, which is why this
  shipped in both `transforms` and `transforms.v2`.
* `test_non_square_horizontal_displacement_is_eight_times_too_large` — on the reporter's
  100x800 image the horizontal displacement is 8x too large and the vertical 8x too small.

No shape checker of any kind can see this, because by the time the number reaches the
arithmetic it has stopped being an axis and become an integer.
