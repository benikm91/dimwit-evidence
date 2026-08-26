# 04 — `ElasticTransform` normalised horizontal displacement by the height

**Source:** [pytorch/vision#9299](https://github.com/pytorch/vision/issues/9299)
· fix: [PR #9300](https://github.com/pytorch/vision/pull/9300)

## The defect

`v2.ElasticTransform` builds random displacement fields `dx` (horizontal) and `dy`
(vertical) and normalises them into `grid_sample`'s `[-1, 1]` coordinate space. `dx` must
be divided by the **width**, `dy` by the **height**. The code did the reverse:

```python
size = list(query_size(flat_inputs))   # (H, W)
dx = dx * self.alpha[0] / size[0]      # buggy: horizontal / height
dy = dy * self.alpha[1] / size[1]      # buggy: vertical / width
```

On a 100x800 image the horizontal displacement is divided by 100 instead of 800 — eight
times too large — while the vertical displacement is divided by 800 instead of 100.

## Why it is interesting

* **Invisible on square inputs.** Every unit test on a square image passes. The augmentation
  had been shipping wrong for both `transforms` and `transforms.v2`.
* **The defect is in a scalar, not a tensor.** `size[0]` and `size[1]` are plain `int`s. By
  the time the divisor reaches the arithmetic, every checker in this comparison — DimWit
  included — has nothing left to look at. This is the case that marks the boundary of the
  thesis: names as types govern *tensors*, and a shape entry that has been read out as a
  number is no longer one.
* **What is left is legibility.** DimWit has no positional shape access, so `size[0]` cannot
  be written and the axis has to be named. The mistake becomes "divide the horizontal
  displacement by `Axis[Height]`" instead of "index 0", which a reader can catch — but the
  compiler cannot. See `DIMWIT_SOLUTION.md`.
