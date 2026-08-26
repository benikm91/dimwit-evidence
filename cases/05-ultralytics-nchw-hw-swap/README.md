# 05 — YOLO ONNX example read NCHW height and width in the wrong order

**Source:** [ultralytics/ultralytics#23126](https://github.com/ultralytics/ultralytics/issues/23126)
· fix: [PR #23402](https://github.com/ultralytics/ultralytics/pull/23402)

## The defect

ONNX models declare their input as NCHW, so `input_shape[2]` is the height and
`input_shape[3]` the width. The example did:

```python
self.input_width  = input_shape[2]   # actually the height
self.input_height = input_shape[3]   # actually the width
```

Those two values are then used to letterbox the image and to scale the predicted boxes back
to the original resolution, so every box coordinate is off by the aspect ratio.

## Why it is interesting

Same root cause as case 04 — an axis addressed by position — but the consequence propagates
through a *tensor layout* rather than a scalar. That gives DimWit a second, stronger place
to catch it: the letterboxed image itself is `Tensor3[Height, Width, Channel]`, and building
it with the extents swapped produces `Tensor3[Width, Height, Channel]`, which the model
signature rejects.

As in case 04, it is invisible whenever the model input is square — and YOLO's default
`imgsz` is 640x640.
