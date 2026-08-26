# Plain JAX — verdict

**Category: `missed`**

`preprocess_buggy` returns `(1, 3, 480, 640)` and `preprocess_fixed` returns
`(1, 3, 640, 480)`. Both are valid NCHW batches; JAX has no way to know which one the ONNX
graph expects, because "NCHW" is a convention recorded in prose.

The failure would surface downstream, at the ONNX runtime boundary, as a shape error — but
only for a rectangular model, and only at inference time, and the example in question fed
the tensor into a runtime that resizes rather than rejects.
