# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

The buggy and the fixed expression produce the same output shape, `[batch, heads, seq, seq]`.
NumPy broadcasting is happy to align a `[batch, 1, seq, 1]` operand against a
`[batch, heads, seq, seq]` one, and so is PyTorch. Nothing raises, nothing warns.

The defect surfaces only as *different numbers*: attention weight is placed on padding
positions. Detecting it requires someone to look at the values — which is exactly how it
was found upstream, months after release, and the fix had to rewrite the expected strings
in `test_modeling_pix2struct.py`.

The root cause is that the axis is selected by **position**, and a position carries no
meaning. `[:, None, :, None]` and `[:, None, None, :]` are equally well-formed programs;
only the author's intent distinguishes them, and intent is not written down anywhere the
machine can read.

See `plain.py::test_buggy_and_fixed_agree_on_shape`.
