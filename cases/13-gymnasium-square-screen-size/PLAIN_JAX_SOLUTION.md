# Plain JAX — verdict

**Category: `missed`**

Identical to NumPy, with one addition worth stating plainly in the paper: JAX's shape
machinery does not merely fail to catch this, it *certifies* it.

`test_shape_inference_confirms_the_transposed_output` traces `resize_buggy` with
`jax.eval_shape` and gets `(64, 84)` — correct, complete, and useless. Shape inference
verifies that a program computes the shape it claims to compute. It has no access to the
claim the caller actually made, because that claim lived in the ordering convention of a
tuple and was never written down anywhere the compiler could read.

`test_jit_compiles_the_defect_without_complaint` makes the same point at the next stage: the
swap survives being staged out and compiled. It is a legal program throughout. Every tool in
this comparison, DimWit included, is a consistency checker rather than an oracle — the
difference between them is only how much of the intent they give you a place to write down.
