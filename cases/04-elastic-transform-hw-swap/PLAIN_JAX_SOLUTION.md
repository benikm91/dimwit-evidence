# Plain JAX — verdict

**Category: `missed`**

`jax.eval_shape` on the buggy expression returns `(100, 800)`, exactly as on the fixed one
(`test_shape_inference_has_nothing_to_say`). Abstract interpretation over shapes is
complete here and completely uninformative: the shapes were never wrong.

This case is a useful counterweight in the paper to the cases where JAX/NumPy broadcasting
is the villain. Here nothing broadcasts. The defect is a *scalar provenance* error, and
scalar provenance is exactly what a shape system is not tracking — unless, as in DimWit,
extents are values that carry their axis.
