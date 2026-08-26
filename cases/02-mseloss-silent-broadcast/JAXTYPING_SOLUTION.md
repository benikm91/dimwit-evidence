# jaxtyping — verdict

**Category: `run-time detection` if the loss is annotated `"batch"` on both arguments;
`missed` if it is annotated to match the model that was actually written.**

Both outcomes are in `jaxtyping_case.py`, and the paper should report both, because the
difference is not about jaxtyping's power but about who writes the annotation and when.

* `mse_strict(pred: Float[Array, "batch"], target: Float[Array, "batch"])` rejects the
  `[100, 1]` prediction at run time. This is a real win over plain JAX.
* `mse_as_written_upstream(pred: Float[Array, "batch out"], target: Float[Array, "batch"])`
  is the signature the buggy code actually had — `nn.Linear(2, 1)` really does return
  `[batch, 1]`. The annotation is *true*. The body is still wrong, and the scalar return
  annotation is satisfied. **Missed.**

The structural limitation: jaxtyping checks function boundaries. The defect is an
expression in the middle of a body, and no boundary annotation constrains it.
