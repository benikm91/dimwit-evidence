# tensor_annotations — verdict

**Category: `compile`-equivalent (static rejection), for both signatures — strict and as
written upstream.**

*Supplementary. This column is not part of the results table in the top-level README and
is not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with **both mypy and pyright**, because they
disagree on one line (below). The model is typed as well: `x @ w` with
`x: Array2[Batch, Features]` and `w: Array2[Features, Out]` is inferred as
`Array2[Batch, Out]`, so the `[n, 1]` prediction carries its `Out` axis by type.

* **Strict loss, `mse_strict(pred: Array1[Batch], target: Array1[Batch])`.** Passing the
  model output is rejected at the call site by both checkers — statically, where jaxtyping
  only raises at run time.
* **Loss as written upstream, `pred: Array2[Batch, Out]`.** The annotation is *true*, and
  the body is still rejected by both checkers:

  ```
  mypy:    Unsupported operand types for - ("Array2[float32, Batch, Out]" and "Array1[float32, Batch]")
  pyright: Operator "-" not supported for types "Prediction" and "Target"
  ```

  This is where it differs from jaxtyping, which misses this signature. jaxtyping checks
  only the function boundary; tensor_annotations types the expression `pred - target`.
  The stubs allow `Array2[A1, A2] - Array1[A2]` only (trailing-axis broadcast, compared by
  name), so `[Batch, Out] - [Batch]` needs `Out == Batch` and fails.

**Why it is caught, precisely.** The catch relies on the stubs *not* modelling size-1
broadcasting — the same design decision that made case 01 untypeable ("We deliberately
ignore case b) [one of them is 1] for the time being", `tensor_annotations/jax.pyi`).
Here that omission helps: the `Out` axis has size 1 at run time, and real broadcasting
treats it as `[n, 1] - [n] -> [n, n]`. The stubs treat it as a named axis that must
match, which is the stricter and here the correct rule.

**The fix leaves the type system unless written with a typed op.**

* `jnp.squeeze(pred, axis=1)` and `pred[:, 0]` — the usual fixes — are `Any` in the stubs
  (`squeeze: Any`, `__getitem__ -> Any`). Accepted, but unchecked.
* `jnp.mean(pred, axis=1)` (reducing the size-1 `Out` axis) is typed: `Array1[Batch]`.
  Reducing the wrong axis gives `Array1[Out]`, and here **the checkers disagree**:
  pyright rejects `mse_strict(jnp.mean(pred, axis=0), t)`; mypy accepts it. With an
  expected argument type in context, mypy resolves `mean` to its untyped fallback overload
  (`-> Any`). Bound to a variable first, mypy infers `Array1[Out]` and rejects it as well.
  The axis is still chosen by position (`axis=0`); only the *result* is checked.

There is no run-time component: every rejected function still runs and returns a wrong,
positive loss for weights that fit exactly (`test_annotations_are_not_enforced_at_run_time`).

Both checks presuppose that the prediction is typed `[Batch, Out]` from the start. If the
model output were `Any` — e.g. produced through any of the many `Any`-typed JAX functions
(`dot`, `matmul`, `reshape`, …) — nothing would be checked; here `@` happens to be typed.
