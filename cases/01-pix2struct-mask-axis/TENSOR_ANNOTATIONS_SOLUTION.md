# tensor_annotations — verdict

**Category: `missed` (code as written) / `compile`-equivalent static rejection (typed rewrite)**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) (DeepMind,
v2.0.3, with `tensor_annotations_jax_stubs` and mypy) types an array by its axes,
`Array4[float32, Batch, Heads, Query, Key]`, and checks it **statically**. Axes are
distinct `NewType`s, so — unlike jaxtyping — `Query` and `Key` stay different even when
their extents are equal. The self-attention coincidence that defeats jaxtyping does not
matter here.

* **The upstream code is missed.** `Array.__getitem__` is typed `-> Any`, and the stubs
  have no `expand_dims`. `additive[:, None, :, None]` therefore produces `Any`, which
  `+` and the return annotation accept without complaint. Annotating the intermediate
  does not help: `expanded: Scores = additive[:, None, :, None]` is false and accepted,
  because anything may be assigned from `Any`. There is no run-time component, so the
  program then runs exactly like plain JAX and attends to padding
  (`tensor_annotations_case.py::test_buggy_attends_to_padding`).
* **A typed rewrite is checked.** The stubs' only typed broadcast is trailing-axis:
  `Array4[.., A3, A4] + Array2[A3, A4]`. If `scores` is permuted to
  `[Heads, Query, Batch, Key]` with the typed `jnp.transpose`, the `[Batch, Key]` mask can
  be added by name (`attention_fixed_typed`). The upstream mistake re-expressed in that
  vocabulary, aligning the mask with `Query` (`attention_buggy_typed`), is rejected by
  mypy before anything runs:

  ```
  tensor_annotations_case.py:72: error: Unsupported operand types for +
    ("Array4[float32, Heads, Key, Batch, Query]" and "Array2[Any, Batch, Key]")  [operator]
  ```

The rejection comes from the rewrite, not from the code that was shipped. That is the
same caveat recorded for DimWit in case 04: catching a *different* program is a different
claim from catching the code that was written. tensor_annotations has the right model
(axes as types, compared by identity, before run time). What it misses here is coverage:
positional indexing is the operation the defect lives in, and it falls back to `Any`.

Minor friction observed: the stubs define no reflected operators, so the upstream
`(1.0 - mask) * NEG` is itself a type error and was rewritten as `(mask - 1.0) * 1e9`.
