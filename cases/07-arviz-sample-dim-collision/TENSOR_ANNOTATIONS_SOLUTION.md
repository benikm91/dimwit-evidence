# tensor_annotations — verdict

**Category: `missed` — stacking is not a typed operation: `jnp.reshape` is `Any` in the
stubs, so the stacked axis is never derived from `Chain` and `Draw`.**

*Supplementary. This column is not part of the results table in the top-level README and
is not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

tensor_annotations replaces string dims with axis types, so the upstream collision becomes:
arviz's stacked axis is given the same type as the user's `Sample` axis.

* **Stacking is untyped.** Flattening `[Chain, Draw, A]` into `[stacked, A]` needs
  `jnp.reshape`, declared `reshape: Any`. The stack function's return type
  (`Array2[Sample, A]` buggy, `Array2[StackedSample, A]` fixed) is a declaration the body
  never has to satisfy; any axes would be accepted.
* **The collision is then invisible.** For a user posterior `[Chain, Draw, Sample]`, the
  buggy stack is `Array2[Sample, Sample]`: two axes of one type. The downstream routine
  that averages "the first `Sample`" (upstream's selection by name picking the stacked
  axis) type-checks, and at run time returns one value per user sample (5) instead of one
  per stacked draw (12).

The tests assert zero errors in both checkers plus a control: `jnp.reshape(...)` is `Any`,
and the stack results have exactly their declared types.

**Compared with DimWit.** DimWit's flatten *derives* the stacked label —
`flatten((Axis[Chain], Axis[Draw]))` is `Chain |*| Draw` — so there is no name to choose and
nothing to collide. tensor_annotations has the other half of the answer: axis types are
compared by identity ("we verify equality of axis types based on the identity of the type",
`tensor_annotations/axes.py`), so two modules that each define their own `Sample` do not
clash the way two strings do. But without a typed flatten, the stacked axis can only be
declared, and a declaration can collide.
