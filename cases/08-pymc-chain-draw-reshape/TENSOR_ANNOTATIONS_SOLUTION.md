# tensor_annotations — verdict

**Category: `missed` — neither step is a typed operation: the transpose takes a run-time
permutation, and `reshape` is `Any`.**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The dataset is typed `Array3[float32, Team, Draw, Chain]` and the point list declared
`Array2[float32, ChainDraw, Team]`. Between them are the two steps of the upstream code,
and the stubs type neither:

1. **The transpose is computed from the dims at run time** (`list[int]`). The typed
   `transpose` overloads require a literal tuple (`axes: Tuple[L2, L1, L0]`, …), and the
   untyped fallback accepts only a plain `jax.Array`, not a typed `Array3`. So the call
   matches no overload and both checkers reject it. It is in the helper that buggy and
   fixed share, so it rejects both alike: a stub gap, not a detection. The result is `Any`.
2. **The flatten is `jnp.reshape`**, declared `reshape: Any`. Even after a literal (typed)
   permutation, the axes end here; and the sizes it is fed are `int`s, which carry no axes
   in any case (case 07: `reshape` cannot be typed by axis at all).

So the declared point-list type is never compared with what the body computes, and nothing
distinguishes the buggy reshape (sizes read before the transpose) from the fixed one.

The tests assert exactly the transpose errors (on the shared helper and on the control
line that repeats it) and a control of the inferred types: run-time permutation `Any`/
`Unknown`, literal permutation `Array3[float32, Chain, Draw, Team]`, `reshape` `Any`.

**Compared with the other columns.** jaxtyping catches this at run time when the two
rectangles differ, because it checks the *output sizes* against the input's. tensor_
annotations checks axis identity but never sees sizes, and here no axis identity survives
to the output. DimWit replaces `reshape` with `flatten`, whose result label records which
axes were fused and in which order (`Draw |*| Chain` vs `Chain |*| Draw`).
