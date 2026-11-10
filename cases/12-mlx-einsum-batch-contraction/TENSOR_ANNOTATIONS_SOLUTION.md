# tensor_annotations — verdict

**Category: `missed` — the contraction is not a typed operation: `jnp.tensordot` and
`jnp.einsum` are `Any` in the stubs.**

*Supplementary. This column is not part of the results table in the top-level README and
is not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The operands are typed `a: Array3[float32, Batch, Row, Inner]`, `b: Array2[float32, Inner,
Col]`, and the result declared `Array3[float32, Batch, Row, Col]`. The contraction between
them is written as upstream writes it, and the stubs type neither form:

```
tensordot: Any      # jax-stubs/numpy/__init__.pyi
einsum: Any
```

So contracting `row` (`axes=([1], [0])`, or einsum `"bri,rc->bic"`) instead of `inner` is
accepted, and the return annotation is satisfied by an `Any`. The tests assert zero errors
in both checkers, with a control showing typed operands and an `Any` result. At run time
the square `(3, 3, 3)` input returns wrong numbers; the rectangular one raises — inside
`jnp`, not at any annotation.

**`@` is typed, but is not this program.** `a @ b` infers `Array3[Batch, Row, Col]`
(`Array3.__matmul__` contracts the last axis of `a` with the first of `b`). It has no axis
argument, so the upstream mistake — choosing the wrong axis to contract — cannot be written
with it at all. Rewriting to `@` would remove the defect rather than detect it, and would
measure a different program.

**Compared with the other columns.** jaxtyping catches the rectangular case at run time by
cross-referencing `inner` between operands, and misses the square one. DimWit names the
contracted axis, which must exist in both operands, so `a.dot(Axis[Row])(b)` is not a
contraction at all (`b` has no `Row`). tensor_annotations would need a typed contraction
with the same property; the stubs have none.
