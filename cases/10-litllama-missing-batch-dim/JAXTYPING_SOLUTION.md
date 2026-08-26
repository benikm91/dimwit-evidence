# jaxtyping — verdict

**Category: `run-time detection`**

`Int[Array, "batch seq"]` fixes the rank, so `model(IDX)` on a 1-D vector raises. A real
win, and one of the four rank cases (09, 10, 13, 15) where jaxtyping does well.

The residual gap is in `test_the_annotation_does_not_constrain_the_centring_step`: the
annotation guarantees a batch axis is *present*, not that the body treats it as the batch.
`x.mean(axis=0)` is still positional. Add a leading axis in a later refactor and the
annotation is updated, the check still passes, and the centring is silently wrong again.

jaxtyping polices the boundary. The defect that survives is one axis inside it.
