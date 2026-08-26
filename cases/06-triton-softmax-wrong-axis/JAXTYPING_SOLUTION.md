# jaxtyping — verdict

**Category: `run-time detection` at minimal scope; `missed` at the scope upstream used**

Both are in `jaxtyping_case.py`, and the difference is not about jaxtyping's reach but about
how the function is scoped.

**At the upstream scope — `softmax(x, dim)` over a tile of any rank — it is missed.** The
reported `2x2` tile satisfies `Float[Array, "row col"]` under either behaviour, and softmax
keeps its input's shape, so the return annotation is satisfied too. The corrupted
intermediate lives between the reduction and the subtraction, inside the body, where no
boundary annotation reaches. The non-square tile does raise, but that is `jnp` refusing to
broadcast *inside* the body, not the type checker rejecting anything at the edge.

**At minimal scope — `softmax_vector(v: Float[Array, "n"]) -> Float[Array, "n"]` — the
defect is gone and the scope is enforced.** With one axis there is no `dim` argument, and
`keepdims=False` is harmless because a scalar returning to a vector has one axis to land on.
What remains is the misuse: handing a tile to the vector function. `Float[Array, "n"]` is
rank 1, so that raises — `test_a_tile_handed_to_the_vector_function_is_rejected`.

The honest reading for the paper is that the upgrade comes from the *scope*, not from
jaxtyping: minimal scope removes the fault in any of these systems. What the type system
decides is whether the scope is enforceable at all, and when.

* `plain.py` and `plain_jax.py` define the same `softmax_vector`, and there the scope is a
  docstring: `test_but_nothing_stops_a_tile_going_into_the_vector_function` passes a `2x2`
  tile in and gets silently wrong numbers back.
* jaxtyping enforces it on the executed call, at run time.
* DimWit enforces it at compile time, and `keep_dims` is not merely wrong there but
  inexpressible — see `DIMWIT_SOLUTION.md`.
