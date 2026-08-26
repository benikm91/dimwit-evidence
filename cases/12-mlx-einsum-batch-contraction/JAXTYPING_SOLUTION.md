# jaxtyping — verdict

**Category: `missed` (square) / `run-time detection` (rectangular)**

With `a: Float[Array, "batch row inner"]` and `b: Float[Array, "inner col"]`, jaxtyping
cross-references `inner` between the two arguments and `row`/`col` into the return type.
That is enough to reject the buggy contraction on a `(2, 5, 3)` input.

On the `(3, 3, 3)` input every name binds to 3, the output is `(3, 3, 4)` either way, and
the annotation is satisfied. This is the same conditional win as cases 05 and 08: helpful
exactly when the extents differ, silent exactly when they do not.
