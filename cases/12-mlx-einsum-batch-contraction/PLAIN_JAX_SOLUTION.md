# Plain JAX — verdict

**Category: `missed` (square) / `run-time detection` (rectangular)**

Same as NumPy. `jax.eval_shape` reports `(3, 3, 4)` for both.

`test_vmap_is_the_thing_that_removes_the_choice` is the constructive half: writing
`jax.vmap(lambda m: m @ B)` reduces the operation to an ordinary matrix product, where there
is no axis to select and therefore no axis to select wrongly. JAX makes that style
available; nothing requires it, and the library code in the upstream bug did not use it.
