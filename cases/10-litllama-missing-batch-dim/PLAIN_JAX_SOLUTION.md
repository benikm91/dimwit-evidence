# Plain JAX — verdict

**Category: `missed`**

Both ranks trace and both jit. The interesting line is
`test_vmap_is_the_thing_that_would_have_forced_the_question`: `jax.vmap(model)` on a
`[1, seq]` input gives a *different answer* from `model` on the same array, because under
`vmap` the centring step sees one example at a time.

That is the whole design argument in one assertion. `vmap` makes the batch axis external
and explicit; calling the model directly leaves it implicit and available to be confused
with the sequence axis. JAX offers `vmap` but does not require it, so the convention is
optional and the confusion remains reachable.
