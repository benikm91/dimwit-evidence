# Plain JAX — verdict

**Category: `missed`**

Identical to NumPy, and `jax.jit` compiles the buggy version happily — all shapes are static
and consistent. `jnp.reshape` raises only when the element count changes
(`test_reshape_checks_only_the_element_count`).

The structural point: the transpose is expressed by **name** (via the `dims` tuple) and the
reshape by **position** (a tuple of ints). The two halves of one logical operation use
different addressing schemes, and nothing checks that they agree. Any API that lets a
reshape take bare integers has this hole.
