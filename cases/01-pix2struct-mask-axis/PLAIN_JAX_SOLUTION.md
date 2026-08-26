# Plain JAX — verdict

**Category: `missed`**

JAX inherits NumPy's broadcasting rules, so the verdict is identical to plain NumPy: both
versions type-check, trace and compile.

Worth stating explicitly for the paper, because `jit` is sometimes assumed to add safety:
`jax.jit` performs full shape inference over the traced graph, and it accepts the buggy
program without complaint. Shape inference proves the program is *well formed*; it cannot
know that the author meant the mask to line up with keys rather than queries.

See `plain_jax.py::test_jit_does_not_object_either`.
