# Plain JAX — verdict

**Category: `missed`**

Identical to NumPy. `jnp.cross(a, a, axisa=0, ...)` and `jnp.cross(a, a, axisa=1, ...)`
are both accepted on a `(3, 3)` array and both return `(3, 3)`; see
`test_jnp_cross_itself_requires_a_size_three_axis_only`.

JAX's own contribution to this class of bug is that its `cross` follows the NumPy default
while PyTorch's follows a different one, which is how the discrepancy arose in a
multi-backend library in the first place.
