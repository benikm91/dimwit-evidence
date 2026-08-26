# Plain JAX — verdict

**Category: `missed`**

`jax.eval_shape` accepts the wrong layout and reports the expected `(3,)` output. The
normalisation built on the wrong statistics produces a finite tensor of the right shape
(`test_a_normalisation_built_on_them_still_produces_a_valid_image`), so nothing downstream —
not a shape assertion, not a NaN check — has anything to fire on.

Multi-backend interfaces are where this class of defect lives: two implementations of one
Python signature, disagreeing about a convention that the signature cannot express.
