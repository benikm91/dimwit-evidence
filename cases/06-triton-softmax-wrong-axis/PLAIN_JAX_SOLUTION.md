# Plain JAX — verdict

**Category: `missed`**

`jax.eval_shape` returns the identical abstract value for both versions. More importantly,
`test_a_cross_entropy_loss_built_on_the_wrong_axis_still_trains` shows the defect passes
cleanly through `jax.grad`: the gradients are finite, the loss is a real number, and the
model trains — on the wrong objective.

This is the failure mode that motivates the whole project, and it is also the failure mode
that types cannot reach. Worth being explicit about that in the paper.
