# Plain JAX — verdict

**Category: `missed`**

JAX has no equivalent of PyTorch's warning, so this is strictly quieter than the original.
`jnp.mean((pred - target) ** 2)` traces, jits and differentiates. `jax.grad` produces a
clean gradient — of the wrong objective.

The relevant point for the paper is that every automatic mechanism in the JAX stack
(tracing, shape inference, XLA compilation, autodiff) operates on a program that is
*well formed*. None of them has access to the fact that the author meant a per-example
residual rather than a matrix of pairwise ones.
