# Plain JAX — verdict

**Category: `missed`**

Same broadcasting rules, plus one JAX-specific aggravation worth reporting:
`jax.jit` *specialises on the input rank*. Calling the jitted function with a `(3,)` and
then with an `(N, 3)` argument produces two separate compiled programs, each internally
consistent, one of them wrong
(`test_jit_specialises_on_each_rank_without_complaint`).

Rank polymorphism is a deliberate and useful feature of the NumPy API family. This case
shows its cost: a function whose contract is "a list of vectors" cannot state that contract
anywhere the machine will read.
