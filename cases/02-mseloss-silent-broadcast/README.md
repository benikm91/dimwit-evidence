# 02 — MSE between `[n, 1]` predictions and `[n]` targets

**Source:** [pytorch/pytorch#16045](https://github.com/pytorch/pytorch/issues/16045)
— *"fail to throw error when computing loss between tensors with shapes [n, 1] and [n]"*

## The defect

`nn.Linear(2, 1)` returns `[n, 1]`. The targets are `[n]`. `nn.MSELoss` subtracts them,
broadcasting to `[n, n]` — the full matrix of pairwise residuals — and averages that.

The loss decreases. The model trains. The weights are wrong.

The reporter noticed because a two-variable linear regression, a convex problem with a
closed-form solution, refused to reach its optimum. PyTorch's resolution was a
`UserWarning`, not an error: implicit broadcasting is too load-bearing to remove.

## Why it is interesting

The upstream discussion never settled whether the broadcast was intended behaviour. That
is the point: with untyped shapes the question only comes up once someone's regression
fails to converge, and it then has to be answered for every existing caller at once —
which is why the resolution could only be a warning.

A strict, conceptually scoped signature removes the issue at the type level rather than
patching it. `mse(pred: Tensor1[Batch], target: Tensor1[Batch])` forces the question at
the moment the loss is first written: does this function accept a prediction with an
output axis, or not? Whatever the answer, it is recorded, checked, and cannot be reached
by accident later.

## Files

| file | what it shows |
|---|---|
| `plain.py` | NumPy reconstruction, buggy + fixed, plus the PyTorch warning |
| `plain_jax.py` | the same in JAX, with the same hand-written shape assertion as the fix |
| `jaxtyping_case.py` | `mse_strict` (caught at run time) and `mse_as_written_upstream` (missed) |
| `Buggy.scala` | both jaxtyping signatures in DimWit, in one file as they are there — neither can reach the defect |
| `Fixed.scala` | `mseStrict`, with a `@main` check that runs it |
| `PLAIN_SOLUTION.md`, `PLAIN_JAX_SOLUTION.md`, `JAXTYPING_SOLUTION.md`, `DIMWIT_SOLUTION.md` | verdicts |
