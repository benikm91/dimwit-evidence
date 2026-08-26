# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

`pred - target` with shapes `[n, 1]` and `[n]` is a legal broadcast under NumPy's rules:
the trailing axes `1` and `n` are compatible, the missing leading axis is prepended. The
result is `[n, n]`, its mean is a perfectly good scalar, and training proceeds.

`test_buggy_training_finds_the_wrong_weights` shows the observable consequence: gradient
descent converges to weights that are not the true ones, on a convex problem where the
correct answer is unambiguous.

PyTorch is a partial exception and worth quoting precisely in the paper. Since 1.5 it
emits

> Using a target size (torch.Size([100])) that is different to the input size
> (torch.Size([100, 1])). This will likely lead to incorrect results due to broadcasting.

That is a **warning**, on the default warning filter, from inside a training loop that may
print thousands of lines. `test_torch_only_warns` asserts it is a warning and not an
exception. The upstream issue asked for an error and did not get one, because broadcasting
`[n, 1]` against `[n]` is genuinely wanted elsewhere.
