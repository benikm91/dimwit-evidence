# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:29:15
Found:    (pred : Tensor2[Case02Buggy.Batch, Case02Buggy.Out, Float32])
Required: Tensor1[Case02Buggy.Batch, Float32]
    mseStrict(pred, target)
              ^^^^

./Buggy.scala:33:58
Found:    (target : Tensor1[Case02Buggy.Batch, Float32])
Required: Tensor[(Case02Buggy.Batch, Case02Buggy.Out), Float32]
    val residuals: Tensor2[Batch, Out, Float32] = pred - target
                                                         ^^^^^^
```

The bug was not a slip in the caller but a permissive implementation: `nn.MSELoss` never
specified whether a prediction and a target may have different shapes, so `[n, 1]` against
`[n]` was neither allowed nor forbidden, merely unnoticed — and once callers depended on
either reading, the only remaining fix was the ad-hoc `UserWarning`. A static signature
both specifies that boundary and verifies it: `mseStrict(pred: Tensor1[Batch, Float32],
target: Tensor1[Batch, Float32])` has to be decided when the loss is first written, and is
then checked at every call site and in every body — `-` is shape-exact, so even the version
that keeps the `Out` axis must say `-!` to broadcast. The unspecified case never exists to
be discovered later.
