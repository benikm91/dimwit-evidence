# DimWit, rank-only — verdict

**Category: `compile-time detection`**

```
./RankOnly.scala:25:36
Found:    (target : Case02RankOnly.Rank1)
Required: dimwit.tensor.Tensor[(Case02RankOnly.A0, Case02RankOnly.A1), dimwit.tensor.DType.Float32]
    val residuals: Rank2 = (pred - target).relabelAll(rank2)
                                   ^^^^^^
```

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

The defect is still rejected, because it is a rank error: the prediction is `[n, 1]`, the
target is `[n]`, and `-` is shape-exact, so rank 2 minus rank 1 is not a subtraction. The
names are not needed to see it. This is the same reason tensor_annotations and jaxtyping
catch this case.

**Limit.** The detection relies on `-` not broadcasting between ranks. A rank-typed library
that kept NumPy's implicit broadcasting would accept `pred - target`.
