# DimWit, rank-only — verdict

**Category: `compile-time detection`**

```
./RankOnly.scala:32:11
Found:    (idx : dimwit.tensor.Tensor2[Case10RankOnly.A0, Case10RankOnly.A1, dimwit.Int32])
Required: dimwit.tensor.Tensor1[Case10RankOnly.A0, dimwit.Int32]
    model(idx, emb, wOut)
          ^^^
```

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

The defect is still rejected, because it is a rank error: the model is written for one
sequence, rank 1, and is handed a batch of sequences, rank 2. The names are not needed to
see it. This is the same reason tensor_annotations and jaxtyping catch this case.

As in case 03, the detection rests on the scope: `model` takes one sequence and batching is
a `vmap` from the outside. That scope is enforced by rank alone.

**What the names add** is inside the model, not in the detection. With names the centring
reads `x -! x.mean(Axis[Pos])` and the mean can only be broadcast back onto `Embed`. With
positional labels the mean is a rank-1 tensor that has to be placed on axis 1 by hand
(`mean.relabelTo(Axis[A1])`, the counterpart of `mean[None, :]`), and placing it on the
wrong axis is the hazard of case 06. The model body here compiles; it was not run.
