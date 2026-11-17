# DimWit, rank-only — verdict

**Category: `compile-time detection`**

```
./RankOnly.scala:32:10
Found:    (g : Case09RankOnly.Rank1)
Required: Case09RankOnly.Rank2
    ftAo(g, centers)
         ^
```

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

The defect is still rejected, because it is a rank error: the routine declares a grid of
G-vectors, rank 2, and the caller hands it a single vector, rank 1. The names are not needed
to see it. This is the same reason tensor_annotations and jaxtyping catch this case.

**What the names add** is in the fix, not in the detection. With names the caller writes
`one.prependAxis(Axis[GPoint])` and says which axis is being added. With positional labels
the added axis is simply the new position 0.
