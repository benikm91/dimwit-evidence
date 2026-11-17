# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

`Buggy.scala` hands a vector with one value per stacked draw to a routine written for a
vector with one value per user sample. Under positional labels both are `Tensor1[A0]`. The
flatten gives rank 2, the mean gives rank 1, the routine expects rank 1, and there is
nothing to reject.

With semantic names the same call is a compile error (`DIMWIT_SOLUTION.md`):
`Tensor1[Chain |*| Draw]` is not `Tensor1[Sample]`. Both have rank 1, so the detection in
this case comes from the names and not from rank typing.

**Limit of the comparison.** The upstream defect is a collision between two axis *names*,
and positional labels have no names to collide. What is reconstructed here is its
consequence: a value indexed by one axis used where a value indexed by another is expected.
That substitution is silent with positions, as it is in plain NumPy, and it is what
`Buggy.scala` rejects.
