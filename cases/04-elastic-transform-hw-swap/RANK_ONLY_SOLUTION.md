# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

The defect is a scalar divisor. `size(Axis[A0])` and `size(Axis[A1])` are both a plain
`Int`, and a rank-2 tensor divided by either is a rank-2 tensor. There is nothing to reject.

This is the one case that DimWit misses with semantic names too (`DIMWIT_SOLUTION.md`), for
the same reason: an extent read out as a number has left the type system. The verdict is
therefore the same in both variants.

What the names changed there was legibility, and that is lost here. With names the buggy
line reads `alpha._1 / size(Axis[Height])`, the horizontal displacement divided by the
height. Here it reads `alpha._1 / size(Axis[A0])`, which says no more than `size[0]` did
upstream.
