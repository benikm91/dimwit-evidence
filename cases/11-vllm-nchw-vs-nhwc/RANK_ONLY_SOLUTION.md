# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

The defect hands NCHW frames to a pipeline written for NHWC. Under positional labels both
layouts are `Tensor4[A0, A1, A2, A3]`: a layout is a statement about what the positions
mean, and that is exactly what a rank-only type does not say. The call is rank 4 where rank
4 is expected, and the pipeline reduces axes 0, 1 and 2, which are frame, channel and
height of the decoded frames. It returns one mean per *width* position and calls it a mean
per channel.

With semantic names the same call is a compile error (`DIMWIT_SOLUTION.md`): `Nchw` and
`Nhwc` are two types. Both have rank 4, and for the square RGB frames that shipped,
`[2, 3, 3, 3]`, the same extents, so the detection in this case comes from the names and
not from rank typing. tensor_annotations catches this case for the same reason: it compares
axes by identity.
