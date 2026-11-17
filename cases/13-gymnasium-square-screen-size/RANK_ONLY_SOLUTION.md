# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

Without names an extent has no axis to be named after, so the height and the width are
plain `Int`s. Forwarding `(height, width)` into an API that takes `(width, height)` is an
`Int` where an `Int` is declared, and there is nothing to reject. The frame comes back as a
`Tensor2[A0, A1]` with its two extents exchanged.

With semantic names the same call is a compile error (`DIMWIT_SOLUTION.md`): the height is
an `AxisExtent[H]`, the width an `AxisExtent[W]`, and the frame anchors both. The typed
extent is itself a use of the names, which is why it is not carried over here.

**Note.** If the extents are typed by position instead, `AxisExtent[A0]` and
`AxisExtent[A1]`, the swap is rejected (checked separately, not in the repo). That keeps a
part of DimWit's naming machinery and is no longer rank-only, so it is not the verdict.
