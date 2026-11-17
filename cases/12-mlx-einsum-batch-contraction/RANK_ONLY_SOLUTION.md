# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

`Buggy.scala` contracts the row axis of `a` where the inner one was meant. DimWit's `dot`
names the axis it sums over and requires that axis in both operands. Under positional
labels that requirement is always met: every rank-2 tensor has an `A0` and an `A1`, so
`a.dot(Axis[A0])(b)` is a contraction. It sums the row axis of `a` against the inner axis
of `b` and returns a `Tensor2[A0, A1]`, as declared. On a square input it returns the same
numbers as `np.tensordot(a, b, axes=([0], [0]))`, which are not the matrix product (checked
by running it, the check is not in the repo).

With semantic names the same line is a compile error (`DIMWIT_SOLUTION.md`):
`Axis[Row] not found in Tensor[(Inner, Col)]`. Rank is equal on both sides, so the
detection in this case comes from the names and not from rank typing.

**The correct contraction is the awkward one.** Axis 1 of `a` against axis 0 of `b` are two
different positions, so with positional labels `b` has to be relabelled first to line them
up: `a.dot(Axis[A1])(b.relabelAll((Axis[A1], Axis[A2])))`. The contraction that needs no
relabel is a wrong one. As in case 08, restricting the API to a contraction by axis does
not help by itself: an axis given by position says where it is, not what it is.
