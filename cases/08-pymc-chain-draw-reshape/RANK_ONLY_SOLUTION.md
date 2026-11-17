# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the upstream defect in the same library and with the same compiler, but
with the semantic names removed: an axis is labelled by its position (`A0`, `A1`, ...), and
the result of every operation is relabelled back to those positions. A type then states the
rank of a tensor and nothing else.

Upstream, the transpose and the flatten disagreed because the flatten used information read
from the dataset *before* it was transposed. With positional labels that information is the
positions themselves. The sampling axes are 1 and 2 in the dataset `[team, draw, chain]`,
and 0 and 1 once it is transposed to `[chain, draw, team]`. A transpose changes what `A1`
and `A2` refer to, so `flatten((Axis[A1], Axis[A2]))` after it fuses draw and team. The
result is a `Tensor2[A0, A1]`, which is what the function declares, and there is nothing to
reject. On the reporter's `(5, 2, 3)` dataset it has shape `(3, 10)` where the point list
is `(6, 5)` (checked by running it, the check is not in the repo): the wrong rectangle, as
upstream.

With semantic names this cannot happen. A name refers to the same axis before and after a
transpose, so `flatten((Axis[Chain], Axis[Draw]))` fuses chain and draw wherever they are,
and the result type `Chain |*| Draw` records it. Fusing in the wrong order is a compile
error (`DIMWIT_SOLUTION.md`): `Draw |*| Chain` is not `Chain |*| Draw`. All of these have
rank 2, so the detection in this case comes from the names and not from rank typing.

**Notes.**

* The mistake as `Buggy.scala` writes it, fusing in the order draw, chain, also compiles
  under positional labels: `(6, 5)`, the right rectangle with its rows permuted.
* DimWit has no `reshape` under any labels, so the sizes cannot be read at the wrong
  moment. That removes the channel upstream used, not the defect: the same disagreement
  comes back through the positions.
