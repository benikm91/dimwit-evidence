# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

Under these labels the scores are `Tensor4[A0, A1, A2, A3]` and so is the expanded mask,
whether its `seq` axis was placed on axis 2 (query, the bug) or on axis 3 (key, the fix).
The addition is rank 4 + rank 4, and there is nothing for the compiler to reject. Run on the
example of `plain.py`, the program puts weight 1.0 on the padded key, where the fixed
placement puts 0.0.

With semantic names the same line is a compile error (`DIMWIT_SOLUTION.md`): the expanded
mask is `[Batch, Heads, Key, Query]` and the scores are `[Batch, Heads, Query, Key]`. The
two differ only in the names, so the detection in this case comes from the names and not
from rank typing.

**How the expansion is written.** `additive[:, None, :, None]` becomes
`additive.relabelAll((Axis[A0], Axis[A2])).broadcastTo(scores.shape)`. This is how "put
axis 1 of the mask at position 2 of the scores" is spelled in DimWit. The intermediate
`[A0, A2]` exists only inside that expression; the result is relabelled to rank 4 at once.
