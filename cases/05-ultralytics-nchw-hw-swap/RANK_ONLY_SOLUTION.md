# DimWit, rank-only — verdict

**Category: `missed`**

`scala-cli compile RankOnly.scala` succeeds.

`RankOnly.scala` is the defect of `Buggy.scala` in the same library and with the same
compiler, but with the semantic names removed: an axis is labelled by its position (`A0`,
`A1`, ...), and the result of every operation is relabelled back to those positions. A type
then states the rank of a tensor and nothing else.

The mistake reads `input_shape[2]` as the width and `input_shape[3]` as the height, and
builds the letterbox buffer from them. Under positional labels a `(width, height, channel)`
buffer and a `(height, width, channel)` buffer are both `Tensor3[A0, A1, A2]`. The
transpose to CHW and the added batch axis give rank 3 and rank 4 either way, so the
function returns the declared `Tensor4[A0, A1, A2, A3]` with the two spatial axes swapped.

With semantic names the same code is a compile error (`DIMWIT_SOLUTION.md`): the extents
carry their axis, so the swapped buffer is a `Tensor3[Width, Height, Channel]`, which is
not the `Tensor3[Height, Width, Channel]` that was declared. Both have rank 3, and for the
default square 640x640 export also the same extents, so the detection in this case comes
from the names and not from rank typing.

**How the buffer is written.** The extents are read as numbers,
`Axis[A0] -> onnxInputShape(Axis[A3])`: the first axis of the new buffer gets the size of
position 3 of the input shape. Reading them with `extent(...)` instead would label the
buffer by where its sizes came from, `[A3, A2, A2]` buggy and `[A2, A3, A2]` fixed, and the
relabel to positions that follows every operation removes that difference again. A buffer
that is new has positions 0, 1, 2 and nothing else.
