# DimWit, rank-only — verdict

**Category: `missed`** — at the scope upstream used, softmax over any axis of a tile. The
`keep_dims` flag itself is still not expressible; see the last section.

`scala-cli compile RankOnly.scala` succeeds, and `scala-cli run RankOnly.scala` prints, for
the reporter's 2x2 tile `[[0, 2], [3, 1]]`:

| | result | correct? |
|---|---|---|
| `softmaxVector`, `vmap` over axis 0 | `[0.1192, 0.8808] [0.8808, 0.1192]` | yes, rows sum to 1 |
| `softmaxVector`, `vmap` over axis 1 | `[0.0474, 0.9526] [0.7311, 0.2689]` | right values, transposed |
| `softmaxMatrix`, axis 1 | `[0.1192, 0.8808] [0.8808, 0.1192]` | yes, rows sum to 1 |
| `softmaxMatrix`, axis 0 | `[0.0180, 0.1329] [3.6945, 0.5000]` | **no**, columns sum to 3.71 and 0.63 |

`RankOnly.scala` holds `softmaxVector` and `softmaxMatrix` of `Fixed.scala` in the same
library and with the same compiler, but with the semantic names removed: an axis is
labelled by its position (`A0`, `A1`, ...), and the result of every operation is relabelled
back to those positions. A type then states the rank of a tensor and nothing else.

## `softmaxMatrix`: the upstream mechanism, mirrored

The reduction drops the reduced axis, as `keep_dims=False` did. What is left is a rank-1
tensor, a `Tensor1[A0]`, whichever axis it came from, and `x -! max` puts a `Tensor1[A0]`
on axis 0 of `x`. Reducing axis 1 leaves the row index, which belongs on axis 0, so that
direction is right. Reducing axis 0 leaves the column index, which lands on axis 0 as well,
and the result is wrong. It compiles, runs, and has the right shape.

This is triton#11406 with the sides exchanged. NumPy re-attaches a rank-reduced vector to
the last axis, so upstream `dim=1` was wrong and `dim=0` accidentally right. Positional
labels re-attach it to the first axis, so here it is the other way round. With semantic
names `softmaxMatrix` is right for both axes (`DIMWIT_SOLUTION.md`), because the reduced
vector keeps the name of the axis that is left and can only be broadcast back there. Rank
is equal in all of these, so the detection comes from the names and not from rank typing.

## `softmaxVector`: the minimal scope still helps, up to the lift

At the scope of a vector there is one axis and nothing to misalign, with or without names.
Lifted over axis 0 the result is correct. Lifted over axis 1 the values are right but the
mapped axis comes out in front, as with `jax.vmap(f, in_axes=1)`, and the relabel to
positions hides that the tile is now transposed. (Not checked here: with names the axis
order is part of the result type, so the transposition should be visible to the compiler.)

## The `keep_dims` flag

Not in the file any more, but checked: `x.max(Axis[A1], keepDims)` is rejected under
positional labels with the same error `Buggy.scala` gets, "`max` does not take parameters".
A reduction takes rank 2 to rank 1, and a run-time `Boolean` cannot change a type. Rank is
enough to rule the flag out. It is not enough to make the rank-reduced result land on the
right axis.
