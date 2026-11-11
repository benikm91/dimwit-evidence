# tensor_annotations — verdict

**Category: `missed` — the extents are `int`s, and tensor_annotations has no type for a
single extent.**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The case is annotated as favourably as DimWit's `Fixed.scala`: the resize is generic over
the frame's own axes and keeps them —
`inner_resize(frame: Array2[A1, A2], dsize) -> Array2[A1, A2]` — because a resize resizes
an axis rather than renaming it. DimWit catches the swap at exactly this point, by typing
the extents with those same labels (`width: AxisExtent[W]`, `height: AxisExtent[H]`), so
the frame anchors them and the unflipped pair fails.

tensor_annotations has no counterpart to `AxisExtent`. The extents are `tuple[int, int]`;
`dsize[0]` and `dsize[1]` have the same type, nothing ties either to `A1` or `A2`, and the
unflipped pair is exactly as well typed as the flipped one. `resize_buggy` and
`resize_fixed` have identical signatures and both pass.

The body would not have been checked anyway: `jnp.arange` and indexing
(`frame[rows][:, cols]`) are `Any`, so the declared `Array2[A1, A2]` is never compared with
the array that is built. The tests assert zero errors in both checkers, with a control:
`frame.shape` is `Tuple[int, int]`, `jnp.arange` and indexing are `Any`, and
`resize_buggy(frame, (84, 64))` is `Array2[Height, Width]` — as declared, while at run time
it is 64 high and 84 wide.

This is the same boundary as cases 04 and 05: an extent read out as, or passed in as, a
number has left the type system. DimWit draws the line one step later, because its extents
keep their axis.
