# tensor_annotations — verdict

**Category: `missed`**

*Supplementary. This column is not part of the results table in the top-level README and
is not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The defect is a scalar divisor, and tensor_annotations types arrays, not extents. Two
variants are in `tensor_annotations_case.py`; both pass both checkers with zero errors.

* **As written upstream** (`size: tuple[int, int]`). Not even the return annotation
  `Array3[float32, Height, Width, Components]` is checked against anything:
  `jax.random: Any` and `stack: Any` in the stubs, so the result is `Any`.
* **Most favourable typing.** `dx` annotated `Array2[float32, Height, Width]`, and `size`
  read from an image typed `Array2[float32, Height, Width]`. Still missed:
  * `image.shape` is `Tuple[int, int]`. The stubs do not connect shape entries to the
    axes of the array they came from, so `shape[0]` and `shape[1]` are interchangeable.
  * `Array2[Height, Width] / int` is `Array2[Height, Width]` whichever `int` it is. The
    type of the result cannot depend on which number was divided by.

The tests include a control: both checkers infer `image.shape` as `Tuple[int, int]` and
`dx / image.shape[0]` and `dx / image.shape[1]` as the *same* `Array2[Any, Height, Width]`.

**What would catch it — and why it is not tensor_annotations.** Probed separately
(scratch, not in the repo): with plain-Python `NewType` extents
(`HeightPx = NewType("HeightPx", int)`, `WidthPx = ...`) and a helper
`norm_dx(dx, width: WidthPx)`, the call `norm_dx(dx, size[0])` is rejected by both mypy and
pyright. But

* that is ordinary Python typing, equally available to jaxtyping, and needs no
  tensor_annotations at all;
* dividing directly, `dx / size[0]`, is still accepted, because a `NewType` of `int` *is* an
  `int` for arithmetic;
* it changes the interface — the same caveat `DIMWIT_SOLUTION.md` records for DimWit's
  `AxisExtent[Width]`: catching a different program is a different claim.

The case is the same boundary for tensor_annotations as for DimWit: names as types govern
tensors, and an extent read out as a number has left the system. Unlike DimWit, nothing
improves legibility here either: `size[0]` is still written positionally.
