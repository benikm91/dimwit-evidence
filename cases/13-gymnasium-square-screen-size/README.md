# 13 — a (width, height) API meeting a (height, width) API

**Source:** [Farama-Foundation/Gymnasium#1312](https://github.com/Farama-Foundation/Gymnasium/pull/1312)
— *"Allow `AtariPreprocessing` non-square observations"*

## The defect

`AtariPreprocessing` passes `screen_size` straight into `cv2.resize`, whose `dsize`
parameter is ordered **(width, height)**. The array `cv2.resize` returns is ordered
**(height, width)**. Both conventions are correct; they are simply not the same one. The
wrapper declared its observation space in the first order while producing the second:

```python
_shape = self.screen_size + (1 if grayscale_obs else 3,)        # (W, H, C)  -- as shipped
_shape = (self.screen_size[1], self.screen_size[0], ...)        # (H, W, C)  -- the fix
```

The PR author states it directly:

> Since screen size are directly passed into `cv2.resize`, its order should be width x
> height. Observation shape should be `height x width`.

It shipped because `screen_size` had always been square, and `(s, s)` is the one input on
which the two conventions agree. The bug was only reachable once someone asked for a
rectangle.

## How it is reconstructed here

The upstream mismatch is between a produced array and a *declared* space. All four
reconstructions distil it to its two halves so the same program can be written in each
language:

* `inner_resize` / `innerResize` — the `cv2.resize` half. Takes its extents **width-first**,
  returns a **height-first** array, and is agnostic about what its axes mean.
* `resize_*` / `resize*` — the pipeline half. Takes its extents **height-first**.

The defect is a pair of extents crossing between them without being flipped. Mechanically
identical to upstream; the difference is that the mismatch is made to surface in a call
rather than in an untyped shape declaration.

## Results

| implementation | verdict |
| --- | --- |
| plain NumPy | `missed` — a well-formed frame of the wrong orientation |
| plain JAX | `missed` — `eval_shape` and `jit` both confirm the transposed result |
| jaxtyping | `missed` — names do not bind to `int`, so nothing relates the extents |
| **DimWit** | **`compile`** — see below |

## Why DimWit catches it

`Buggy.scala` is rejected:

```
[error] Found:    (height : dimwit.tensor.AxisExtent[H])
[error] Required: dimwit.tensor.AxisExtent[W]
```

Note what is *not* doing the work. `resizeBuggy` is fully generic — neither `Height` nor
`Width` appears in its signature — so this is not a case of two concrete labels clashing.
The check fires at the definition of a generic helper, before any call site supplies a
concrete axis.

What does the work is that **a resize does not rename an axis, it resizes one**, so the
extents are typed with the frame's own labels:

```scala
def innerResize[H: Label, W: Label](
    frame: Tensor2[H, W, Float32],
    width: AxisExtent[W],
    height: AxisExtent[H]
): Tensor2[H, W, Float32]
```

The `frame` argument anchors `H` and `W`, and every extent must then name the axis it is
actually resizing. `innerResize` stays as layout-agnostic as `cv2.resize` is — it works for
any pair of axes — while ceasing to be *label*-agnostic about which extent goes where. The
check costs nothing at the call site.

The instructive contrast is the signature that gives it away. Had the output been given its
own fresh labels —

```scala
def innerResize[A: Label, B: Label, OutW: Label, OutH: Label](
    frame: Tensor2[A, B, Float32], width: AxisExtent[OutW], height: AxisExtent[OutH]
): Tensor2[OutH, OutW, Float32]
```

— the swapped call compiles, because nothing anchors `OutW` and `OutH`. Worse, such a
function must end in `relabelAll` to produce its fresh labels, and `relabelAll` reattaches
names by position: the names get permuted along with the sizes, so `shape(Axis[Height])`
still answers `84` on a frame that is 84 wide and 64 tall. The names survive and stop
meaning anything.

So the rule the case supports is narrower and more actionable than "named axes catch axis
swaps": **type an operation's extents with the labels of the tensor it operates on**, and
introduce fresh axis variables only where the operation genuinely renames an axis — which
resizing, cropping, padding and scaling do not. See `DIMWIT_SOLUTION.md`.

## Supplementary files

| file | what it shows |
|---|---|
| `tensor_annotations_case.py` | *supplementary:* annotated with [tensor_annotations](https://github.com/google-deepmind/tensor_annotations), generic over the frame's own axes as DimWit's `innerResize` is; the extents are `tuple[int, int]`, so the unflipped pair passes mypy and pyright — the tests assert zero errors plus a control of the inferred types |
| `TENSOR_ANNOTATIONS_SOLUTION.md` | *supplementary* verdict; run with `./scripts/check_tensor_annotations.sh -v` (not part of `run_all.sh`) |
