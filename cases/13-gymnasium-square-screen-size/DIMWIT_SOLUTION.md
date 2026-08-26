# DimWit — verdict

**Category: `compile`**

```
[error] Found:    (height : dimwit.tensor.AxisExtent[H])
[error] Required: dimwit.tensor.AxisExtent[W]
[error]     innerResize(frame, height, width)
[error]                        ^^^^^^
```

A resize does not rename an axis, so `innerResize` types its extents with the frame's own
labels — `frame: Tensor2[H, W, Float32]`, `width: AxisExtent[W]`, `height: AxisExtent[H]` —
and the frame anchors both variables, catching the swap even though `resizeBuggy` is fully
generic and names neither axis. Had the output been given fresh labels instead
(`width: AxisExtent[OutW] … : Tensor2[OutH, OutW, Float32]`) the swap would compile, since
nothing anchors them and the `relabelAll` such a signature needs reattaches names by
position, leaving every by-name query answering as expected. This is the complement of case
06's limit: a lone `Axis[Col]` has one occurrence and nothing to disagree with, because
DimWit checks agreement *between* occurrences of a label, not the correctness of one used
once.
