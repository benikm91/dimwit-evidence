# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:25:102
Found:    (y : dimwit.tensor.Tensor1[Case03Buggy.Rgb, dimwit.Float32])
Required: dimwit.tensor.Tensor1[Case03Fixed.Spatial, dimwit.tensor.DType.Float32]
    zipvmap(Axis[Sample])(a, b)((x: Tensor1[Spatial, Float32], y: Tensor1[Rgb, Float32]) => cross(x, y))
                                                                                                      ^
```

The upstream defect needs a permissive scope to exist in: `cross` is conceptually a function
of two 3-vectors, and only because the libraries widened it to arbitrary rank with an axis
argument per operand is there a default to disagree about. `cross[L](v1: Tensor1[L, V], v2: Tensor1[L, V]): Tensor1[L, V]`
is that concept at its minimal scope — vectors in, both on one axis, and the axis survives
into the result, so `axisa`, `axisb` and `axisc` collapse into the `L` the caller already
fixed by choosing what to pass. `Buggy.scala` uses that same definition and only gets the
call wrong: every axis in play has extent 3, so `torch.cross` would find one in each operand
and return a number, whereas here the names carry the semantics and a spatial vector cannot
be crossed with a colour triple. Batching is a separate, named concern — `zipvmap(Axis[Sample])`
says which axis it maps over — rather than a default hidden in the operation.
