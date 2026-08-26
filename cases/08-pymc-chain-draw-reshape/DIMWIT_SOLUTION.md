# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:21:5
Found:    (fused : Tensor2[Case08Fixed.Draw |*| Case08Fixed.Chain, Case08Fixed.Team, Float32])
Required: Case08Fixed.PointList
    fused
    ^^^^^
```

The upstream defect is that a transpose expressed by *name* and a reshape expressed by
*position* disagreed, because `reshape` takes a tuple of integers with no connection to the
axes those integers came from. DimWit has no `reshape`: `flatten` records which axes were
fused and in which order, in the resulting label, so `Draw |*| Chain` and `Chain |*| Draw`
are different types and the point list's declared layout is checked against the fusion that
actually happened. `unflatten` closes the loop from the other side — it demands a `Shape`
whose labels reconstruct the fused axis, so the round trip cannot be reassembled in the
wrong order either.
