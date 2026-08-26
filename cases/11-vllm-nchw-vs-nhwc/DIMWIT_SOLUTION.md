# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:18:24
Found:    (decoded : Case11Fixed.Nchw)
Required: Case11Fixed.Nhwc
    perChannelMeanNhwc(decoded)
                       ^^^^^^^
```

The layout is a contract carried entirely by prose in every other tool here: nothing in an
`ndarray` records whether axis 1 is a channel or a row, so swapping the decoder backend
swaps the meaning of every subsequent axis index with no diagnostic. `Nchw` and `Nhwc` are
two types, written once, and the transpose between them is spelled with axis names rather
than a tuple of positions. Crucially this holds at `[2, 3, 3, 3]` — the square-RGB frame
that shipped — where jaxtyping's `frame height width channel` binds equally well to both
readings and passes.
