# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:22:7
Found:    Tensor[(Case05Fixed.Width, Case05Fixed.Height, Case05Fixed.Channel), Float32]
Required: Tensor3[Case05Fixed.Height, Case05Fixed.Width, Case05Fixed.Channel, Float32]
```

`input_shape[2]` cannot be written: the spatial extents are taken out of a
`Shape[(Batch, Channel, Height, Width)]` by name, and each one carries that name with it, so
naming them in the wrong order does not produce a swapped buffer — it produces a
`Tensor3[Width, Height, Channel]`, which is not the letterbox buffer the pipeline declared.
Unlike case 04 the mistake survives into a *tensor* rather than collapsing to a number, and
that is what the types can still see: it is rejected identically for the 640x480 export and
the default square 640x640 one, where jaxtyping's `height` and `width` bind to the same
number and it passes.
