# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:42:71
Found:    Tensor4[Case01Buggy.Batch, Case01Buggy.Heads, Case01Buggy.Key, Case01Buggy.Query, Float32]
Required: Tensor[(Case01Buggy.Batch, Case01Buggy.Heads, Case01Buggy.Query, Case01Buggy.Key), Float32]
    val biased: Tensor4[Batch, Heads, Query, Key, Float32] = scores + expanded
                                                                      ^^^^^^^^
```

The axes carry names, every operation validates them, and so putting a `Key` axis where
`Query` is required is a compile error — whereas jaxtyping binds `query` and `key` to the
same extent in self-attention and accepts it. The expansion is not needed in the first
place: `+!` broadcasts by adding the *missing* named dimensions, so a
`Tensor2[Batch, Key, Float32]` mask lands on `Batch` and `Key` and nowhere else.
