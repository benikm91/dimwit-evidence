# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:24:26
Found:    Tensor1[Case07Fixed.Chain |*| Case07Fixed.Draw, Float32]
Required: Tensor1[Case07Fixed.Sample, Float32]
    summariseUserSamples(meanOverUserSamples(stacked))
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Flattening two axes does not invent a name, it composes one: `flatten((Axis[Chain], Axis[Draw]))`
yields the label `Chain |*| Draw`, so the upstream question — what should the stacked
dimension be called? — never arises, and neither does `stack_buggy`. Axis labels are Scala
types rather than strings in one flat runtime namespace, so a library's axis and a user's
axis cannot collide even when both are spelled `Sample`, and handing one where the other is
expected is a type error rather than a silent selection of the first match. This is the case
where DimWit beats every other *named*-axis system, not just the positional ones — xarray,
ArviZ and einops patterns all identify axes by strings, and `__sample__` is the best a
string namespace can do.
