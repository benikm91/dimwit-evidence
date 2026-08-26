# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:21:15
Found:    (g : Tensor1[Case09Fixed.Component, Float32])
Required: Tensor2[Case09Fixed.GPoint, Case09Fixed.Component, Float32]
    ftAoFixed(g, centers)
              ^
```

The defect is inside the routine, not at the call: `ft_ao`'s body assumes `Gv` is `(N, 3)`
while its signature accepts anything, so a bare `(3,)` reduces the only axis instead of the
last one, and the column that should have multiplied a grid becomes a scalar broadcasting
over the centres. Nothing raises; the result is one rank short and wrong.

DimWit makes the assumption the signature. A grid of G-vectors is
`Tensor2[GPoint, Component, Float32]`, so the vector cannot arrive in the first place and
there is no missing axis for a broadcast to invent. `Gv.reshape(-1, 3)`, the upstream fix,
moves to the caller and becomes `one.prependAxis(Axis[GPoint])` — the same act, except that
it names the axis being added and the routine could not have been called without it. Rank is
the part of a shape that cannot coincide by accident, which is why jaxtyping also catches
this one; the difference is that DimWit rejects it before the program runs, and that the
grid case is `vmap` over a per-vector body rather than a second implementation hoping to
agree with the first.
