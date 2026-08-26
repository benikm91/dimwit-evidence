# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:26:18
value max is not a member of Tensor2[Case06Fixed.Row, Case06Fixed.Col, Float32].
An extension method was tried, but could not be fully constructed:
    dimwit.max()
    failed with:
        value max: <overloaded dimwit.max> does not take parameters
    val z = x -! x.max(Axis[Col], keepDims)
                 ^^^^^
```

**The root cause is a permissive implementation, and the permissiveness is a scoping
choice.** `tl.softmax` accepts a tile of any rank. That is what forces a `dim` argument, and
what makes the internal reductions turn a rank-n tile into a rank-(n-1) one that then has to
find its way back onto n axes. `keep_dims` is the flag steering that return trip, its default
steers it wrong, and the result is an accidental reduction of a dimension.

**DimWit scopes softmax to a vector, which removes the options.** `Case06Fixed.softmaxVector`
is `Tensor1[L, Float32] => Tensor1[L, Float32]`. One axis, so no `dim`; the reduction still
drops the dimension — `max` and `sum` of a vector are `Tensor0[Float32]`, exactly what
`keep_dims=False` asked for — but a scalar returning to a vector has one axis to land on and
it is the axis that was reduced. There is nothing left for a flag to decide, and none can be
added: a reduction's result type is fixed by the axis it names, so a runtime `Boolean` cannot
change it. `x.max(Axis[Col], keepDims)` is not a wrong call, it is not a call. The tile
version is the vector one lifted: `x.vmap(Axis[Row])(softmaxVector)`.

**Broadcasting closes it a second time, even at the permissive scope.** `Fixed.scala` also
carries `softmaxMatrix`, any rank with the axis passed in. DimWit extends a lower-rank tensor
along the dimensions it is missing **by name**, so reducing `L` away and broadcasting the
result back can only put it on `L` again — whichever axis the caller names, whether or not
the extents coincide. Size broadcasting is rejected outright, and the broadcast is written
down — the `!` on `-!` and `/!` — rather than inferred from position. So the minimal scope
is the argument, and the general case shows the argument does not depend on it.

jaxtyping reaches the same place from the other side: wrapping JAX's permissive softmax in a
strict vector annotation, `Float[Array, "n"] -> Float[Array, "n"]`, removes the defect and
rejects a tile — at run time, on the executed call. See `JAXTYPING_SOLUTION.md`.

**Limit.** This is the accidental loss of an axis, not the deliberate choice of the wrong
one. Reducing over `Axis[Row]` where `Axis[Col]` was meant stays well typed — softmax over
either axis of a tile has the same type — and the same holds for `mean`, `std`, `cumsum` and
layer norm.
