# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:22:24
Axis[Case12Fixed.Row] not found in Tensor[(Case12Fixed.Inner, Case12Fixed.Col)].
    a.dot(Axis[Row])(b)
                     ^
```

`einsum` describes a contraction with single letters and `tensordot` with axis positions;
both are legal whenever the two candidate axes happen to share an extent, which square
weight matrices and `batch == seq_len` make routine. A DimWit contraction names the axis it
sums over and that axis must occur in **both** operands, so `a.dot(Axis[Row])(b)` is not a
wrong contraction — it is not a contraction at all, because `b` has no `Row`. The batch axis
is not a letter competing for the same namespace either: it is added by `vmap` around a
matrix product that has no batch in its signature.
