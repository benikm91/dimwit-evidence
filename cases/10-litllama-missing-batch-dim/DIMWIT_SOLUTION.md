# DimWit — verdict

**Category: `compile-time detection`**

```
./Buggy.scala:22:11
Found:    (idx : Tensor2[Case10Fixed.Batch, Case10Fixed.Pos, Int32])
Required: Tensor1[Case10Fixed.Pos, Int32]
    model(idx, emb, wOut)
          ^^^
```

The mechanism is not really the rejection — it is that `model` is written for **one
sequence**, so `Batch` never appears in its signature and there is no batch axis to forget.
`idx_cond.view(1, -1)`, the upstream fix, has nothing to correspond to: batching is `vmap`
from the outside, and `logits[-1]` becomes `slice(Axis[Pos].at(...))`, which names the axis
it indexes instead of taking whichever one happens to lead. jaxtyping catches the missing
axis too, at run time, because rank cannot coincide by accident — but it leaves
`x.mean(axis=0)` inside the body positional, so a later refactor that moves the batch axis
passes the annotation and still centres over the wrong one.
