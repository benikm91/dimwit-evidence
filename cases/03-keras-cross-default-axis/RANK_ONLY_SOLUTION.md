# DimWit, rank-only — verdict

**Category: `compile-time detection`** — at DimWit's minimal scope for `cross`.

```
./RankOnly.scala:27:14
Found:    (b : Case03RankOnly.Rank2)
Required: dimwit.tensor.Tensor1[L, V]
    cross(a, b).relabelAll(rank2)
             ^
```

`RankOnly.scala` is the upstream defect in the same library and with the same compiler, but
with the semantic names removed: an axis is labelled by its position (`A0`, `A1`, ...), and
the result of every operation is relabelled back to those positions. A type then states the
rank of a tensor and nothing else.

The upstream call hands `cross` two matrices and leaves the axis to a default. `cross` as
`Fixed.scala` defines it is a function of two vectors, rank 1 x rank 1 => rank 1. It has no
axis argument and so no default, and a rank-2 argument is a rank error. The caller has to
lift it with `zipvmap` and say which axis is mapped. That choice is then made at the call,
in the open, and is no longer a property of `cross`.

The names are not needed for this. The defect is removed by the scope of `cross`, and rank
typing is enough to enforce that scope.

**Limits.**

* **This holds for rank-only typing with DimWit's minimal scoping, not for rank typing as
  such.** A rank-typed `cross(a, b, axis)` over arrays of any rank, the interface every
  array library ships, accepts either axis.
* **What names add is a second check.** `Buggy.scala` crosses a `Spatial` vector with an
  `Rgb` triple, which is rejected with names (`DIMWIT_SOLUTION.md`). Under positional labels
  both are `Tensor1[A0]` and the call compiles. That mistake was constructed for the case;
  it is not what keras#23219 reports.
* **The lift is the caller's choice in both variants.** `zipvmap(Axis[A1])` in place of
  `zipvmap(Axis[A0])` compiles under positional labels.
