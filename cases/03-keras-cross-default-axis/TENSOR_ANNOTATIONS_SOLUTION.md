# tensor_annotations — verdict

**Category: `compile`-equivalent (static rejection) at the minimal interface — `cross` of
two 3-vectors on one axis, as DimWit defines it. `missed` at the upstream scope, where
`cross` takes an axis argument and `jnp.cross` is `Any`.**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The case is judged the way DimWit's is: the defect needs a permissive scope to exist in,
and the comparison is made at the scope the concept actually has.

**Minimal scope — rejected.** Mirroring `Fixed.scala`:

```python
def cross(v1: Array1[DT, L], v2: Array1[DT, L]) -> Array1[DT, L]: ...
```

Two vectors on the same axis `L`, which survives into the result; there is no `axisa`,
`axisb`, `axisc` and no default to disagree about. Batching is a per-sample function mapped
over `Sample`. DimWit's `Buggy.scala` — a spatial vector crossed with a colour triple, every
axis of extent 3 — is rejected on the same line by both checkers:

```
mypy:    Argument 2 to "cross" has incompatible type "Array1[float32, Rgb]"; expected "Array1[float32, Spatial]"
pyright: Argument of type "Array1[float32, Rgb]" cannot be assigned to parameter "v2" of type "Array1[DT@cross, L@cross]"
```

Axes are compared by identity, so the shared extent 3 is irrelevant — where jaxtyping's
size-bound names cannot tell the axes apart. The body (`jnp.cross` on two vectors) is
untyped, but at this scope it has no axis to choose.

**One limit at the minimal scope: the lift is unchecked.** `jax.vmap` is `Any` in the
stubs, so the batching step is not checked: mapping over `Spatial` instead of `Sample`
(`in_axes=1`) is accepted and silently wrong (`cross_batch_wrong_lift`). DimWit's
`zipvmap(Axis[Sample])` names the axis it maps over.

**Upstream scope — missed.** With `cross` over a whole array and an axis argument, the
stubs declare `cross: Any`: buggy (first axis of size 3) and fixed (`axis=-1`) are both
accepted, as is a deliberately swapped return annotation. The axis search is typed
(`a.shape: Tuple[int, int]`) but yields an `int`, which carries no axis identity. Even a
typed shape-preserving `cross` stub would not help here: crossing either axis gives the
same result type.

The tests assert exactly the one error above in both checkers, a control (`a` typed;
`jnp.cross` and `jax.vmap` `Any`), and the run-time behaviour of every variant.
