# tensor_annotations — verdict

**Category: `compile`-equivalent (static rejection) — the rank error is in the interface
(`Gv: Array2[GPoint, Component]`; a vector is `Array1`), checked at every call whose
argument is typed. Missed only when the vector reaches the call as `Any`, e.g. from
indexing (`GRID[1]`) or `jnp.array(...)`.**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

This is a rank error, and rank is the one thing tensor_annotations puts into the class
itself: `Array1` and `Array2` are different classes, so with `Gv: Array2[GPoint, Component]`
a rank-1 argument is a type error — whenever its class is known. Whether it is depends
entirely on where the vector comes from:

| how the caller gets the single G-vector | type at the call | result |
|---|---|---|
| `g = GRID[1]` — as in `plain.py` / `plain_jax.py` | `Any` (`__getitem__ -> Any`) | **missed** |
| `jnp.array([1.0, 0.0, 0.0])` / `jnp.asarray(...)` | `Any` (`array: Any`, `asarray: Any`) | **missed** |
| a typed parameter `g: Array1[float32, Component]` — as DimWit's `Buggy.scala` writes it | `Array1[float32, Component]` | **rejected** |
| `jnp.zeros((3,))` | `Array1[float32, Any]` — typed by rank only | **rejected**: `Any` axes do not rescue a wrong rank |

The verdict is judged as in case 10 and as DimWit's `Buggy.scala` writes it: the caller
holds the single vector as a typed value (`g: Array1[float32, Component]`), and the call is
rejected statically, independent of sizes. The check lives in `ft_ao`'s interface, not in
its body — removing the `jnp.asarray` line from the body changes nothing at the call.

What gets through is an argument that is already `Any` before the call. `plain.py` takes the
vector from a grid by indexing (`GRID[1]`), and every array written out as a literal
(`jnp.array`) is `Any` too; both are accepted. That miss belongs to how the caller produced
the vector, not to what `ft_ao` declares.

Two further observations:

* **The body is untyped** in either version (`jnp.asarray` is `Any`), so the defect inside
  `ft_ao` — reducing the only axis — is not visible; only the signature can help.
* **The caller-side fix is rejected too** on a typed vector: `g.reshape(1, 3)` fails with
  `"Array1[float32, Component]" has no attribute "reshape"`. A stub gap, as `x.max` in
  case 06 — and the function form, `jnp.reshape`, is `Any`. Adding the grid axis cannot be
  written typed at all; compare DimWit's `prependAxis(Axis[GPoint])`.

**Compared with the other columns.** jaxtyping catches this at run time, reliably, because
it checks the rank of the actual array whatever its static type. tensor_annotations checks
rank statically but only for arrays whose class survived to the call — and indexing and
literals do not survive. DimWit has neither gap: indexing yields typed tensors, and the
rank error is a compile error.
