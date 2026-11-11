# tensor_annotations — verdict

**Category: `compile`-equivalent (static rejection) — the missing batch axis is a rank
error, and rank is part of the class (`Array1` vs `Array2`).**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The model is typed `idx: Array2[int32, Batch, Seq]` and `generate`'s token vector
`idx: Array1[int32, Seq]` — both are plain signature annotations. The upstream call is
reconstructed with its conditional slice:

```python
idx_cond = idx if T <= max_len else idx[-max_len:]
logits = model(idx_cond, emb, w_out)
```

Slicing is `Any`, so `idx_cond` is `Array1[int32, Seq] | Any`. Unlike everywhere else in
this column, the `Any` does not swallow the defect: a union is checked member by member,
and the `Array1` member is not an `Array2`.

```
mypy:    Argument 1 to "model" has incompatible type "Array1[int32, Seq] | Any";
           expected "Array2[int32, Batch, Seq]"
pyright: Argument of type "TokenVector | Any" cannot be assigned to parameter "idx" of type "Tokens"
```

Static, and — as for every rank error — independent of sizes. Two limits:

* **It needs the token vector's type at the boundary.** Here `generate` receives it as a
  typed parameter. A vector produced by `jnp.array(...)` or by indexing alone would be `Any`
  and accepted (case 09).
* **The fix cannot be written typed.** `idx_cond.reshape(1, -1)` — JAX's spelling of
  upstream's `.view(1, -1)` — is rejected: `Array1` has no `reshape` method in the stubs (a
  stub gap, as in cases 06 and 09). `jnp.reshape(idx_cond, (1, -1))` is accepted, but as
  `Any`. Inside the model, `emb[idx]` and the positional `x.mean(axis=0)` are untyped too,
  so the residual gap jaxtyping leaves — the body using the wrong axis — remains here.

**Compared with the other columns.** jaxtyping catches the same rank error at run time;
tensor_annotations catches it statically. DimWit avoids it by construction: the model is
written for one sequence and lifted with `vmap`, so there is no batch axis to forget, and
the body's reductions name their axis.
