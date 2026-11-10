# 08 — `dataset_to_point_list` reshaped with the pre-transpose sizes

**Source:** [pymc-devs/pymc#7178](https://github.com/pymc-devs/pymc/issues/7178)
· fix: [PR #7180](https://github.com/pymc-devs/pymc/pull/7180)

## The defect

The function moves the sampling dimensions (`chain`, `draw`) to the front and then flattens
them. The transpose and the reshape were written in a single comprehension, so the sizes
fed to `reshape` were read from the object *before* it was transposed:

```python
{vn: da.transpose(*sample_dims, ...).values.reshape((-1, *da.shape[num_sample_dims:]))
 for vn, da in ds.items()}                                   # ^^ pre-transpose shape
```

The fix splits it in two so the reshape sees the transposed array.

Correct whenever `chain, draw` are already the leading dimensions — which they normally are,
which is why it survived. A dataset with dims `("team", "draw", "chain")` reshapes into the
wrong rectangle and the point list is silently scrambled.

## Why it is interesting

The transpose is expressed by **name** and the reshape by **position**, and the two halves
of the same operation disagree. This is the general hazard of `reshape`: it takes a tuple of
integers with no connection to the axes those integers came from.

DimWit has no `reshape`. It has `flatten`, which records the fused axes in the resulting
label (`Chain |*| Draw`), and `unflatten`, which demands a `Shape` whose labels match.

## Supplementary files

| file | what it shows |
|---|---|
| `tensor_annotations_case.py` | *supplementary:* annotated with [tensor_annotations](https://github.com/google-deepmind/tensor_annotations); the run-time transpose permutation is rejected for buggy and fixed alike, and `jnp.reshape` is `Any`, so the point-list type is never checked — the tests assert the exact errors plus a control of the inferred types |
| `TENSOR_ANNOTATIONS_SOLUTION.md` | *supplementary* verdict; run with `./scripts/check_tensor_annotations.sh -v` (not part of `run_all.sh` or the results table) |
