# 12 — batched tensordot contracted the wrong axis

**Source:** [ml-explore/mlx#4125](https://github.com/ml-explore/mlx/pull/4125)
— *"Fix einsum not broadcasting batch dimensions in batched tensordot"*

## The defect

`einsum` describes a contraction with single letters, and the batch axis is whichever letter
appears on both sides without being summed. MLX's batched `tensordot` path did not broadcast
the batch dimensions, so the contraction was performed against the wrong axis.

Reconstructed here as the underlying hazard rather than the library internals: a batched
matrix product `a[batch, row, inner] @ b[inner, col]` where the implementation contracts
`row` instead of `inner`.

## Why it is interesting

`np.tensordot(a, b, axes=([1], [0]))` and `axes=([2], [0])` are both legal whenever
`a.shape[1] == a.shape[2]`, and they give different answers. Square weight matrices and
`batch == seq_len` make that coincidence routine in transformer code.

In DimWit a contraction names the axis being summed over, and the axis must exist in **both**
operands. `a.dot(Axis[Row])(b)` does not type-check because `b` has no `Row` axis — the
mistake is not merely detected, it is unstateable.

## Supplementary files

| file | what it shows |
|---|---|
| `tensor_annotations_case.py` | *supplementary:* annotated with [tensor_annotations](https://github.com/google-deepmind/tensor_annotations); `jnp.tensordot` and `jnp.einsum` are `Any` in the stubs, so the wrong contraction passes mypy and pyright — the tests assert zero errors plus a control of the inferred types |
| `TENSOR_ANNOTATIONS_SOLUTION.md` | *supplementary* verdict; run with `./scripts/check_tensor_annotations.sh -v` (not part of `run_all.sh`) |
