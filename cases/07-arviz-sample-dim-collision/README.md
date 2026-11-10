# 07 — stacking `chain` and `draw` into a dimension called `sample` collided with user data

**Source:** [arviz-devs/arviz#1647](https://github.com/arviz-devs/arviz/pull/1647)
— *"Renamed `sample` dim to `__sample__` when stacking `chain` and `draw`"*

## The defect

ArviZ flattens the two MCMC sampling dimensions, `chain` and `draw`, into one. It named the
result `sample`. Users whose model already had a dimension called `sample` — a very natural
name for observations — ended up with two dimensions of the same name, and subsequent
selection by name picked the wrong one.

The fix renamed the internal dimension to `__sample__`: a **string-namespace workaround**,
which is the only remedy available when dimension names are runtime strings.

## Why it is interesting

Every named-axis system in wide use — xarray, ArviZ, Haliax, einops' pattern strings —
identifies axes by **strings in a single flat namespace**. Two libraries, or a library and
its user, can pick the same string for unrelated concepts, and nothing detects the clash.

DimWit's axis labels are Scala **types**, which live in the ordinary namespace of the
language: `dimwit.evidence.Sample` and `arviz.Sample` are different types even though they
share a short name, and the flattened axis is not a name at all but the structured type
`Chain |*| Draw`. This is the one case in the dossier where DimWit beats *every* other
named-axis system, not just the positional ones.

## Supplementary files

| file | what it shows |
|---|---|
| `tensor_annotations_case.py` | *supplementary:* annotated with [tensor_annotations](https://github.com/google-deepmind/tensor_annotations); stacking needs `jnp.reshape`, which is `Any` in the stubs, so the stacked axis is never derived and the colliding declaration passes mypy and pyright — the tests assert zero errors plus a control of the inferred types |
| `TENSOR_ANNOTATIONS_SOLUTION.md` | *supplementary* verdict; run with `./scripts/check_tensor_annotations.sh -v` (not part of `run_all.sh` or the results table) |
