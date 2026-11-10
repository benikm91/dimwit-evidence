# 10 — generation called the model on an unbatched token vector

**Source:** [Lightning-AI/lit-llama#166](https://github.com/Lightning-AI/lit-llama/pull/166)
— *"Fix generate.py missing batch dimension for model"*

## The defect

```python
-        logits = model(idx_cond)
-        logits = logits[-1] / temperature
+        logits = model(idx_cond.view(1, -1))
+        logits = logits[0, -1] / temperature
```

`idx_cond` is a 1-D vector of token ids. The model is written for `[batch, seq]`, but every
operation inside it is rank-polymorphic, so the unbatched call ran — with any op that
assumes the leading axis is the batch now operating over the sequence instead. The indexing
that followed, `logits[-1]`, then read the "last" of whatever axis happened to be leading.

## Why it is interesting

This is the single most common shape fault in the literature. In the SFData corpus
65.8% of the 146 crashing faults are feature-input or label-output
mismatches, which is almost always a batch axis present on one side and absent on the other.

It is also the case that motivates DimWit's central stylistic choice: **layers are written
for one example and lifted with `vmap`**, so `Batch` never appears in a layer signature and
there is no batch axis to forget.

## Supplementary files

| file | what it shows |
|---|---|
| `tensor_annotations_case.py` | *supplementary:* annotated with [tensor_annotations](https://github.com/google-deepmind/tensor_annotations); the unbatched call is rejected statically by mypy and pyright (rank is in the class, even through `Array1 \| Any` from slicing); the fix's `reshape` is untyped — the tests assert the exact errors |
| `TENSOR_ANNOTATIONS_SOLUTION.md` | *supplementary* verdict; run with `./scripts/check_tensor_annotations.sh -v` (not part of `run_all.sh` or the results table) |
