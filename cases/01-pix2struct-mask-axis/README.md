# 01 — Pix2Struct: attention mask broadcast onto the query axis

**Source:** [huggingface/transformers#23974](https://github.com/huggingface/transformers/issues/23974)
· fix: [PR #23976](https://github.com/huggingface/transformers/pull/23976)
· file: `src/transformers/models/pix2struct/modeling_pix2struct.py`

## The defect

The Pix2Struct visual encoder receives a padding mask of shape `[batch, seq]` and has to
extend it to the shape of the attention scores, `[batch, heads, query, key]`. The code
inserted the two singleton axes in the wrong positions:

```python
position_bias = position_bias + attention_mask[:, None, :, None]   # buggy
position_bias = position_bias + attention_mask[:, None, None, :]   # fixed
```

`[:, None, :, None]` gives shape `[batch, 1, seq, 1]`, which broadcasts the mask along the
**query** axis. The intent was to mask **key** positions. The model therefore attended to
padding tokens.

## Why it is interesting

* **It is silent.** Both versions produce `[batch, heads, seq, seq]`. Nothing crashes, no
  warning is emitted, and the shipped model produced plausible captions. The fix had to
  update the expected strings in the test suite — proof that outputs were changing.
* **Sizes cannot separate the axes.** In self-attention `query == key`, so any checker that
  reasons about extents sees two interchangeable `seq` axes. Only the *role* of the axis
  distinguishes them.

## Files

| file | what it shows |
|---|---|
| `plain.py` | NumPy reconstruction, buggy + fixed, with tests |
| `plain_jax.py` | the same in JAX |
| `jaxtyping_case.py` | annotated with jaxtyping; passes for self-attention, catches cross-attention |
| `Buggy.scala` | the mistake in DimWit: `[:, None, :, None]` written out with the two singleton axes named — does not compile |
| `Fixed.scala` | the DimWit program, in which the expansion is not needed at all |
| `PLAIN_SOLUTION.md`, `PLAIN_JAX_SOLUTION.md`, `JAXTYPING_SOLUTION.md`, `DIMWIT_SOLUTION.md` | verdicts |
