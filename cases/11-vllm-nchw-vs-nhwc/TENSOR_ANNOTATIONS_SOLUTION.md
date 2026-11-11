# tensor_annotations — verdict

**Category: `compile`-equivalent (static rejection) — `Nchw` and `Nhwc` are different types,
compared by axis identity, so the miswiring is rejected whatever the extents.**

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree).

The two layouts are written once as types,
`Nchw = Array4[float32, Frame, Channels, Height, Width]` and
`Nhwc = Array4[float32, Frame, Height, Width, Channels]`, as in DimWit's `Fixed.scala`.
Handing the NCHW backend's output to the NHWC consumer is rejected by both checkers:

```
pyright: Argument of type "Nchw" cannot be assigned to parameter "frames" of type "Nhwc"
mypy:    Argument 1 to "per_channel_mean_nhwc" has incompatible type
           "Array4[float32, Frame, Channels, Height, Width]"; expected "Array4[float32, Frame, Height, Width, Channels]"
```

This is the case where axis identity beats size binding most directly: for the square RGB
frames that shipped, `[2, 3, 3, 3]`, jaxtyping's `frame height width channel` binds to both
readings and passes; here the comparison never looks at sizes. The fix is typed as well:
`jnp.transpose(x, (0, 2, 3, 1))` with a literal permutation infers `Nhwc`, and a wrong
permutation `(0, 3, 2, 1)` gives `[Frame, Width, Height, Channels]` and is rejected.

Two limits:

* **The consumer's reduction is a stub gap.** `frames.mean(axis=(0, 1, 2))`, as
  `plain_jax.py` writes it, fails with "no attribute `mean`": `Array4` has no `mean` method
  in the stubs. It sits in the consumer that buggy and fixed share, so it rejects both
  alike — not a detection. The function form `jnp.mean(frames, axis=(0, 1, 2))` is typed
  (`Array1[Channels]`).
* **A backend's declared layout is not checked.** The decode body builds its array with
  `jnp.asarray` (`Any`), so `-> Nchw` is a declaration. A backend that *declared* `Nhwc`
  while returning NCHW data would be accepted — the layout is only as right as the
  annotation where untyped data enters.

**Compared with the other columns.** jaxtyping misses the shipped square case and, even for
non-square frames, binds `channel` to whatever sits last. tensor_annotations and DimWit both
reject the miswiring statically by identity; DimWit additionally spells the transpose with
axis names instead of a permutation tuple.
