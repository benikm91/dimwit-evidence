# 11 — a video backend returned NCHW where the pipeline expected NHWC

**Source:** [vllm-project/vllm#51076](https://github.com/vllm-project/vllm/pull/51076)
— *"[Bugfix][Multimodal] Fix PyNvVideoCodec video backend returning NCHW instead of NHWC"*

## The defect

Two decoder backends for the same interface disagreed about channel placement. One returned
`[frames, channels, height, width]`, the other `[frames, height, width, channels]`. The
consumer was written for one of them.

## Why it is interesting

The layout is a **contract carried entirely by prose**. Nothing in a `jnp.ndarray` or a
`torch.Tensor` records whether axis 1 is a channel or a row, so swapping the backend swaps
the meaning of every subsequent axis index with no diagnostic.

And it is silent in the most common configuration: a square frame with 3 channels
(`[2, 3, 3, 3]`) has *exactly the same shape* under both readings, so a per-channel
normalisation computes statistics over the wrong axes and produces plausible numbers.

DimWit writes the layout once, as a type: `Tensor4[Frame, Channel, Height, Width, Float32]`
and `Tensor4[Frame, Height, Width, Channel, Float32]` are different types regardless of
their extents.
