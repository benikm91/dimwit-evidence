//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 14 (buggy) — DimWit. DOES NOT COMPILE, on purpose.
  *
  * `resizeBuggy` advertises (height, width) and forwards the pair straight into
  * `innerResize`, which expects (width, height). That is precisely the defect PR #1312
  * fixed: a `screen_size` in one convention handed to an API in the other, invisible for as
  * long as the value stayed square.
  *
  * DimWit rejects it, and the reason is worth stating exactly, because it is *not* "the
  * labels differ so the arguments clash". Note that this wrapper is fully generic — no
  * `Height` or `Width` appears in its signature at all. What does the work is that a resize
  * does not rename axes: `innerResize` types its extents with the frame's own labels, so the
  * `frame` argument anchors `H` and `W`, and every extent must then name the axis it is
  * actually resizing. The mistake is caught here, at the definition of a generic helper,
  * without waiting for a call site to supply concrete labels:
  *
  * {{{
  * [error] Found:    (height : dimwit.tensor.AxisExtent[H])
  * [error] Required: dimwit.tensor.AxisExtent[W]
  * [error]     innerResize(frame, height, width)
  * [error]                        ^^^^^^
  * [error] Found:    (width : dimwit.tensor.AxisExtent[W])
  * [error] Required: dimwit.tensor.AxisExtent[H]
  * [error]     innerResize(frame, height, width)
  * [error]                                ^^^^^
  * }}}
  *
  * The contrast that matters for the paper is with a signature that gives the *output* its
  * own fresh labels — `innerResize[A, B, OutW, OutH](frame: Tensor2[A, B, ...], width:
  * AxisExtent[OutW], height: AxisExtent[OutH]): Tensor2[OutH, OutW, ...]`. That version
  * compiles with the extents swapped, because nothing anchors `OutW` and `OutH` and
  * inference simply solves them the wrong way round. It is a plausible thing to write and it
  * silently gives up the check — so the rule the case supports is: **type an operation's
  * extents with the labels of the tensor it operates on.** Introduce fresh axis variables
  * only when the operation genuinely renames an axis, which resizing does not.
  */
object Case14Buggy:

  import dimwit.*
  import Case14Fixed.innerResize

  /** As shipped: advertises (height, width) and forwards it unflipped. */
  def resizeBuggy[H: Label, W: Label](
      frame: Tensor2[H, W, Float32],
      height: AxisExtent[H],
      width: AxisExtent[W]
  ): Tensor2[H, W, Float32] =
    innerResize(frame, height, width)
