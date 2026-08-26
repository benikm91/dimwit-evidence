//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 14 (fixed) — a (width, height) API meeting a (height, width) API, DimWit.
  *
  * `innerResize` is the `cv2.resize` half: it takes its extents **width-first**.
  * `resizeFixed` is the pipeline half: **height-first**. The flip between them is the line
  * PR #1312 fixed, and `Buggy.scala` — which omits it — does not compile.
  *
  * The detail that makes the check work is that a resize does not *rename* axes, it resizes
  * them, so the extents carry the frame's own labels. `innerResize` stays generic in `H` and
  * `W` — it is as layout-agnostic as `cv2.resize` is — but the `frame` argument anchors both
  * variables, so an extent can only be applied to the axis it names. There is no fresh
  * output label for a mistake to hide behind, and no `relabelAll` on the way out.
  */
object Case14Fixed:

  import dimwit.*

  trait Height derives Label
  trait Width derives Label

  /** The `cv2.resize` analogue: generic in the frame's two axes, extents ordered
    * width-first. Layout-agnostic, but not label-agnostic — `width` must be an extent of
    * whichever axis the frame calls its second, and `height` of its first.
    */
  def innerResize[H: Label, W: Label](
      frame: Tensor2[H, W, Float32],
      width: AxisExtent[W],
      height: AxisExtent[H]
  ): Tensor2[H, W, Float32] =
    val rows = (0 until height.size).map(r => r * frame.shape(Axis[H]) / height.size)
    val cols = (0 until width.size).map(c => c * frame.shape(Axis[W]) / width.size)
    frame
      .slice(Axis[H].at(rows))
      .slice(Axis[W].at(cols))

  /** The pipeline-facing half: height-first, flipping the extents at the boundary. */
  def resizeFixed(
      frame: Tensor2[Height, Width, Float32],
      height: AxisExtent[Height],
      width: AxisExtent[Width]
  ): Tensor2[Height, Width, Float32] =
    innerResize(frame, width, height)

  @main def case14Check(): Unit =
    dimwit.initialize()

    val frame = Tensor(Shape(Axis[Height] -> 210, Axis[Width] -> 160))
      .fromArray(Array.tabulate(210 * 160)(_.toFloat))

    val rect = resizeFixed(frame, Axis[Height] -> 84, Axis[Width] -> 64)
    assert(rect.shape.labels == List("Height", "Width"), "axis order preserved")
    assert(rect.shape.dimensions == List(84, 64), "84 tall by 64 wide, as asked")

    val square = resizeFixed(frame, Axis[Height] -> 84, Axis[Width] -> 84)
    assert(square.shape.dimensions == List(84, 84))

    println("case14 ok: extents flipped at the boundary; the declared order is the delivered order")
