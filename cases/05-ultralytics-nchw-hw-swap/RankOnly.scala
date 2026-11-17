//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 05 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case05RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label
  trait A2 derives Label
  trait A3 derives Label

  type Rank3 = Tensor3[A0, A1, A2, Float32]
  type Rank4 = Tensor4[A0, A1, A2, A3, Float32]

  val rank3 = (Axis[A0], Axis[A1], Axis[A2])
  val rank4 = (Axis[A0], Axis[A1], Axis[A2], Axis[A3])

  // image: [srcHeight, srcWidth, channel], onnxInputShape: [batch, channel, height, width]
  def preprocessBuggy(image: Rank3, onnxInputShape: Shape[(A0, A1, A2, A3)]): Rank4 =
    // input_shape[2] read as the width, input_shape[3] as the height: the buffer is
    // (width, height, channel) instead of (height, width, channel), and rank 3 either way
    val letterboxed: Rank3 =
      Tensor(
        Shape(
          Axis[A0] -> onnxInputShape(Axis[A3]),
          Axis[A1] -> onnxInputShape(Axis[A2]),
          Axis[A2] -> image.shape(Axis[A2])
        )
      ).fill(0.0f)

    // HWC -> CHW, then add the batch axis in front
    val chw: Rank3 = letterboxed.transpose((Axis[A2], Axis[A0], Axis[A1])).relabelAll(rank3)
    chw.prependAxis(Axis[A3]).relabelAll(rank4)
