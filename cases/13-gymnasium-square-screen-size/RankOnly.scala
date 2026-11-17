//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 13 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  * An extent has no axis to be named after, so it is an `Int`.
  */
object Case13RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label

  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  // The `cv2.resize` half, width-first. frame: [height, width]
  def innerResize(frame: Rank2, width: Int, height: Int): Rank2 =
    val rows = (0 until height).map(r => r * frame.shape(Axis[A0]) / height)
    val cols = (0 until width).map(c => c * frame.shape(Axis[A1]) / width)
    val picked: Rank2 = frame.slice(Axis[A0].at(rows)).relabelAll(rank2)
    picked.slice(Axis[A1].at(cols)).relabelAll(rank2)

  // The pipeline half, height-first
  def resizeBuggy(frame: Rank2, height: Int, width: Int): Rank2 =
    // forwarded unflipped: an Int where an Int is declared, nothing to reject
    innerResize(frame, height, width)
