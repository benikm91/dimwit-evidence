//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 12 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case12RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label

  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  // a: [row, inner], b: [inner, col], result: [row, col]
  def matmulBuggy(a: Rank2, b: Rank2): Rank2 =
    // tensordot(a, b, axes=([0], [0])): contracts the row axis of `a` instead of its inner
    // one. Both operands have an axis 0, so there is a contraction to form
    a.dot(Axis[A0])(b).relabelAll(rank2)
