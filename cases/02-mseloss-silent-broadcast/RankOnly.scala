//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 02 (rank-only) — the defect of `Buggy.scala` with positional labels. DOES NOT COMPILE.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case02RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label

  type Rank0 = Tensor0[Float32]
  type Rank1 = Tensor1[A0, Float32]
  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  // pred: [batch, out], target: [batch]
  def mseBuggy(pred: Rank2, target: Rank1): Rank0 =
    // Rank 2 - rank 1 => compile-error
    val residuals: Rank2 = (pred - target).relabelAll(rank2)
    val squared: Rank2 = (residuals * residuals).relabelAll(rank2)
    squared.mean
