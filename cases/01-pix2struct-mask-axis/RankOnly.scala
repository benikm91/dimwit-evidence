//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 01 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case01RankOnly:

  import scala.language.implicitConversions
  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.nn.ActivationFunctions.softmax

  trait A0 derives Label
  trait A1 derives Label
  trait A2 derives Label
  trait A3 derives Label

  type Rank2 = Tensor2[A0, A1, Float32]
  type Rank4 = Tensor4[A0, A1, A2, A3, Float32]

  val rank2 = (Axis[A0], Axis[A1])
  val rank4 = (Axis[A0], Axis[A1], Axis[A2], Axis[A3])

  val NEG: Float = -1e9f

  // scores: [batch, heads, query, key], keyPaddingMask: [batch, key]
  def attentionBuggy(scores: Rank4, keyPaddingMask: Rank2): Rank4 =
    val inverted: Rank2 = (1.0f -! keyPaddingMask).relabelAll(rank2)
    val additive: Rank2 = (inverted *! NEG).relabelAll(rank2)

    // additive[:, None, :, None]: the mask lands on axis 2 (query) instead of axis 3 (key)
    val expanded: Rank4 =
      additive.relabelAll((Axis[A0], Axis[A2])).broadcastTo(scores.shape).relabelAll(rank4)

    // Rank 4 + rank 4: nothing to reject
    val biased: Rank4 = (scores + expanded).relabelAll(rank4)
    biased.vapply(Axis[A3])(softmax).relabelAll(rank4)
