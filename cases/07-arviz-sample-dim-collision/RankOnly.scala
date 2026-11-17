//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 07 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case07RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label
  trait A2 derives Label

  type Rank0 = Tensor0[Float32]
  type Rank1 = Tensor1[A0, Float32]
  type Rank2 = Tensor2[A0, A1, Float32]
  type Rank3 = Tensor3[A0, A1, A2, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  /** A downstream routine written against the USER's sample dimension. */
  def summariseUserSamples(perSample: Rank1): Rank0 =
    perSample.mean

  // posterior: [chain, draw, sample]
  def stackBuggy(posterior: Rank3): Rank0 =
    // stacked: [chain * draw, sample]
    val stacked: Rank2 = posterior.flatten((Axis[A0], Axis[A1])).relabelAll(rank2)

    // averaging over the user's samples leaves one value per stacked draw
    val perDraw: Rank1 = stacked.mean(Axis[A1]).relabelTo(Axis[A0])

    // Rank 1 where rank 1 is expected: nothing to reject
    summariseUserSamples(perDraw)
