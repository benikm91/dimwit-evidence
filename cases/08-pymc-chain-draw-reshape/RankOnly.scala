//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 08 (rank-only) — the upstream defect with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case08RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label
  trait A2 derives Label

  type Rank2 = Tensor2[A0, A1, Float32]
  type Rank3 = Tensor3[A0, A1, A2, Float32]

  val rank2 = (Axis[A0], Axis[A1])
  val rank3 = (Axis[A0], Axis[A1], Axis[A2])

  // ds: [team, draw, chain], result: [chain * draw, team]
  def toPointListBuggy(ds: Rank3): Rank2 =
    // ordered: [chain, draw, team]
    val ordered: Rank3 = ds.transpose((Axis[A2], Axis[A1], Axis[A0])).relabelAll(rank3)

    // The sample axes are 1 and 2 in `ds`, and 0 and 1 after the transpose. Read before the
    // transpose, as upstream read its sizes, they fuse draw * team: rank 2, nothing to reject
    ordered.flatten((Axis[A1], Axis[A2])).relabelAll(rank2)
