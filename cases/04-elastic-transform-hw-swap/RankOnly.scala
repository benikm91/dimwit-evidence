//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 04 (rank-only) — the defect of `Buggy.scala` with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case04RankOnly:

  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.stats.Uniform

  trait A0 derives Label
  trait A1 derives Label
  trait A2 derives Label

  type Rank2 = Tensor2[A0, A1, Float32]
  type Rank3 = Tensor3[A0, A1, A2, Float32]

  val rank2 = (Axis[A0], Axis[A1])
  val rank3 = (Axis[A0], Axis[A1], Axis[A2])

  lazy val KEY = Random.Key(7)

  // size: [height, width], result: [height, width, direction]
  def displacementBuggy(alpha: (Float, Float), size: Shape[(A0, A1)]): Rank3 =
    val noise = Uniform(Tensor(size).fill(-1.0f), Tensor(size).fill(1.0f))
    val (kx, ky) = KEY.split2()

    val nx: Rank2 = noise.sample(kx).relabelAll(rank2)
    val ny: Rank2 = noise.sample(ky).relabelAll(rank2)

    // size[0] and size[1] swapped: both are an Int, nothing to reject
    val dx: Rank2 = (nx *! alpha._1 /! size(Axis[A0])).relabelAll(rank2)
    val dy: Rank2 = (ny *! alpha._2 /! size(Axis[A1])).relabelAll(rank2)

    stack(Seq(dx, dy), Axis[A2], afterAxis = Axis[A1]).relabelAll(rank3)
