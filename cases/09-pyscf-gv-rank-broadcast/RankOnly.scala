//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 09 (rank-only) — the defect of `Buggy.scala` with positional labels. DOES NOT COMPILE.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case09RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label

  type Rank1 = Tensor1[A0, Float32]
  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  // gv: [gpoint, component], centers: [center], result: [gpoint, center]
  def ftAo(gv: Rank2, centers: Rank1): Rank2 =
    gv.vmap(Axis[A0])(v =>
      val g: Rank1 = v.relabelTo(Axis[A0])
      val phase = ((g * g).sum * Tensor0(-0.5f)).exp
      centers *! phase
    ).relabelAll(rank2)

  // g: [component], a single G-vector
  def ftAoBuggy(g: Rank1, centers: Rank1): Rank2 =
    // Rank 1 where rank 2 is declared => compile-error
    ftAo(g, centers)
