//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 10 (rank-only) — the defect of `Buggy.scala` with positional labels. DOES NOT COMPILE.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  */
object Case10RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label

  type Rank1 = Tensor1[A0, Float32]
  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  // One sequence. idx: [pos], emb: [vocab, embed], wOut: [embed, vocab], result: [pos, vocab]
  def model(idx: Tensor1[A0, Int32], emb: Rank2, wOut: Rank2): Rank2 =
    val x: Rank2 = emb.take(Axis[A0])(idx).relabelAll(rank2)
    val mean: Rank1 = x.mean(Axis[A0]).relabelTo(Axis[A0])
    // mean[None, :]: the mean over positions is placed on axis 1
    val centred: Rank2 = (x -! mean.relabelTo(Axis[A1])).relabelAll(rank2)
    centred.vmap(Axis[A0])(row => row.relabelTo(Axis[A0]).dot(Axis[A0])(wOut)).relabelAll(rank2)

  // idx: [batch, pos]
  def nextTokenLogitsBuggy(idx: Tensor2[A0, A1, Int32], emb: Rank2, wOut: Rank2): Rank2 =
    // Rank 2 where rank 1 is declared => compile-error
    model(idx, emb, wOut)
