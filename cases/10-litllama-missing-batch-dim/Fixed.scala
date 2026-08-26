//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 10 (fixed) — next-token logits, DimWit.
  *
  * Same interface as `jaxtyping_case.py`, except that `model` is written for ONE sequence.
  * `Batch` never appears in its signature, so it cannot be forgotten; batching is `vmap`,
  * from the outside, and `idx.view(1, -1)` has nothing to correspond to.
  */
object Case10Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Pos derives Label
  trait Embed derives Label
  trait Vocab derives Label

  /** One sequence in, one row of logits per position out. */
  def model(
      idx: Tensor1[Pos, Int32],
      emb: Tensor2[Vocab, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Pos, Vocab, Float32] =
    val x = emb.take(Axis[Vocab])(idx)
    // centring across positions, stated as such rather than as `axis=0`
    val centred = x -! x.mean(Axis[Pos])
    centred.vmap(Axis[Pos])(row => row.dot(Axis[Embed])(wOut))

  /** The logits for the position we are about to extend. Named, not `-1`. */
  def nextTokenLogitsFixed(
      idx: Tensor1[Pos, Int32],
      emb: Tensor2[Vocab, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor1[Vocab, Float32] =
    val logits = model(idx, emb, wOut)
    logits.slice(Axis[Pos].at(logits.shape(Axis[Pos]) - 1))

  /** Batched generation, if it is wanted, is a lift of the single-sequence function. */
  def batchedLogits(
      idx: Tensor2[Batch, Pos, Int32],
      emb: Tensor2[Vocab, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Batch, Vocab, Float32] =
    idx.vmap(Axis[Batch])(seq => nextTokenLogitsFixed(seq, emb, wOut))

  @main def case10Check(): Unit =
    dimwit.initialize()

    val emb = Tensor(Shape(Axis[Vocab] -> 7, Axis[Embed] -> 4)).fromArray(Array.tabulate(28)(_.toFloat))
    val wOut = Tensor(Shape(Axis[Embed] -> 4, Axis[Vocab] -> 7)).fill(0.5f)
    val one = Tensor1(Axis[Pos]).fromArray(Array(1, 3, 5, 2))

    assert(nextTokenLogitsFixed(one, emb, wOut).shape(Axis[Vocab]) == 7)

    val many = Tensor(Shape(Axis[Batch] -> 2, Axis[Pos] -> 4))
      .fromArray(Array(1, 3, 5, 2, 0, 2, 4, 6))
    val batched = batchedLogits(many, emb, wOut)
    assert(batched.shape(Axis[Batch]) == 2 && batched.shape(Axis[Vocab]) == 7)
    println("case10 ok: single-sequence model, batching added by vmap")
