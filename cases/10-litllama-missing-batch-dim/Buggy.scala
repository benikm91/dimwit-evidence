//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 10 (buggy) — next-token logits, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The model is the one from `Fixed.scala`; only the call is wrong. Upstream, a token
  * vector was handed to a model written for `[batch, seq]` and every operation inside it
  * was rank-polymorphic enough to accept it. Here rank and axis names are both in the type.
  */
object Case10Buggy:

  import dimwit.*
  import Case10Fixed.{Batch, Embed, Pos, Vocab, model}

  def nextTokenLogitsBuggy(
      idx: Tensor2[Batch, Pos, Int32],
      emb: Tensor2[Vocab, Embed, Float32],
      wOut: Tensor2[Embed, Vocab, Float32]
  ): Tensor2[Pos, Vocab, Float32] =
    // a batch of sequences passed straight into the single-sequence model => compile-error
    model(idx, emb, wOut)
