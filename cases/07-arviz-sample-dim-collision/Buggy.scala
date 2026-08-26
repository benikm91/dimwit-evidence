//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 07 (buggy) — stacking chain and draw, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Upstream, the stacked axis and the user's own axis both answered to `sample`, so a
  * routine written for the user's samples silently received the stacked one. Here the two
  * are different types and the substitution is a type error.
  */
object Case07Buggy:

  import dimwit.*
  import Case07Fixed.{Chain, Draw, Sample, Stacked, meanOverUserSamples, stackFixed}

  /** A downstream routine written against the USER's sample dimension. */
  def summariseUserSamples(perSample: Tensor1[Sample, Float32]): Tensor0[Float32] =
    perSample.mean

  def stackBuggy(posterior: Tensor3[Chain, Draw, Sample, Float32]): Tensor0[Float32] =
    val stacked: Stacked = stackFixed(posterior)
    // averaging over Sample leaves one value per stacked draw, and `sample` no longer
    // means what the downstream routine means by it => compile-error
    summariseUserSamples(meanOverUserSamples(stacked))
