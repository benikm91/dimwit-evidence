//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 07 (fixed) — stacking chain and draw, DimWit.
  *
  * Flattening two axes does not invent a name: it produces the structured label
  * `Chain |*| Draw`. The user's own `Sample` axis is a different type, so `stackBuggy` —
  * "call the stacked dimension `sample`" — has nothing to correspond to.
  */
object Case07Fixed:

  import dimwit.*

  trait Chain derives Label
  trait Draw derives Label

  /** The user's own dimension, which in the upstream bug shared the name `sample`. */
  trait Sample derives Label

  /** Flattened posterior draws. The axis records where it came from, in its type. */
  type Stacked = Tensor2[Chain |*| Draw, Sample, Float32]

  def stackFixed(posterior: Tensor3[Chain, Draw, Sample, Float32]): Stacked =
    posterior.flatten((Axis[Chain], Axis[Draw]))

  /** Averages over the USER's samples, leaving one value per stacked draw. */
  def meanOverUserSamples(stacked: Stacked): Tensor1[Chain |*| Draw, Float32] =
    stacked.mean(Axis[Sample])

  /** Averages over the MCMC draws, leaving one value per user sample. */
  def meanOverDraws(stacked: Stacked): Tensor1[Sample, Float32] =
    stacked.mean(Axis[Chain |*| Draw])

  @main def case07Check(): Unit =
    dimwit.initialize()

    val posterior = Tensor(Shape(Axis[Chain] -> 4, Axis[Draw] -> 3, Axis[Sample] -> 5))
      .fromArray(Array.tabulate(4 * 3 * 5)(_.toFloat))

    val stacked = stackFixed(posterior)
    assert(stacked.shape(Axis[Chain |*| Draw]) == 12, "chain and draw must fuse into 12")
    assert(stacked.shape(Axis[Sample]) == 5, "the user's Sample axis must survive untouched")
    assert(meanOverUserSamples(stacked).shape(Axis[Chain |*| Draw]) == 12)
    assert(meanOverDraws(stacked).shape(Axis[Sample]) == 5)
    println("case07 ok: Chain |*| Draw and Sample stayed distinct")
