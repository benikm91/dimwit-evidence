//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 04 (buggy) — elastic displacement normalisation, DimWit. THIS ONE COMPILES.
  *
  * `size[0]` cannot be written, so the author has to name the axis — and then divides the
  * horizontal displacement by `Axis[Height]` in as many words. Wrong in a way a reader can
  * see, but `size(Axis[L])` hands back a plain `Int`, so nothing checks it.
  */
object Case04Buggy:

  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.stats.Uniform
  import Case04Fixed.{Direction, Height, KEY, Width}

  def displacementBuggy(
      alpha: (Float, Float),
      size: Shape[(Height, Width)]
  ): Tensor3[Height, Width, Direction, Float32] =
    val noise = Uniform(Tensor(size).fill(-1.0f), Tensor(size).fill(1.0f))
    val (kx, ky) = KEY.split2()

    // Named dimensions makes bug more obvious (x - Height), but compiles
    val dx = noise.sample(kx) *! alpha._1 /! size(Axis[Height])
    val dy = noise.sample(ky) *! alpha._2 /! size(Axis[Width]) 

    stack(Seq(dx, dy), Axis[Direction], afterAxis = Axis[Width])
