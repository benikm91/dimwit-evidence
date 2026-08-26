//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 04 (fixed) — elastic displacement normalisation, DimWit.
  *
  * Same interface as `jaxtyping_case.py::displacement_fixed`. What jaxtyping can only write
  * as `size: tuple[int, int]` is a `Shape[(Height, Width)]` here, so the divisor is chosen
  * by naming an axis rather than by indexing a position.
  */
object Case04Fixed:

  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.stats.Uniform

  trait Height derives Label
  trait Width derives Label

  /** The two displacement channels, (dx, dy) — Python's trailing axis of size 2. */
  trait Direction derives Label

  // lazy: the key is a JAX value, so it must not be built before `dimwit.initialize()`
  lazy val KEY = Random.Key(7)

  def displacementFixed(
      alpha: (Float, Float),
      size: Shape[(Height, Width)]
  ): Tensor3[Height, Width, Direction, Float32] =
    val noise = Uniform(Tensor(size).fill(-1.0f), Tensor(size).fill(1.0f))
    val (kx, ky) = KEY.split2()

    // horizontal displacement is a fraction of the Width, vertical one of the Height
    val dx = noise.sample(kx) *! alpha._1 /! size(Axis[Width])
    val dy = noise.sample(ky) *! alpha._2 /! size(Axis[Height])

    stack(Seq(dx, dy), Axis[Direction], Axis[Width])

  @main def case04Check(): Unit =
    dimwit.initialize()

    // the reporter's 100 x 800 image
    val size = Shape(Axis[Height] -> 100, Axis[Width] -> 800)
    val field = displacementFixed((50.0f, 50.0f), size)
    assert(field.shape == Shape(Axis[Height] -> 100, Axis[Width] -> 800, Axis[Direction] -> 2), s"got ${field.shape}")

    // over 80000 uniform samples the extreme is within a whisker of the scale factor, so
    // the ratio of the two extremes is (alpha/width) / (alpha/height) = height/width
    val dxMax = field.slice(Axis[Direction].at(0)).abs.max.item
    val dyMax = field.slice(Axis[Direction].at(1)).abs.max.item
    val ratio = dxMax / dyMax

    assert(math.abs(ratio - 100f / 800f) < 1e-2f, s"horizontal must use the width: ratio $ratio")
    println("case04 ok: horizontal normalised by Width, vertical by Height")
