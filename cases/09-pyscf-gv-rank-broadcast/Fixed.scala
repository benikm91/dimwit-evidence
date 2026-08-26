//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 09 (fixed) — Fourier transform over a grid of G-vectors, DimWit.
  *
  * Upstream the routine's *body* assumed `Gv` was `(N, 3)` while its signature accepted
  * anything, so a bare `(3,)` reduced one axis too many and the result broadcast as a
  * scalar. Here the assumption is the signature: a grid of G-vectors is
  * `Tensor2[GPoint, Component, Float32]`, and a caller holding one vector has to say so.
  */
object Case09Fixed:

  import dimwit.*

  /** Reciprocal-space grid point. */
  trait GPoint derives Label

  /** x / y / z of a reciprocal-space vector. */
  trait Component derives Label

  /** Basis-function centre. */
  trait Center derives Label

  def ftAoFixed(
      gv: Tensor2[GPoint, Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor2[GPoint, Center, Float32] =
    gv.vmap(Axis[GPoint]): g =>
      val phase = ((g * g).sum * Tensor0(-0.5f)).exp
      centers *! phase

  @main def case09Check(): Unit =
    dimwit.initialize()

    val centers = Tensor1(Axis[Center]).fromArray(Array(0f, 0.5f, 1f, 1.5f))
    val grid = Tensor(Shape(Axis[GPoint] -> 3, Axis[Component] -> 3))
      .fromArray(Array(0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f, 1f))

    val onGrid = ftAoFixed(grid, centers)
    assert(onGrid.shape(Axis[GPoint]) == 3 && onGrid.shape(Axis[Center]) == 4)

    // A caller holding a single G-vector adds the grid axis explicitly. This is the
    // counterpart of `Gv.reshape(-1, 3)` — except that it names the axis it is adding, and
    // the routine could not have been called without it.
    val one: Tensor1[Component, Float32] = grid.slice(Axis[GPoint].at(1))
    val asGrid: Tensor2[GPoint, Component, Float32] = one.prependAxis(Axis[GPoint])

    val single = ftAoFixed(asGrid, centers)
    assert(single.shape(Axis[GPoint]) == 1 && single.shape(Axis[Center]) == 4, s"got ${single.shape}")
    assert(single.approxEquals(onGrid.slice(Axis[GPoint].at(1)).prependAxis(Axis[GPoint])).item,
      "one vector must give the row the grid gives")

    println("case09 ok: the grid axis is in the type, and a single vector has to add it by name")
