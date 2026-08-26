//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 08 (fixed) — flattening the sampling dimensions, DimWit.
  *
  * Same interface as `jaxtyping_case.py::to_point_list_fixed`. There is no `reshape` here:
  * `flatten` records which axes were fused, in which order, in the label `Chain |*| Draw`,
  * so the transpose and the flatten cannot disagree about the layout.
  */
object Case08Fixed:

  import dimwit.*

  trait Chain derives Label
  trait Draw derives Label
  trait Team derives Label

  /** One row per (chain, draw) pair, one column per team. */
  type PointList = Tensor2[Chain |*| Draw, Team, Float32]

  def toPointListFixed(ds: Tensor3[Team, Draw, Chain, Float32]): PointList =
    val ordered = ds.transpose((Axis[Chain], Axis[Draw], Axis[Team]))
    ordered.flatten((Axis[Chain], Axis[Draw]))

  /** The inverse needs a Shape whose labels reconstruct the fused axis. */
  def fromPointList(
      points: PointList,
      chains: AxisExtent[Chain],
      draws: AxisExtent[Draw]
  ): Tensor3[Chain, Draw, Team, Float32] =
    points.unflatten(Axis[Chain |*| Draw], Shape(chains, draws))

  @main def case08Check(): Unit =
    dimwit.initialize()

    // the reporter's layout: dims are (team, draw, chain) = (5, 2, 3)
    val ds = Tensor(Shape(Axis[Team] -> 5, Axis[Draw] -> 2, Axis[Chain] -> 3))
      .fromArray(Array.tabulate(5 * 2 * 3)(_.toFloat))

    val points = toPointListFixed(ds)
    assert(points.shape(Axis[Chain |*| Draw]) == 6, s"6 (chain, draw) pairs, got ${points.shape(Axis[Chain |*| Draw])}")
    assert(points.shape(Axis[Team]) == 5, s"5 teams, got ${points.shape(Axis[Team])}")

    val restored = fromPointList(points, Axis[Chain] -> 3, Axis[Draw] -> 2)
    assert(restored.shape(Axis[Chain]) == 3 && restored.shape(Axis[Draw]) == 2)
    println("case08 ok: (6, 5) point list, and it round-trips")
