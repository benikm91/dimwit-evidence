//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 06 (rank-only) — softmax without a `keep_dims` flag, with positional labels. COMPILES.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  * The two functions are `softmaxVector` and `softmaxMatrix` of `Fixed.scala`.
  *
  * `softmaxMatrix` is right over axis 1 and wrong over axis 0: the reduced vector is rank 1,
  * and rank 1 broadcasts onto axis 0 whichever axis it came from. See the `@main` below.
  */
object Case06RankOnly:

  import dimwit.*

  trait A0 derives Label
  trait A1 derives Label

  type Rank1 = Tensor1[A0, Float32]
  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  /** Minimal scope: a scalar has one axis to land on. */
  def softmaxVector(v: Rank1): Rank1 =
    val z: Rank1 = v -! v.max
    val num: Rank1 = z.exp
    num /! num.sum

  /** Any axis of a matrix: reduce it away, then broadcast the rank-1 result back. */
  def softmaxMatrix[L: Label](x: Rank2, axis: Axis[L])(using
      ev: AxisRemover[(A0, A1), L],
      lbl: Labels[ev.RemainingAxes]
  ): Rank2 =
    // whichever axis was reduced, what is left is rank 1
    val max: Rank1 = x.max(axis).retag[Tuple1[A0]]
    val num: Rank2 = (x -! max).exp.relabelAll(rank2)
    val sum: Rank1 = num.sum(axis).retag[Tuple1[A0]]
    (num /! sum).relabelAll(rank2)

  @main def case06RankOnlyCheck(): Unit =
    dimwit.initialize()

    // the reporter's 2x2 tile
    val x: Rank2 = Tensor(Shape(Axis[A0] -> 2, Axis[A1] -> 2)).fromArray(Array(0f, 2f, 3f, 1f))
    def show(name: String, t: Rank2): Unit =
      val rows = (0 until 2).map(i => t.slice(Axis[A0].at(i)).toArray.map(v => f"$v%.4f").mkString("[", ", ", "]"))
      println(f"$name%-28s ${rows.mkString(" ")}  sums over axis 0: ${t.sum(Axis[A0]).toArray.mkString(", ")}  over axis 1: ${t.sum(Axis[A1]).toArray.mkString(", ")}")

    show("vector, vmap over axis 0", x.vmap(Axis[A0])(v => softmaxVector(v.relabelTo(Axis[A0]))).relabelAll(rank2))
    show("vector, vmap over axis 1", x.vmap(Axis[A1])(v => softmaxVector(v)).relabelAll(rank2))
    show("matrix, axis 1", softmaxMatrix(x, Axis[A1]))
    show("matrix, axis 0", softmaxMatrix(x, Axis[A0]))
