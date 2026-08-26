//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 03 (fixed) — cross product, DimWit.
  *
  * DimWit has no `cross` yet, so it is defined here at the scope the concept actually has:
  * two 3-vectors on one axis. There is nothing for `axisa`, `axisb` and `axisc` to say, and
  * no "first axis of extent 3" to default to. Batching is composed on top with `zipvmap`.
  */
object Case03Fixed:

  import dimwit.*

  /** The x/y/z axis of a 3-vector. */
  trait Spatial derives Label

  /** Rows of the batch. Also has extent 3 in the upstream repro. */
  trait Sample derives Label

  /** The cross product of two 3-vectors on the same axis `L`, which survives into the
    * result: `axisc` is not a parameter, it is `L`. At this minimal scope the
    * implementation is canonical — no axis is left to choose, so there is no default for
    * another library to have chosen differently.
    */
  def cross[L: Label, V: IsNumber](v1: Tensor1[L, V], v2: Tensor1[L, V]): Tensor1[L, V] =
    require(v1.shape(Axis[L]) == 3, s"cross needs a 3-vector, got ${v1.shape(Axis[L])}")
    require(v2.shape(Axis[L]) == 3, s"cross needs a 3-vector, got ${v2.shape(Axis[L])}")

    def at(v: Tensor1[L, V], i: Int): Tensor0[V] = v.slice(Axis[L].at(i))
    val (a1, a2, a3) = (at(v1, 0), at(v1, 1), at(v1, 2))
    val (b1, b2, b3) = (at(v2, 0), at(v2, 1), at(v2, 2))

    stack(Seq(a2 * b3 - a3 * b2, a3 * b1 - a1 * b3, a1 * b2 - a2 * b1), Axis[L])

  @main def case03Check(): Unit =
    dimwit.initialize()

    // the concept itself: two 3-vectors on Spatial
    val u = Tensor1(Axis[Spatial]).fromArray(Array(0f, 1f, 2f))
    val v = Tensor1(Axis[Spatial]).fromArray(Array(8f, 7f, 6f))
    val w = cross(u, v).toArray
    assert(w.sameElements(Array(-8f, 16f, -8f)), s"got ${w.mkString(",")}")

    // the 3x3 input from the upstream repro: both axes have extent 3
    val a = Tensor(Shape(Axis[Sample] -> 3, Axis[Spatial] -> 3))
      .fromArray(Array(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))
    val b = Tensor(Shape(Axis[Sample] -> 3, Axis[Spatial] -> 3))
      .fromArray(Array(8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f, 0f))

    // batching names the axis it maps over; the cross still runs on Spatial
    val out: Tensor2[Sample, Spatial, Float32] =
      zipvmap(Axis[Sample])(a, b)((x: Tensor1[Spatial, Float32], y: Tensor1[Spatial, Float32]) => cross(x, y))

    val r0 = out.slice(Axis[Sample].at(0)).toArray
    assert(r0.sameElements(w), s"got ${r0.mkString(",")}")

    println("case03 ok: mapped over Sample, crossed over Spatial")
