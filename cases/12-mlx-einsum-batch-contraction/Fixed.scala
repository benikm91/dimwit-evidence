//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 12 (fixed) — batched contraction, DimWit.
  *
  * Same interface as `jaxtyping_case.py::batched_matmul_fixed`. `dot` names the axis it
  * sums over, and that axis must exist in BOTH operands. The batch axis is not a letter in
  * a subscript string; it is added by `vmap` and is not part of the contraction at all.
  */
object Case12Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Row derives Label
  trait Inner derives Label
  trait Col derives Label

  /** One matrix times one weight matrix. No batch anywhere in sight. */
  def matmul(
      a: Tensor2[Row, Inner, Float32],
      b: Tensor2[Inner, Col, Float32]
  ): Tensor2[Row, Col, Float32] =
    a.dot(Axis[Inner])(b)

  def batchedMatmulFixed(
      a: Tensor3[Batch, Row, Inner, Float32],
      b: Tensor2[Inner, Col, Float32]
  ): Tensor3[Batch, Row, Col, Float32] =
    a.vmap(Axis[Batch])(m => matmul(m, b))

  @main def case12Check(): Unit =
    dimwit.initialize()

    // the coincidence that hides the bug elsewhere: batch == row == inner == 3
    val a = Tensor(Shape(Axis[Batch] -> 3, Axis[Row] -> 3, Axis[Inner] -> 3))
      .fromArray(Array.tabulate(27)(_.toFloat))
    val b = Tensor(Shape(Axis[Inner] -> 3, Axis[Col] -> 4)).fill(1.0f)

    val out = batchedMatmulFixed(a, b)
    assert(out.shape(Axis[Batch]) == 3 && out.shape(Axis[Row]) == 3 && out.shape(Axis[Col]) == 4)
    // with b all ones every output entry is the row sum of a
    val first = out.slice(Axis[Batch].at(0)).slice(Axis[Row].at(0)).toArray
    assert(first.forall(v => math.abs(v - 3.0f) < 1e-4f), s"expected 0+1+2 = 3, got ${first.mkString(",")}")
    println("case12 ok: contracted over Inner, batched by vmap")
