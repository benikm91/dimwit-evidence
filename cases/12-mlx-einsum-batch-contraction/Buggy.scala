//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 12 (buggy) — batched contraction, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The axes are the ones from `Fixed.scala`. `tensordot(a, b, axes=([1], [0]))` contracts
  * the row axis instead of the inner one, which NumPy allows whenever the two happen to
  * have the same extent.
  */
object Case12Buggy:

  import dimwit.*
  import Case12Fixed.{Col, Inner, Row}

  def matmulBuggy(
      a: Tensor2[Row, Inner, Float32],
      b: Tensor2[Inner, Col, Float32]
  ): Tensor2[Row, Col, Float32] =
    // a contraction names the axis it sums over, and that axis has to exist in both
    // operands; `b` has no Row axis, so there is no contraction to form => compile-error
    a.dot(Axis[Row])(b)
