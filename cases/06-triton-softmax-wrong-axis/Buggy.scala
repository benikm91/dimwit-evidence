//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 06 (buggy) — softmax with a `keep_dims` flag, DimWit. DOES NOT COMPILE, on purpose.
  *
  * With softmax defined on a vector — see `Case06Fixed.softmaxVector` — the wrong broadcast
  * cannot happen: the reduction still drops the dimension, but a scalar returning to a
  * vector has exactly one axis to land on. `keep_dims` exists only because Python's softmax
  * takes a tensor of any rank, so its reduction leaves a rank-(n-1) result whose axis has to
  * be guessed back, and the guess is positional.
  *
  * Passing such a flag here is not merely wrong, it is not expressible: a DimWit reduction's
  * result type is fixed by the axis it names, and a runtime `Boolean` cannot change a type.
  */
object Case06Buggy:

  import dimwit.*
  import Case06Fixed.{Col, Row}

  def softmaxBuggy(
      x: Tensor2[Row, Col, Float32],
      keepDims: Boolean
  ): Tensor2[Row, Col, Float32] =
    // `z = x - max(x, _dim, keep_dims=keep_dims)` => compile-error: there is no such argument
    // keepDims does not make sense with strict types and minimal scoping.
    val z = x -! x.max(Axis[Col], keepDims)
    val num = z.exp
    num /! num.sum(Axis[Col], keepDims)
