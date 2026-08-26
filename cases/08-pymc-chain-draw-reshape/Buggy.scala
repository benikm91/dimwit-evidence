//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 08 (buggy) — flattening the sampling dimensions, DimWit. DOES NOT COMPILE.
  *
  * The axes and the point-list type are the ones from `Fixed.scala`; only the use is wrong.
  * Upstream, the reshape used sizes read off the dataset before it was transposed, so the
  * fusion followed the dataset's own order rather than the point list's.
  */
object Case08Buggy:

  import dimwit.*
  import Case08Fixed.{Chain, Draw, PointList, Team}

  def toPointListBuggy(ds: Tensor3[Team, Draw, Chain, Float32]): PointList =
    // fused in the dataset's order (draw, chain) instead of the point list's (chain, draw);
    // there is no tuple of integers to get wrong, the fused label records it => compile-error
    val ordered = ds.transpose((Axis[Draw], Axis[Chain], Axis[Team]))
    val fused: Tensor2[Draw |*| Chain, Team, Float32] = ordered.flatten((Axis[Draw], Axis[Chain]))
    fused
