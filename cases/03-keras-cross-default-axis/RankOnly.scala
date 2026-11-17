//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 03 (rank-only) — the upstream defect with positional labels. DOES NOT COMPILE.
  *
  * Axes are named after their position, and every result is relabelled back to
  * `A0, A1, ...`, so a type states the rank of a tensor and nothing else.
  * `cross` is the one defined in `Fixed.scala`: rank 1 x rank 1 => rank 1.
  */
object Case03RankOnly:

  import dimwit.*
  import Case03Fixed.cross

  trait A0 derives Label
  trait A1 derives Label

  type Rank2 = Tensor2[A0, A1, Float32]

  val rank2 = (Axis[A0], Axis[A1])

  // a, b: [sample, spatial]
  def crossBatch(a: Rank2, b: Rank2): Rank2 =
    // keras.ops.cross(a, b): `cross` has no axis to default to, and rank 2 is not
    // rank 1 => compile-error
    cross(a, b).relabelAll(rank2)
