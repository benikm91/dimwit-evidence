//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 03 (buggy) — cross product, DimWit. DOES NOT COMPILE, on purpose.
  *
  * `cross` is the one defined in `Fixed.scala`; only the use is wrong here.
  */
object Case03Buggy:

  import dimwit.*
  import Case03Fixed.{cross, Sample, Spatial}

  /** Another axis of extent 3. `torch.cross` cannot tell it from `Spatial`. */
  trait Rgb derives Label

  def crossBatch(
      a: Tensor2[Sample, Spatial, Float32],
      b: Tensor2[Sample, Rgb, Float32]
  ): Tensor2[Sample, Spatial, Float32] =
    // Every axis here has extent 3, so `torch.cross` finds a length-3 axis in both operands
    // and returns a number. `cross` needs the two vectors on the same axis, and the names
    // carry the semantics: a spatial vector and a colour triple cannot be crossed by
    // accident, however well their extents happen to line up => compile-error
    zipvmap(Axis[Sample])(a, b)((x: Tensor1[Spatial, Float32], y: Tensor1[Rgb, Float32]) => cross(x, y))
