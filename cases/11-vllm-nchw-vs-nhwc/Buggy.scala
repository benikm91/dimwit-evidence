//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 11 (buggy) — video frame layout, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The two layouts and the pipeline are the ones from `Fixed.scala`; only the wiring is
  * wrong. The backend was swapped and nobody transposed.
  */
object Case11Buggy:

  import dimwit.*
  import Case11Fixed.{Channel, Nchw, perChannelMeanNhwc}

  def normalise(decoded: Nchw): Tensor1[Channel, Float32] =
    // [2, 3, 3, 3] is the same array under either reading, so NumPy computes something;
    // Nchw and Nhwc are different types whatever the extents => compile-error
    perChannelMeanNhwc(decoded)
