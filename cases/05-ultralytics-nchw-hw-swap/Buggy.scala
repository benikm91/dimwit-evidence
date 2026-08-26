//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 05 (buggy) — YOLO ONNX preprocessing, DimWit. DOES NOT COMPILE, on purpose.
  *
  * The axes and the model input type are the ones from `Fixed.scala`; only the use is wrong.
  */
object Case05Buggy:

  import dimwit.*
  import Case05Fixed.{Batch, Channel, Height, ModelInput, SrcHeight, SrcWidth, Width}

  def preprocessBuggy(
      image: Tensor3[SrcHeight, SrcWidth, Channel, Float32],
      onnxInputShape: Shape[(Batch, Channel, Height, Width)]
  ): ModelInput =
    // `input_shape[2]` was read as the width and `input_shape[3]` as the height. There is
    // no position to misread here, so the mistake has to be made by name — and the buffer
    // that comes out is a (Width, Height, Channel) one => compile-error
    val letterboxed: Tensor3[Height, Width, Channel, Float32] =
      Tensor(
        Shape(
          onnxInputShape.extent(Axis[Width]),
          onnxInputShape.extent(Axis[Height]),
          image.shape.extent(Axis[Channel])
        )
      ).fill(0.0f)

    letterboxed.transpose((Axis[Channel], Axis[Height], Axis[Width])).prependAxis(Axis[Batch])
