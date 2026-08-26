//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 05 (fixed) — YOLO ONNX preprocessing, DimWit.
  *
  * Same interface as `jaxtyping_case.py::preprocess_fixed`. What jaxtyping can only call
  * `tuple[int, int, int, int]` is a `Shape[(Batch, Channel, Height, Width)]` here, so the
  * spatial extents are chosen by name and carry that name with them.
  */
object Case05Fixed:

  import dimwit.*

  trait Batch derives Label
  trait Channel derives Label
  trait Height derives Label
  trait Width derives Label

  /** The source image's own resolution, which is not the model's. */
  trait SrcHeight derives Label
  trait SrcWidth derives Label

  /** The model's declared input. NCHW is in the type, not in a comment. */
  type ModelInput = Tensor4[Batch, Channel, Height, Width, Float32]

  def preprocessFixed(
      image: Tensor3[SrcHeight, SrcWidth, Channel, Float32],
      onnxInputShape: Shape[(Batch, Channel, Height, Width)]
  ): ModelInput =
    val letterboxed: Tensor3[Height, Width, Channel, Float32] =
      Tensor(
        Shape(
          onnxInputShape.extent(Axis[Height]),
          onnxInputShape.extent(Axis[Width]),
          image.shape.extent(Axis[Channel])
        )
      ).fill(0.0f)

    letterboxed.transpose((Axis[Channel], Axis[Height], Axis[Width])).prependAxis(Axis[Batch])

  @main def case05Check(): Unit =
    dimwit.initialize()

    val image = Tensor(Shape(Axis[SrcHeight] -> 1080, Axis[SrcWidth] -> 1920, Axis[Channel] -> 3)).fill(0.0f)
    val rect = Shape(Axis[Batch] -> 1, Axis[Channel] -> 3, Axis[Height] -> 640, Axis[Width] -> 480)

    val input: ModelInput = preprocessFixed(image, rect)

    assert(input.shape(Axis[Height]) == 640, s"height must be 640, got ${input.shape(Axis[Height])}")
    assert(input.shape(Axis[Width]) == 480, s"width must be 480, got ${input.shape(Axis[Width])}")
    println("case05 ok: NCHW input built as (Batch, Channel, Height, Width) = (1, 3, 640, 480)")
