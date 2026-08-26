//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 01 (buggy) — Pix2Struct attention mask, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Interface as in `jaxtyping_case.py::attention_buggy`. Python's `additive[:, None, :, None]`
  * inserts two anonymous singleton axes; DimWit has no anonymous axes, so the same expansion
  * has to name them — and once named, the mask no longer lines up with the scores.
  */
object Case01Buggy:

  import scala.language.implicitConversions
  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.nn.ActivationFunctions.softmax

  trait Batch derives Label
  trait Heads derives Label
  trait Query derives Label
  trait Key derives Label

  val NEG: Float = -1e9f

  def attentionBuggy(
      scores: Tensor4[Batch, Heads, Query, Key, Float32],
      keyPaddingMask: Tensor2[Batch, Key, Float32]
  ): Tensor4[Batch, Heads, Query, Key, Float32] =
    val additive = (1.0f -! keyPaddingMask) *! NEG

    // additive[:, None, :, None]
    val expanded: Tensor4[Batch, Heads, Key, Query, Float32] =
      additive.broadcastTo(
        Shape(
          scores.shape.extent(Axis[Batch]),
          Axis[Heads] -> 1,
          scores.shape.extent(Axis[Key]),
          Axis[Query] -> 1
        )
      )

    // Python adds these by position; DimWit adds them by name, and Query and Key are swapped => compile-error
    val biased: Tensor4[Batch, Heads, Query, Key, Float32] = scores + expanded

    biased.vapply(Axis[Key])(softmax)
