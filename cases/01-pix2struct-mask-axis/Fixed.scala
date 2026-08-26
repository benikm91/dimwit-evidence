//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 01 (fixed) — Pix2Struct attention mask, DimWit.
  *
  * Interface as in `jaxtyping_case.py::attention_fixed`, with the four axis names moved
  * out of the annotation string and into types.
  */
object Case01Fixed:

  import scala.language.implicitConversions
  import dimwit.*
  import dimwit.Conversions.given
  import dimwit.nn.ActivationFunctions.softmax

  trait Batch derives Label
  trait Heads derives Label
  trait Query derives Label
  trait Key derives Label

  val NEG: Float = -1e9f

  def attentionFixed(
      scores: Tensor4[Batch, Heads, Query, Key, Float32],
      keyPaddingMask: Tensor2[Batch, Key, Float32]
  ): Tensor4[Batch, Heads, Query, Key, Float32] =
    val additive = (1.0f -! keyPaddingMask) *! NEG

    // No axis expansion: `additive` carries `Key`, so it can only land on `Key`.
    val biased = scores +! additive

    biased.vapply(Axis[Key])(softmax)

  @main def case01Check(): Unit =
    dimwit.initialize()

    // `_example()` of plain.py: one batch, one head, three positions, the last is padding.
    val scores = Tensor(
      Shape(Axis[Batch] -> 1, Axis[Heads] -> 1, Axis[Query] -> 3, Axis[Key] -> 3)
    ).fill(0.0f)
    val mask = Tensor(Shape(Axis[Batch] -> 1, Axis[Key] -> 3)).fromArray(Array(1.0f, 1.0f, 0.0f))

    val weights = attentionFixed(scores, mask)

    val onPadding = weights.slice(Axis[Key].at(2)).sum.item
    assert(math.abs(onPadding) < 1e-6f, s"padding must receive zero weight, got $onPadding")

    val total = weights.sum.item
    assert(math.abs(total - 3.0f) < 1e-4f, s"each query row must sum to 1, total was $total")

    println("case01 ok: padded key position receives zero attention weight")
