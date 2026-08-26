//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 02 (fixed) — mean squared error, DimWit.
  *
  * `jaxtyping_case.py::mse_strict`, with `"batch"` moved out of the annotation string and
  * into the type. `mse_as_written_upstream` has no counterpart here: see `Buggy.scala`.
  */
object Case02Fixed:

  import dimwit.*

  trait Batch derives Label

  def mseStrict(pred: Tensor1[Batch, Float32], target: Tensor1[Batch, Float32]): Tensor0[Float32] =
    val residuals = pred - target
    (residuals * residuals).mean

  @main def case02Check(): Unit =
    dimwit.initialize()

    val target = Tensor1(Axis[Batch]).fromArray(Array(2f, 3f, 5f, 7f))

    val exact = mseStrict(target, target).item
    assert(exact < 1e-6f, s"identical vectors must give zero loss, got $exact")

    val off = Tensor1(Axis[Batch]).fromArray(Array(3f, 4f, 6f, 8f))
    val one = mseStrict(off, target).item
    assert(math.abs(one - 1.0f) < 1e-6f, s"a residual of 1 everywhere must give 1.0, got $one")

    println(s"case02 ok: loss with the true parameters is $exact")
