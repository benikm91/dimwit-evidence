//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 02 (buggy) — mean squared error, DimWit. DOES NOT COMPILE, on purpose.
  *
  * Both signatures of `jaxtyping_case.py`, as they are there. Neither can reach the
  * upstream defect: the strict one will not accept the `[n, 1]` prediction that
  * `nn.Linear(2, 1)` returns, and the one that declares the `Out` axis cannot subtract
  * across it.
  */
object Case02Buggy:

  import dimwit.*

  trait Batch derives Label

  /** The `out_features = 1` axis of `nn.Linear(2, 1)`. */
  trait Out derives Label

  def mseStrict(pred: Tensor1[Batch, Float32], target: Tensor1[Batch, Float32]): Tensor0[Float32] =
    val residuals = pred - target
    (residuals * residuals).mean

  def mseAsWrittenUpstream(
      pred: Tensor2[Batch, Out, Float32],
      target: Tensor1[Batch, Float32]
  ): Tensor0[Float32] =
    // The model's prediction is not the argument the strict loss declares => compile-error
    mseStrict(pred, target)

    // ... and PyTorch's own broadcast to [n, n] is not available either, because `-` is
    // shape-exact; the broadcast would have to be asked for by name, as `-!` => compile-error
    val residuals: Tensor2[Batch, Out, Float32] = pred - target
    (residuals * residuals).mean
