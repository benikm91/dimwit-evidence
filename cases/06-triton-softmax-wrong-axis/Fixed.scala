//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 06 (fixed) — softmax over the class axis, DimWit.
  *
  * Same interface as `jaxtyping_case.py::softmax_fixed`, minus `dim` and `keep_dims`.
  * Softmax is defined on a vector and lifted with `vmap`, and that is the whole argument:
  * at that scope the operation is well defined without an axis argument, and the reduction
  * that caused the upstream defect has nowhere to go wrong.
  */
object Case06Fixed:

  import dimwit.*
  import dimwit.nn.ActivationFunctions.softmax
  import dimwit.tensor.tensorops.TensorOpsUtil.Broadcast

  trait Row derives Label
  trait Col derives Label

  /** Softmax at its minimal scope, written out.
    *
    * The reductions still drop a dimension — `max` and `sum` of a `Tensor1[L, V]` are
    * `Tensor0[V]`, exactly as `keep_dims=False` intended. What cannot happen is the
    * misalignment: a scalar broadcast back onto a `Tensor1[L, V]` has one axis to land on,
    * and it is the axis that was reduced. There is no second axis for it to drift onto, so
    * no flag is needed to pin it down.
    */
  def softmaxVector[L: Label](v: Tensor1[L, Float32]): Tensor1[L, Float32] =
    val z = v -! v.max
    val num = z.exp
    num /! num.sum

  def softmaxFixed(x: Tensor2[Row, Col, Float32]): Tensor2[Row, Col, Float32] =
    x.vmap(Axis[Row])(softmaxVector)

  /** The same operation at the scope Triton used: any rank, with the axis passed in.
    *
    * Not the idiomatic version — `softmaxVector` above is — but worth showing, because even
    * here the defect cannot occur. DimWit broadcasts by extending a lower-rank tensor along
    * the dimensions it is *missing by name*, so reducing `L` away and broadcasting the result
    * back can only put it on `L` again. Which axis the caller names, and whether the extents
    * happen to coincide, makes no difference. The broadcast is also written down rather than
    * inferred from position: the `!` on `-!` and `/!` states the intent.
    */
  def softmaxMatrix[T <: Tuple: Labels, L: Label](
      x: Tensor[T, Float32],
      axis: Axis[L]
  )(using ev: AxisRemover[T, L])(using
      lblR: Labels[ev.RemainingAxes],
      bc: Broadcast[T, ev.RemainingAxes, Float32] { type Out = T }
  ): Tensor[T, Float32] =
    // DimWit automatically broadcasts the scalar to the missing axis (by missing name), no shape error, no wrong axis.
    val num = (x -! x.max(axis)).exp
    num /! num.sum(axis)

  @main def case06Check(): Unit =
    dimwit.initialize()

    // the reporter's 2x2 tile, where every other tool is silent
    val x = Tensor(Shape(Axis[Row] -> 2, Axis[Col] -> 2)).fromArray(Array(0f, 2f, 3f, 1f))
    val out = softmaxFixed(x)

    val rowSums = out.sum(Axis[Col]).toArray
    assert(rowSums.forall(s => math.abs(s - 1.0f) < 1e-5f), s"rows must sum to 1, got ${rowSums.mkString(",")}")

    val first = out.slice(Axis[Row].at(0)).toArray
    assert(math.abs(first(0) - 0.1192f) < 1e-3f, s"got ${first.mkString(",")}")

    // the hand-written vector softmax agrees with DimWit's own
    assert(x.vmap(Axis[Row])(softmax).approxEquals(out).item, "must match dimwit.nn softmax")

    // the any-rank version agrees, for either axis
    assert(softmaxMatrix(x, Axis[Col]).approxEquals(out, 1e-5f).item, "along Col must match")
    assert(softmaxMatrix(x, Axis[Row]).sum(Axis[Row]).toArray.forall(s => math.abs(s - 1.0f) < 1e-5f))

    // and a non-square tile, which upstream could not do at all
    val rect = Tensor(Shape(Axis[Row] -> 2, Axis[Col] -> 3)).fromArray(Array.tabulate(6)(_.toFloat))
    assert(softmaxFixed(rect).sum(Axis[Col]).toArray.forall(s => math.abs(s - 1.0f) < 1e-5f))
    assert(softmaxMatrix(rect, Axis[Col]).sum(Axis[Col]).toArray.forall(s => math.abs(s - 1.0f) < 1e-5f))
    assert(softmaxMatrix(rect, Axis[Row]).sum(Axis[Row]).toArray.forall(s => math.abs(s - 1.0f) < 1e-5f))

    println("case06 ok: normalised over Col, rows sum to 1, square and non-square alike")
