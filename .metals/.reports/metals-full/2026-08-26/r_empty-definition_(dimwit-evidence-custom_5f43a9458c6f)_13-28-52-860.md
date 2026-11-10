file://<WORKSPACE>/cases/03-keras-cross-default-axis/Fixed.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:
	 -dimwit/b2.
	 -dimwit/b2#
	 -dimwit/b2().
	 -b2.
	 -b2#
	 -b2().
	 -scala/Predef.b2.
	 -scala/Predef.b2#
	 -scala/Predef.b2().
offset: 922
uri: file://<WORKSPACE>/cases/03-keras-cross-default-axis/Fixed.scala
text:
```scala
//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0

/** Case 03 (fixed) — cross product, DimWit.
  *
  * `cross3` operates on a single `Tensor1[Component, Float32]`. Which axis holds the three
  * components is fixed by the type, not discovered by searching the shape for a `3`.
  */
object Case03Fixed:

  import dimwit.*

  /** Rows of the batch. Also happens to have extent 3 in the failing example. */
  trait Sample derives Label

  /** The x/y/z axis of a 3-vector. */
  trait Component derives Label

  private def at(t: Tensor1[Component, Float32], i: Int): Tensor0[Float32] =
    t.slice(Axis[Component].at(i))

  def cross[L: Label](a: Tensor1[Component, Float32], b: Tensor1[Component, Float32]): Tensor1[Component, Float32] =
    val (a0, a1, a2) = (at(a, 0), at(a, 1), at(a, 2))
    val (b0, b1, b2) = (at(b, 0), at(b, 1), at(b, 2))
    stack(
      Seq(a1 * b2 - a2 * b1, a2 * b0 - a0 * b@@2, a0 * b1 - a1 * b0),
      Axis[Component]
    )

  /** Batched over `Sample`, which is the only axis that could be batched over. */
  def crossBatched(
      a: Tensor2[Sample, Component, Float32],
      b: Tensor2[Sample, Component, Float32]
  ): Tensor2[Sample, Component, Float32] =
    zipvmap(Axis[Sample])(a, b)((x: Tensor1[Component, Float32], y: Tensor1[Component, Float32]) => cross3(x, y))

  @main def case03Check(): Unit =
    dimwit.initialize()

    // the 3x3 input from the upstream repro: both axes have extent 3
    val a = Tensor(Shape(Axis[Sample] -> 3, Axis[Component] -> 3))
      .fromArray(Array(0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f))
    val b = Tensor(Shape(Axis[Sample] -> 3, Axis[Component] -> 3))
      .fromArray(Array(8f, 7f, 6f, 5f, 4f, 3f, 2f, 1f, 0f))

    val out = crossBatched(a, b)
    // row 0: (0,1,2) x (8,7,6) = (1*6-2*7, 2*8-0*6, 0*7-1*8) = (-8, 16, -8)
    val r0 = out.slice(Axis[Sample].at(0)).toArray
    assert(r0.sameElements(Array(-8f, 16f, -8f)), s"got ${r0.mkString(",")}")
    println("case03 ok: contracted over Component, never over Sample")

```


#### Short summary: 

empty definition using pc, found symbol in pc: 