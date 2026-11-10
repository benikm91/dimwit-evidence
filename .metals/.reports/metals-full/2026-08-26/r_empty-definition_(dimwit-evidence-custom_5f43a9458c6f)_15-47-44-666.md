file://<WORKSPACE>/cases/14-gymnasium-square-screen-size/Buggy.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:
	 -dimwit/preprocessFixed.
	 -dimwit/preprocessFixed#
	 -dimwit/preprocessFixed().
	 -Case14Fixed.preprocessFixed.
	 -Case14Fixed.preprocessFixed#
	 -Case14Fixed.preprocessFixed().
	 -preprocessFixed.
	 -preprocessFixed#
	 -preprocessFixed().
	 -scala/Predef.preprocessFixed.
	 -scala/Predef.preprocessFixed#
	 -scala/Predef.preprocessFixed().
offset: 939
uri: file://<WORKSPACE>/cases/14-gymnasium-square-screen-size/Buggy.scala
text:
```scala
//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 14 (buggy) — Atari preprocessing, DimWit. THIS FILE COMPILES.
  *
  * That is the finding, not a harness failure. The upstream defect is one integer standing
  * for two different concepts, and DimWit cannot object: `Axis[Height] -> screenSize` and
  * `Axis[Width] -> screenSize` are well typed for any `screenSize`, and have to be, since
  * square tensors are legitimate. What is left is that the reuse of the one number sits
  * between two named axes rather than inside an anonymous `(size, size)` tuple.
  */
object Case14Buggy:

  import dimwit.*
  import Case14Fixed.{Height, Width, preprocessFixed}

  /** `screen_size: int` — one number, silently used for both spatial axes. */
  def preprocessBuggy(
      frame: Tensor2[Height, Width, Float32],
      screenSize: Int
  ): Tensor2[Height, Width, Float32] =
    preproces@@sFixed(frame, Axis[Height] -> screenSize, Axis[Width] -> screenSize)

  @main def case14BuggyDemo(): Unit =
    dimwit.initialize()

    val frame = Tensor(Shape(Axis[Height] -> 210, Axis[Width] -> 160))
      .fromArray(Array.tabulate(210 * 160)(_.toFloat))

    val out = preprocessBuggy(frame, 84)
    assert(out.shape(Axis[Height]) == out.shape(Axis[Width]), "expected a forced square")
    println(s"case14 buggy: compiled and ran; 210x160 became ${out.shape(Axis[Height])}x${out.shape(Axis[Width])}")

```


#### Short summary: 

empty definition using pc, found symbol in pc: 