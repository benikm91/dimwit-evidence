file://<WORKSPACE>/cases/04-elastic-transform-hw-swap/Buggy.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:
	 -dimwit/Tensor0.
	 -dimwit/Tensor0#
	 -dimwit/Tensor0().
	 -Tensor0.
	 -Tensor0#
	 -Tensor0().
	 -scala/Predef.Tensor0.
	 -scala/Predef.Tensor0#
	 -scala/Predef.Tensor0().
offset: 952
uri: file://<WORKSPACE>/cases/04-elastic-transform-hw-swap/Buggy.scala
text:
```scala
//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 04 (buggy) — elastic displacement normalisation, DimWit. THIS ONE COMPILES.
  *
  * `size[0]` cannot be written, so the author has to name the axis — and then divides the
  * horizontal displacement by `Axis[Height]` in as many words. Wrong in a way a reader can
  * see, but `size(Axis[L])` hands back a plain `Int`, so nothing checks it.
  */
object Case04Buggy:

  import dimwit.*
  import dimwit.stats.Uniform
  import Case04Fixed.{Direction, Height, KEY, Width}

  def displacementBuggy(
      alpha: (Float, Float),
      size: Shape[(Height, Width)]
  ): Tensor3[Height, Width, Direction, Float32] =
    val noise = Uniform(Tensor(size).fill(-1.0f), Tensor(size).fill(1.0f))
    val (kx, ky) = KEY.split2()

    // as shipped: horizontal displacement normalised by the height, vertical by the width
    val dx = noise.sample(kx) *! Tenso@@r0(alpha._1 / size(Axis[Height]).toFloat)
    val dy = noise.sample(ky) *! Tensor0(alpha._2 / size(Axis[Width]).toFloat)

    stack(Seq(dx, dy), Axis[Direction], afterAxis = Axis[Width])

```


#### Short summary: 

empty definition using pc, found symbol in pc: 