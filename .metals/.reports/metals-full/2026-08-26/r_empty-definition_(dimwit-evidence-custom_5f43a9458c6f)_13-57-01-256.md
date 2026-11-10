file://<WORKSPACE>/cases/04-elastic-transform-hw-swap/Buggy.scala
empty definition using pc, found symbol in pc: 
semanticdb not found
empty definition using fallback
non-local guesses:
	 -dimwit/normaliseHorizontal.
	 -dimwit/normaliseHorizontal#
	 -dimwit/normaliseHorizontal().
	 -Case04Fixed.normaliseHorizontal.
	 -Case04Fixed.normaliseHorizontal#
	 -Case04Fixed.normaliseHorizontal().
	 -normaliseHorizontal.
	 -normaliseHorizontal#
	 -normaliseHorizontal().
	 -scala/Predef.normaliseHorizontal.
	 -scala/Predef.normaliseHorizontal#
	 -scala/Predef.normaliseHorizontal().
offset: 412
uri: file://<WORKSPACE>/cases/04-elastic-transform-hw-swap/Buggy.scala
text:
```scala
//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 04 (buggy) — elastic displacement normalisation, DimWit. DOES NOT COMPILE.
  *
  * The axes and the two normalisers are the ones from `Fixed.scala`; only the use is wrong.
  */
object Case04Buggy:

  import dimwit.*
  import dimwit.stats.Uniform
  import Case04Fixed.{Direction, Height, Width, normaliseH@@orizontal, normaliseVertical}

  lazy val KEY = Random.Key(7)

  def displacementBuggy(
      alpha: (Float, Float),
      size: Shape[(Height, Width)]
  ): Tensor3[Height, Width, Direction, Float32] =
    val noise = Uniform(Tensor(size).fill(-1.0f), Tensor(size).fill(1.0f))
    val (kx, ky) = KEY.split2()

    // `size[0]` and `size[1]` are positions in Python and cannot be written here at all;
    // the axis has to be named, and naming the wrong one is checked => compile-error
    val dx: Tensor2[Height, Width, Float32] = normaliseHorizontal(noise.sample(kx), alpha._1, size.extent(Axis[Height]))
    val dy: Tensor2[Height, Width, Float32] = normaliseVertical(noise.sample(ky), alpha._2, size.extent(Axis[Width]))

    stack(Seq(dx, dy), Axis[Direction], Axis[Width])

```


#### Short summary: 

empty definition using pc, found symbol in pc: 