//> using scala 3.8.1
//> using dep ch.contrafactus::dimwit-core:0.1.0
//> using file Fixed.scala

/** Case 09 (buggy) — DimWit. DOES NOT COMPILE, on purpose.
  *
  * The routine is the one from `Fixed.scala`; only the call is wrong. `ft_ao(mol, g)` was
  * called with a single `(3,)` vector where the body assumed `(N, 3)`, and NumPy obliged by
  * reducing the only axis and broadcasting the scalar that came out.
  */
object Case09Buggy:

  import dimwit.*
  import Case09Fixed.{Center, Component, GPoint, ftAoFixed}

  def ftAoBuggy(
      g: Tensor1[Component, Float32],
      centers: Tensor1[Center, Float32]
  ): Tensor2[GPoint, Center, Float32] =
    // one vector where a grid is declared; there is no missing axis to invent => compile-error
    ftAoFixed(g, centers)
