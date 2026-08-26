# DimWit — verdict

**Category: `missed`**

`scala-cli compile Buggy.scala` succeeds. The defect is a scalar divisor, and
`size(Axis[Height])` hands back a plain `Int`, so by the time it reaches the arithmetic
there is nothing left for the type system to check.

What does change is the failure mode. `size[0]` cannot be written at all: DimWit has no
positional shape access, so the author has to name the axis, and the buggy line says
`alpha._1 / size(Axis[Height])` — dividing the *horizontal* displacement by the *height*,
in as many words. Upstream this was `size[0]` versus `size[1]`, an index slip that survived
review of both `transforms` and `transforms.v2`. That is a legibility improvement rather
than a detection, and the paper should label it as one.

DimWit can in principle do better here: `size.extent(Axis[Width])` returns an
`AxisExtent[Width]`, a different type from `AxisExtent[Height]`, and an API that takes the
wrapped extent would reject the swap at compile time. It is not used above because the
straight transliteration of `dx * alpha[0] / size[0]` divides by a number.
