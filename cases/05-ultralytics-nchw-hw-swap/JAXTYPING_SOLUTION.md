# jaxtyping — verdict

**Category: `missed` as the code would naturally be written; `run-time detection` only if the
layout is carried by an array and the model input is rectangular**

This verdict was corrected by building the case. The first draft assumed jaxtyping would
catch the swap. It does not, for a reason worth reporting:

> **jaxtyping binds axis names only from array annotations.** A parameter declared
> `height: int` does not bind the symbol `height`, so a return annotation mentioning it
> constrains nothing.

`test_int_arguments_bind_nothing_so_the_swap_is_missed` shows
`to_model_input_int_args(image, 640, 480)` returning `(1, 3, 480, 640)` — transposed — and
being accepted, because `height` and `width` in the return annotation are free variables.

To get a check you have to hand the function an array that carries the layout:

* `test_a_template_array_does_bind_them_and_the_swap_is_caught` — with a
  `UInt8[Array, "channel height width"]` template the symbols are bound and the swap raises.
* `test_even_with_a_template_a_square_model_hides_it` — and it is still missed for the
  default 640x640 export.

So the pattern is narrower than it first appears: a size-based checker catches an axis swap
only when the sizes differ **and** both sizes are carried by arrays rather than by scalars.

Compare DimWit, where `AxisExtent[Height]` is a distinct type from `AxisExtent[Width]`
precisely so that a *scalar* extent keeps its axis identity — which is the gap this case
exposes.
