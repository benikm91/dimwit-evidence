# jaxtyping — verdict

**Category: `missed`**

The annotation `Float[Array, "n three"]` looks like it pins the component axis down. It
does not. jaxtyping binds each name to an **extent**; on a `(3, 3)` input `n` binds to 3
and `three` binds to 3, the two are interchangeable, and both the buggy and the fixed
function satisfy the signature.

This case is the cleanest statement of the structural gap, and is worth citing in the
paper as such:

> A name that denotes a size cannot distinguish two axes of the same size.
> A name that denotes an identity can.

`test_naming_the_axis_three_does_not_pin_it_down` demonstrates it directly.
