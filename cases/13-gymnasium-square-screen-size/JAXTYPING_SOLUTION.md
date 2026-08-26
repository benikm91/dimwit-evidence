# jaxtyping — verdict

**Category: `missed`**

The axis names are present in every signature and none of them bind to anything relevant.
jaxtyping binds a variable to an *array's* axis; the extents here arrive as Python `int`s
inside a tuple, and jaxtyping has no name for an `int`
(`test_names_do_not_bind_to_integers` calls `inner_resize` with both orderings and both
type-check).

So `Float[Array, "out_h out_w"]` records the result instead of constraining it: `out_h` and
`out_w` are fresh variables bound by whatever comes back, and a `(64, 84)` array satisfies
them exactly as well as an `(84, 64)` one.
`test_the_two_functions_have_identical_signatures` is the compact statement of the problem —
`resize_buggy` and `resize_fixed` are annotation-for-annotation indistinguishable, and one
of them is the bug.

Annotating harder does not rescue it. The only annotation that would flag `resize_buggy` is
`Float[Array, "h w"] -> Float[Array, "w h"]` on `inner_resize`, and that is false: the two
axes are genuinely independent, and every square target satisfies both readings at once.

This is the same limit as case 05 — jaxtyping cannot bind names from scalar `int` parameters
— reached from the other direction. In case 05 the unnamed integers described the *input*
layout; here they describe the *output* extents. Either way, the moment a shape is carried
by numbers rather than by an array, the annotations stop applying.
