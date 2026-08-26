# jaxtyping — verdict

**Category: `run-time detection`, conditional on the two rectangles differing**

This is a genuine jaxtyping win and should be reported as one. Annotating the return as
`Float[Array, "chain_draw team"]` binds `team` from the input, and the buggy body produces
`chain` columns instead — so `to_point_list_buggy` raises
(`test_the_buggy_output_violates_the_annotation`).

Two caveats for the paper:

* It fires at run time, on the executed path. PyMC's own test suite did not exercise a
  non-leading layout, so a run-time check would not have fired either.
* `test_it_is_missed_when_the_two_rectangles_coincide` — with `team == chain` the two
  outputs have the same shape and the annotation is satisfied by both.
