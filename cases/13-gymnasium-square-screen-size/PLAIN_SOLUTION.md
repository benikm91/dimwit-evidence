# Plain (NumPy) — verdict

**Category: `missed`**

`resize_buggy(frame, (84, 64))` returns a `(64, 84)` array. Nothing raises: both extents are
positive integers, both indexing operations are in bounds, and the result is a perfectly
well-formed 2-D frame. It is simply the transpose of the one that was asked for
(`test_the_swap_is_not_caught`).

The mismatch is between two *conventions*, and a tuple carries no convention. `(84, 64)`
means "height then width" on one side of the call and "width then height" on the other, and
the only record of which is which is the parameter name — `dsize` versus `size_hw` — which
NumPy never sees.

`test_a_square_target_hides_the_defect_entirely` is why it reached production: `(s, s)` is
the fixed point of the two orderings, so for as long as `screen_size` was square the two
functions were not merely equivalent but bit-identical. The defect was latent in the code
for as long as nobody exercised the one input that could distinguish them.

`test_the_declared_shape_and_the_actual_array_disagree` states the upstream symptom in one
line: the shape the wrapper advertised was the reverse of the shape it produced, and only a
consumer that cared about orientation would ever notice.
