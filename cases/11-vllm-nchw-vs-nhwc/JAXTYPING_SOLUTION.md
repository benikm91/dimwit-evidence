# jaxtyping — verdict

**Category: `missed` (square frames) / no help even when it "passes"**

The annotation `Float[Array, "frame height width channel"]` reads like the fix, and for the
shipping case — square RGB frames — it is satisfied by the wrong layout
(`test_square_rgb_frames_slip_through`).

Worse, for a non-square frame it *still passes*: `channel` simply binds to 5 and
`per_channel_mean` returns five values. `test_the_annotation_cannot_state_which_axis_is_the_channel`
shows the same function accepting `(2, 3, 4, 5)` and `(2, 5, 4, 3)` with equal enthusiasm.

That is the crucial difference from cases 05 and 08, where jaxtyping earned a
`run-time detection`: there, one axis size was *cross-referenced* between the input and the
output annotation. Here every axis is free, so there is no constraint to violate.
