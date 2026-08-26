# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

A square RGB frame batch is `[2, 3, 3, 3]` under *both* layouts
(`test_the_two_layouts_are_indistinguishable_for_square_rgb_frames`), so the pipeline
reduces over frames, channels and rows instead of frames, rows and columns, and returns
three plausible numbers. No exception (`test_no_exception_is_raised_anywhere`).

`test_a_non_square_frame_would_have_crashed_instead` is the useful contrast: with a 4x5
frame the same code returns five "channel" means, which is obviously wrong to a human and
still not an error to NumPy.

The layout is a contract recorded in documentation. Nothing in the array carries it.
