# Plain (NumPy) — verdict

**Category: `missed`**

`reshape` validates exactly one thing: that the element count is preserved. Both `(6, 5)`
and `(10, 3)` hold 30 elements, so NumPy accepts the wrong rectangle
(`test_numpy_raises_nothing_because_the_element_count_matches`).

The result is not merely reshaped, it is scrambled: rows no longer correspond to
(chain, draw) pairs, so every downstream consumer that treats a row as one posterior draw
is reading interleaved values.

`test_it_is_correct_while_the_sample_dims_are_leading` explains the survival: in the normal
case `chain, draw` already lead, the pre- and post-transpose tails coincide, and the code is
correct by accident.
