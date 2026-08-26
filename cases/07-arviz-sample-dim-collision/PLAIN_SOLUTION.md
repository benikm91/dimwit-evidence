# Plain (NumPy / xarray-style) — verdict

**Category: `missed`**

`NamedArray` in `plain.py` is the model every mainstream named-axis library uses: an array
plus a tuple of **runtime strings**. Nothing prevents the same string appearing twice.

* `test_stacking_produces_two_dimensions_with_the_same_name` — after stacking, `dims` is
  `("sample", "sample")`. No exception.
* `test_selection_by_name_now_picks_the_wrong_axis` — `mean_over("sample")` resolves to the
  first match and collapses the MCMC draws instead of the user's samples. The result still
  has a sensible shape and is still called `sample`.

The upstream fix renamed the internal dimension to `__sample__`.
`test_the_fix_is_a_convention_not_a_guarantee` shows what that buys: a user dimension
actually called `__sample__` reproduces the bug exactly. Reserving a string is a social
convention, enforced by nothing.
