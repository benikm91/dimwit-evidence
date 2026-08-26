# Plain JAX — verdict

**Category: `missed`**

Strictly worse than the NumPy/xarray case, and worth saying so plainly: JAX arrays carry no
axis names at all, so the names live in a separate Python object that the caller maintains
by hand.

`test_bookkeeping_can_drift_from_the_array_without_any_check` makes the point: a `dims`
tuple with the wrong number of entries sits happily beside its array. There is no
collision *detection* because there is no collision *representation* — the two axes were
never named in the first place.
