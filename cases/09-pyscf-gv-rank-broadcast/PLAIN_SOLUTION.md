# Plain (NumPy) — verdict

**Category: `missed`**

NumPy's broadcasting rule prepends missing leading axes, so a `(3,)` argument flows through
a routine written for `(N, 3)` and produces an answer one rank short. Every individual
operation is legal (`test_numpy_never_objects`).

The upstream regression test is reproduced verbatim in
`test_a_single_vector_silently_loses_a_rank`: `ft_ao(g)` returns `(4,)` where
`ft_ao(g.reshape(1, 3))` returns `(1, 4)`.

`test_the_rank_loss_propagates_as_a_wrong_sum_not_an_exception` shows why this is worse than
a crash: a caller that sums over the G-grid gets a scalar from one branch and a length-4
array from the other, both of which look like plausible physics.
