# Plain (NumPy) — verdict

**Category: `missed` when the axes coincide, `run-time detection` otherwise**

`np.tensordot(a, b, axes=([1], [0]))` and `axes=([2], [0])` are both legal on a `(3, 3, 3)`
input and both return `(3, 3, 4)` (`test_both_contractions_are_legal_when_the_axes_coincide`).
They contract different axes and give different answers.

With `a` of shape `(2, 5, 3)` the buggy contraction raises — which is how such bugs are
normally found, and why they survive precisely in the square case.

`test_einsum_subscripts_are_positional_too` extends the point to `einsum`: `"bij,jk->bik"`
is a string, and `"bji,jk->bik"` is one keystroke away, equally well-formed, and silently
different. Naming axes with single letters inside a string literal is not the same as naming
them in a type.
