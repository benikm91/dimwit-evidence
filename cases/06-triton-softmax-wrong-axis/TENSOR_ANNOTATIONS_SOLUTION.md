# tensor_annotations — verdict

**Category: `missed` — the reductions are not typed the way softmax uses them: `max` is
not typed at all, and `sum` only for a literal axis.**

Two independent reasons, each sufficient on its own:

1. **`max` is not a typed operation.** Softmax starts with `x - max(x, dim)`, and the stubs
   do not type `max`: `Array2` has no `max` method at all, and `jnp.max` is declared
   `max: Any`. Its result is `Any`, `x - Any` is `Any`, and everything downstream is
   unchecked — including the broadcast that the stubs *would* check by name.
2. **`sum` is typed only for a literal axis.** Its overloads match `axis: Literal[0]`,
   `Literal[1]`, … and compute the result's axes from that literal (`jnp.sum(x, axis=1)` on
   `[Row, Col]` is `Array1[Row]`). Upstream passes `dim` at run time; an `int` matches only
   the untyped fallback overload, so `jnp.sum(x, axis=dim)` is `Any`
   (`test_mypy`/`test_pyright` control). The same holds for `keepdims=True`, which the stubs
   leave untyped ("yet be to be typed"). So even if `max` were typed like `sum`, the
   upstream code would still be missed.

The variants below show what is left once both are worked around.

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright (they agree on every line).

The mechanism is the one tensor_annotations is built to see: `keep_dims=False` drops the
reduced axis, and the reduced vector re-attaches to the *trailing* axis. The stubs
broadcast trailing axes **by name**, so `[Row, Col] / [Row]` is a type error. Whether the
checker gets that far depends on how the softmax is written.

| variant in `tensor_annotations_case.py` | result |
|---|---|
| **A** as written upstream: `x.max(...)`, `num.sum(...)` | buggy *and* fixed rejected — `Array2` has no `max`/`sum` methods in the stubs. A stub gap, not a detection. |
| **B** `jnp.max/jnp.sum(x, axis=dim, keepdims=keep_dims)`, `dim: int` | **missed** — a run-time `int` axis or `bool` flag matches only the untyped overloads (`Any`) |
| **C** literal `axis=1`, default `keepdims=False`, `num: Tile` annotated | **rejected**, also on the square 2x2 tile from the issue: `Unsupported operand types for / ("Array2[float32, Row, Col]" and "Array1[float32, Row]")` |
| C, fixed with `keepdims=True` | accepted — but only because it is untyped: the stubs say "keepdims = True: yet be to be typed" and return `Any` |
| C, `axis=0` (accidentally correct upstream) | accepted, and typed: `[Col]` is the trailing axis |
| **D** minimal scope, `softmax_vector(v: Array1[N]) -> Array1[N]` | a tile handed to it is rejected statically; the lift `jax.vmap(softmax_vector)` is `Any` (`vmap: Any`) |

Two things keep **C** from being the verdict for the upstream code:

* **The axis must be a literal.** `tl.softmax(x, dim)` takes `dim` at run time. An `int`
  carries no axis identity, so the reduction's result type cannot be computed — the same
  limit as case 03's axis argument.
* **The annotation must be added by hand.** `max: Any` in the stubs, and `x - Any` is `Any`,
  so the first half of softmax (`x - max(x)`) cannot be checked at all, and everything after
  it is `Any` too. Only `num: Tile = jnp.exp(...)` restores a type for the second half, where
  the `sum`-then-divide is caught.

And the fix it accepts is accepted for the wrong reason — as `Any`, not because
`keepdims=True` is understood to keep the axis.

**Compared with the other columns.** C catches what jaxtyping misses at the upstream scope
(the square tile), and does it statically. D matches jaxtyping's minimal-scope result
statically rather than at run time — up to the lift, which jaxtyping's `jax.vmap` test runs
and checks, and tensor_annotations cannot type. DimWit gets the same result without either
condition: the reduction is by `Axis[Col]`, not by an integer, and `keepDims` is not
expressible (`DIMWIT_SOLUTION.md`).
