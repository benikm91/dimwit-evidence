# jaxtyping — verdict

**Category: `missed` (self-attention) / `run-time detection` (cross-attention)**

jaxtyping lets you name axes in an annotation, e.g.
`Float[Array, "batch heads query key"]`, and checks at run time that the names bind
consistently. That is genuinely more than plain JAX offers — but the binding is by
**size**, not by identity.

* Pix2Struct's visual encoder is **self-attention**, so `query == key == seq`. The names
  `query` and `key` bind to the same number, the buggy broadcast yields exactly the
  declared return shape, and the annotation is satisfied. The defect is **missed**.
* Change the same code to **cross-attention** (`query = 4`, `key = 3`) and the buggy
  broadcast produces `[1, 1, 4, 4]` where `[1, 1, 4, 3]` was declared. jaxtyping raises —
  at run time, on the executed path only.

The general lesson for the paper: a checker that distinguishes axes by their extent is
blind whenever two axes happen to share an extent, and in tensor code they share extents
constantly (`batch == seq`, square images, `heads == layers`, the `3` of a cross product).

See `jaxtyping_case.py::test_self_attention_buggy_passes_the_typechecker`.
