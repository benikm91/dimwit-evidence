# tensor_annotations — verdict

**Category: `missed` (code as written, and every variant built from shape entries) /
static rejection only for a typed rewrite**

**In short: `.shape[i]` is an `int`, not a named axis extent.** `template.shape` is typed
`Tuple[int, int, int]`, so `shape[1]` (the height) and `shape[2]` (the width) have the same
type, and reading them in the wrong order is well typed. The buffer built from those ints,
`jnp.zeros((h, w, c))`, then gets no axes either (`Array3[Any, Any, Any]`). The axis identity
is lost at the moment an extent is read out of a shape.

DimWit catches it because that is the one step where it keeps the identity:
`shape.extent(Axis[Width])` returns an `AxisExtent[Width]`, not an `Int`, and a buffer built
from extents carries their axes, so the swap produces `Tensor3[Width, Height, Channel]` and
is rejected. tensor_annotations has no type for a single extent, so it cannot.

*Not run by `run_all.sh`; run it with `./scripts/check_tensor_annotations.sh -v`.*

[tensor_annotations](https://github.com/google-deepmind/tensor_annotations) v2.0.3 with
`tensor_annotations_jax_stubs`, checked with mypy and pyright.

The upstream mistake reads two extents out of an NCHW shape in the wrong order and builds a
buffer from them. Every variant that does that is missed, because a buffer built from
numbers gets no axes:

* **As written** (`onnx_input_shape: tuple[int, int, int, int]`). `jnp.zeros((h, w, c))` is
  typed by rank only, `Array3[float32, Any, Any, Any]`, and `[None, ...]` (adding the batch
  axis) is `Any`, so the `ModelInput` return annotation is never compared to anything.
* **Buffer annotated** `Letterbox = Array3[Height, Width, Channels]`: accepted, `Any` axes
  fit every annotation.
* **Layout carried by a typed template array** `Array3[Channels, Height, Width]`, extents
  read from `template.shape`: accepted, `.shape` is `Tuple[int, int, int]`. This is the
  variant that made jaxtyping fire (for rectangular inputs); here it does not help, because
  tensor_annotations checks axis identity but never sees the extents at all.

**The typed rewrite is rejected — by both checkers, also for square inputs.** If the buffer
is not built from numbers but from the template with typed ops,
`jnp.transpose(jnp.zeros_like(template), perm)`, the swap becomes a wrong permutation and
the result `[Width, Height, Channels]` does not match `Letterbox`:

```
pyright: Type "Array3[float32, Width, Height, Channels]" is not assignable to return type "Letterbox"
mypy:    Argument 1 to "zeros_like" has incompatible type "Array3[float32, Channels, Height, Width]";
           expected "Array3[float32, Height, Width, Channels]"
         Argument 2 to "transpose" has incompatible type "tuple[Literal[2], Literal[1], Literal[0]]";
           expected "tuple[Literal[0], Literal[1], Literal[2]]"
```

Static and by identity, so — unlike jaxtyping's template variant — independent of whether
the model is 640x480 or the default 640x640. Two caveats:

* It is a re-expressed program, as in cases 01 and 02. The shipped mistake is in reading
  extents; the rewrite moves it into a permutation, where the types can see it. DimWit's
  `Buggy.scala` keeps the extents and still catches it, because `Shape.extent(Axis[Width])`
  returns an extent that remembers its axis — the thing tensor_annotations has no type for.
* The rewrite only reaches `Letterbox`. Adding the batch axis (`[None, ...]`) is `Any` in
  the stubs (case 01), so the full `ModelInput` cannot be produced typed.

mypy reports the rejection against the *arguments* (it infers the nested calls from the
expected return type); pyright against the return. The line is the same.
