# 03 — `ops.cross` picked the first length-3 axis instead of the last

**Source:** [keras-team/keras#23219](https://github.com/keras-team/keras/pull/23219)
— *"Fix torch backend cross() using wrong default axis"*

## The defect

The cross product is an operation on two 3-vectors, but every array library exposes it as
an operation on arbitrary high-dimensional arrays, with the vector axis supplied as extra
arguments — `numpy.cross(a, b, axisa, axisb, axisc, axis)`, `torch.cross(a, b, dim)`. Those
arguments have defaults, and the defaults do not agree: NumPy, JAX and TensorFlow use the
**last** axis; `torch.cross` uses **the first axis of size 3**.

Keras' torch backend forwarded `axis=None` straight through, so it inherited torch's
default and disagreed with its own other backends:

```python
a = np.arange(9.).reshape(3, 3)
b = np.arange(9.)[::-1].reshape(3, 3)
keras.ops.cross(a, b)   # torch: axis 0.  numpy: axis 2.  Same shape, different numbers.
```

The bug is therefore not a slip inside Keras. It is a wrapper inheriting an under-specified
part of a dependency's interface: an axis that the caller never named and that each library
guesses differently. The fix defaults `axis` to `-1` before the call, and adds a `(2, 3, 3)`
regression test.

## Why it is interesting

This is a case about **scope**: minimal scopes clarify concepts, and this is what a
non-minimal one costs. The concept "cross product" is a function of two 3-vectors. Every
library ships it at a much broader scope — arbitrary rank, an axis argument per operand,
and a default for each — which buries that core logic under a convention. The convention is
then not canonical: two libraries chose differently, and a wrapper inherited the
disagreement.

The permissive scope is what makes the defect possible; the wrong default is only which
way it fell. **An API that identifies an axis by its extent is ambiguous exactly when two
axes share an extent**, and `3` is the most collision-prone extent in numerical code — RGB,
spatial coordinates, quaternion-adjacent layouts, and the cross product itself. On the
`(3, 3)` input, "the first axis of size 3" and "the last axis of size 3" are both valid
readings of the same array, so no amount of defaulting repairs the interface: as long as
the vector axis is a parameter, some caller omits it.

Every checker in this comparison that reasons about sizes is structurally unable to help,
because the sizes are identical in the buggy and the correct program.
