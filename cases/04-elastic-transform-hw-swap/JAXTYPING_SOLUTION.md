# jaxtyping — verdict

**Category: `missed`**

jaxtyping names axes only inside an *array* annotation. Its whole vocabulary — `Float`,
`Int`, `Shaped`, … — annotates arrays; there is no type for a shape tuple, so the two
numbers the defect lives in can be described no more precisely than `tuple[int, int]`, under
which `(height, width)` and `(width, height)` are the same type
(`test_the_size_tuple_carries_no_axis_identity`).

`displacement` takes no array argument at all, so the return annotation is the only thing
jaxtyping gets to check — and it is satisfied, because the returned array really does have
the shape it claims. `jaxtyping_case.py` is `plain_jax.py` plus that one annotation, which
is the honest measurement here and not an under-annotated file.
