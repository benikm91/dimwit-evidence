# jaxtyping — verdict

**Category: `missed`**

jaxtyping annotations *look* like the fix for this case — they contain the word `sample` —
but the name binds to a size. `summarise_user_samples(x: Float[Array, "sample param"])`
accepts a `(12, 5)` array whether the 12 came from the user's samples or from 4 chains x
3 draws.

`test_it_is_accepted_even_when_the_sizes_were_never_meant_to_match` shows the general
shape of the limitation: the annotation constrains rank and size agreement, never
provenance. Only a rank error is caught.
