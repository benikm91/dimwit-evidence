# jaxtyping — verdict

**Category: `run-time detection`**

A clean win, and the paper should say so. `Float[Array, "gpoint 3"]` fixes the rank at 2,
and `ft_ao(GRID[1])` raises immediately.

The general observation, which is worth generalising across the dossier: **rank is the one
property a size-based annotation pins down reliably**, because two ranks cannot coincide by
accident the way two extents can. jaxtyping therefore does well on cases 09, 10, 13 and 15
(all rank errors) and poorly on 01, 03, 06 and 07 (all identity errors).

The remaining gap is *when*: the check runs on the executed path, so it finds the defect
the first time that path is taken, not before the program starts.
