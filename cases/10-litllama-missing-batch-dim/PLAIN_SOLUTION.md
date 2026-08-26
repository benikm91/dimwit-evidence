# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

Every operation in the model is rank-polymorphic — fancy indexing, `mean`, `@` — so the
unbatched call runs to completion and returns `(vocab,)`, exactly the shape the caller
expects (`test_the_unbatched_call_runs_and_returns_a_plausible_shape`).

The damage is done by the one operation that assumed the leading axis was the batch. It now
centres across the sequence, and the logits are different
(`test_but_the_numbers_are_different`). Generation continues and emits valid token ids
(`test_sampling_from_the_wrong_logits_still_produces_valid_tokens`), so the failure looks
like a quality problem, not a bug.

`logits[-1]` is the aggravating factor: negative indexing means the expression stays valid
whatever the rank, and reads the last element of whichever axis happens to be first.
