# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

`softmax(x, axis=0)` and `softmax(x, axis=-1)` have the same signature, the same output
shape and the same dtype. The buggy output is still in `[0, 1]` and still sums to 1 —
along the other axis — so no downstream sanity check that inspects value ranges will fire.

`test_a_single_row_batch_makes_every_class_equally_certain` is worth including in the paper:
with a batch of one, normalising over the batch turns every logit into 1.0, so the model
reports total confidence in every class and still raises nothing.
