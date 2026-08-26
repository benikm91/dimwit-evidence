# Plain (NumPy / PyTorch) — verdict

**Category: `missed`**

Both conventions produce `(3, 3)` from `(3, 3)` inputs. `np.cross` validates one thing —
that the chosen axis has extent 3 — and both axes satisfy it. There is nothing left to
check.

`test_the_bug_is_invisible_until_two_axes_collide` is the part worth putting in the paper:
on a `(2, 4, 3)` input the two conventions agree exactly, because only one axis has extent
3. The defect is latent in every call and observable only when a second axis happens to
have the same length. That is a fault that testing finds by luck.
