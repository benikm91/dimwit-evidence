"""Case 02 — MSE between [n, 1] and [n] silently broadcasts to [n, n] (NumPy + PyTorch).

Original: pytorch/pytorch#16045.
"""

import numpy as np

RNG = np.random.default_rng(0)


def _dataset(n=200):
    """y = 10 + 9*x0 - 2*x1 + noise, the reporter's example."""
    x = RNG.normal(size=(n, 2))
    y = 10 + 9 * x[:, 0] - 2 * x[:, 1] + 0.01 * RNG.normal(size=n)
    return x, y


def mse_as_written_upstream(pred, target):
    """pred [n, 1], target [n] -> residuals [n, n]. No error, no warning."""
    return float(((pred - target) ** 2).mean())


def mse_strict(pred, target):
    """Both [n]. The only broadcast is the intended one: none."""
    assert pred.shape == target.shape, f"shape mismatch {pred.shape} vs {target.shape}"
    return float(((pred - target) ** 2).mean())


def fit(x, y, column_vector_predictions, steps=4000, lr=0.05):
    """Plain gradient descent. `column_vector_predictions` selects buggy vs fixed."""
    w = np.zeros(2)
    b = 0.0
    n = len(y)
    for _ in range(steps):
        pred = x @ w + b
        if column_vector_predictions:
            resid = pred[:, None] - y          # [n, n]  <- the defect
            gw = 2 * (resid.mean(axis=1) @ x) / n
            gb = 2 * resid.mean()
        else:
            resid = pred - y                    # [n]
            gw = 2 * (resid @ x) / n
            gb = 2 * resid.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b


# --------------------------------------------------------------------------- tests

def test_buggy_loss_has_no_shape_complaint():
    """The whole problem: numpy is perfectly happy."""
    pred = np.zeros((100, 1))
    target = np.zeros(100)
    assert (pred - target).shape == (100, 100)
    assert mse_as_written_upstream(pred, target) == 0.0


def test_buggy_and_fixed_losses_differ():
    pred_col = np.linspace(0, 1, 100)[:, None]
    target = np.linspace(0, 1, 100)
    assert mse_as_written_upstream(pred_col, target) > 0.0          # residual matrix is not zero
    assert mse_strict(pred_col[:, 0], target) == 0.0   # the predictions are exact


def test_buggy_training_finds_the_wrong_weights():
    """Erroneous behaviour: converges, but not to the true parameters."""
    x, y = _dataset()
    w, b = fit(x, y, column_vector_predictions=True)
    assert not np.allclose(w, [9.0, -2.0], atol=0.5), f"expected wrong weights, got {w}"


def test_fixed_training_recovers_the_true_weights():
    x, y = _dataset()
    w, b = fit(x, y, column_vector_predictions=False)
    assert np.allclose(w, [9.0, -2.0], atol=0.05), f"got {w}"
    assert abs(b - 10.0) < 0.05


def test_torch_only_warns():
    """PyTorch's actual resolution of #16045: a UserWarning, not an exception."""
    torch = __import__("pytest").importorskip("torch")
    import warnings
    pred = torch.zeros(100, 1)
    target = torch.zeros(100)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss = torch.nn.MSELoss()(pred, target)
    assert loss.item() == 0.0
    assert any("target size" in str(w.message) for w in caught), "expected the broadcast warning"
