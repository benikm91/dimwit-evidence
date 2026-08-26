"""Case 02 under jaxtyping.

Verdict hinges on whether the loss is annotated at all, and with which names.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


@jaxtyped(typechecker=beartype)
def mse_strict(
    pred: Float[Array, "batch"],
    target: Float[Array, "batch"],
) -> Float[Array, ""]:
    """Annotated so both operands must be rank-1 over the same axis."""
    return jnp.mean((pred - target) ** 2)


@jaxtyped(typechecker=beartype)
def mse_as_written_upstream(
    pred: Float[Array, "batch out"],
    target: Float[Array, "batch"],
) -> Float[Array, ""]:
    """The signature the buggy model actually had: a [batch, out] prediction."""
    return jnp.mean((pred - target) ** 2)


# --------------------------------------------------------------------------- tests

def test_strict_annotation_rejects_the_column_vector():
    """Run-time detection: passing [100, 1] where "batch" was declared raises."""
    with pytest.raises(Exception):
        mse_strict(jnp.zeros((100, 1)), jnp.zeros(100))


def test_strict_annotation_accepts_the_fixed_call():
    assert float(mse_strict(jnp.zeros(100), jnp.zeros(100))) == 0.0


def test_the_upstream_signature_is_accepted_and_still_wrong():
    """MISSED: the annotation describes the shapes truthfully, and the body is still wrong.

    jaxtyping constrains the boundary, not the arithmetic inside it. `pred - target`
    broadcasts to [batch, batch] and the scalar return annotation is satisfied.
    """
    out = mse_as_written_upstream(jnp.linspace(0, 1, 100).reshape(100, 1), jnp.linspace(0, 1, 100))
    assert float(out) > 0.0  # should have been 0.0


def test_out_axis_of_size_one_is_indistinguishable_from_no_axis():
    """With out == 1 there is no size evidence that anything is wrong."""
    assert mse_as_written_upstream(jnp.zeros((5, 1)), jnp.zeros(5)).shape == ()
