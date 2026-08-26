"""Case 07 under jaxtyping — `plain_jax.py` with as much typing as jaxtyping can express.

The dimension names live in a tuple of Python strings, which jaxtyping has no type for, so
the collision itself is invisible to it. Annotating the arrays instead does not help either:
jaxtyping's axis names bind to *sizes*, so a stacked chain*draw axis of 12 and a user
`sample` axis of 12 are interchangeable.
"""

import math

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype


class NamedArray:
    """An array whose axes are identified by runtime strings."""

    def __init__(self, data: Float[Array, "..."], dims: tuple[str, ...]):
        assert len(dims) == data.ndim, "one name per axis"
        self.data = jnp.asarray(data)
        self.dims = tuple(dims)

    def axis_of(self, name: str) -> int:
        """Selection by name. With duplicate names this silently returns the first."""
        return self.dims.index(name)

    def mean_over(self, name: str) -> "NamedArray":
        return NamedArray(
            jnp.mean(self.data, axis=self.axis_of(name)),
            tuple(d for i, d in enumerate(self.dims) if i != self.axis_of(name)),
        )

    def stack(self, new_name: str, to_stack: tuple[str, ...]) -> "NamedArray":
        """Flatten `to_stack` into a single leading axis called `new_name`."""
        rest = [d for d in self.dims if d not in to_stack]
        order = list(to_stack) + rest
        moved = jnp.transpose(self.data, [self.dims.index(d) for d in order])
        n = math.prod(self.data.shape[self.dims.index(d)] for d in to_stack)
        return NamedArray(jnp.reshape(moved, (n, *moved.shape[len(to_stack):])), (new_name, *rest))


def mean_over_user_samples(stacked: NamedArray) -> NamedArray:
    """A downstream routine written against the USER's `sample` dimension."""
    return stacked.mean_over("sample")


def stack_buggy(posterior: NamedArray) -> NamedArray:
    """As shipped: the stacked dimension is called `sample`."""
    return posterior.stack("sample", ("chain", "draw"))


def stack_fixed(posterior: NamedArray) -> NamedArray:
    """After PR #1647: a name chosen to be unlikely rather than a name that cannot clash."""
    return posterior.stack("__sample__", ("chain", "draw"))


def _posterior_with_a_user_sample_dim() -> NamedArray:
    return NamedArray(
        jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5), ("chain", "draw", "sample")
    )


# --------------------------------------------------------------------------- tests

def test_the_dimension_names_are_strings_that_jaxtyping_cannot_reach():
    """MISSED. `tuple[str, ...]` is as much as can be said; the duplicate is well typed."""
    assert stack_buggy(_posterior_with_a_user_sample_dim()).dims == ("sample", "sample")


def test_selection_by_name_now_picks_the_wrong_axis():
    stacked = stack_buggy(_posterior_with_a_user_sample_dim())
    assert mean_over_user_samples(stacked).data.shape == (5,)


def test_fixed_keeps_the_two_dimensions_distinct():
    stacked = stack_fixed(_posterior_with_a_user_sample_dim())
    assert stacked.dims == ("__sample__", "sample")
    assert mean_over_user_samples(stacked).dims == ("__sample__",)


# ----------------------------------------- not the ladder: the names inside an annotation
#
# The nearest jaxtyping equivalent of an axis name is a symbol in an array annotation. It
# binds to a *size*, so it cannot tell the stacked axis from the user's.

@jaxtyped(typechecker=beartype)
def summarise_user_samples(x: Float[Array, "sample param"]) -> Float[Array, "param"]:
    """Meant to consume the user's `sample` dimension."""
    return jnp.mean(x, axis=0)


def test_a_stacked_chain_draw_axis_is_accepted_as_a_sample_axis():
    """MISSED. 4 chains x 3 draws = 12; a user `sample` axis of 12 is indistinguishable."""
    assert summarise_user_samples(jnp.zeros((12, 5))).shape == (5,)


def test_only_a_rank_error_is_caught():
    with pytest.raises(Exception):
        summarise_user_samples(jnp.zeros((4, 3, 5)))
