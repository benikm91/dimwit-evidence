"""Case 07 in JAX — the same program as `plain.py`, on jnp arrays.

JAX arrays have no axis names at all, so the collision cannot even be represented by the
array: the names live in bookkeeping the caller maintains alongside it. That is a stronger
form of the same problem, not a weaker one.
"""

import math

import jax.numpy as jnp
import pytest


class NamedArray:
    """An array whose axes are identified by runtime strings."""

    def __init__(self, data, dims):
        assert len(dims) == data.ndim, "one name per axis"
        self.data = jnp.asarray(data)
        self.dims = tuple(dims)

    def axis_of(self, name):
        """Selection by name. With duplicate names this silently returns the first."""
        return self.dims.index(name)

    def mean_over(self, name):
        return NamedArray(
            jnp.mean(self.data, axis=self.axis_of(name)),
            tuple(d for i, d in enumerate(self.dims) if i != self.axis_of(name)),
        )

    def stack(self, new_name, to_stack):
        """Flatten `to_stack` into a single leading axis called `new_name`."""
        rest = [d for d in self.dims if d not in to_stack]
        order = list(to_stack) + rest
        moved = jnp.transpose(self.data, [self.dims.index(d) for d in order])
        n = math.prod(self.data.shape[self.dims.index(d)] for d in to_stack)
        return NamedArray(jnp.reshape(moved, (n, *moved.shape[len(to_stack):])), (new_name, *rest))


def mean_over_user_samples(stacked):
    """A downstream routine written against the USER's `sample` dimension."""
    return stacked.mean_over("sample")


def stack_buggy(posterior):
    """As shipped: the stacked dimension is called `sample`."""
    return posterior.stack("sample", ("chain", "draw"))


def stack_fixed(posterior):
    """After PR #1647: a name chosen to be unlikely rather than a name that cannot clash."""
    return posterior.stack("__sample__", ("chain", "draw"))


def _posterior_with_a_user_sample_dim():
    """4 chains x 3 draws of a quantity indexed by the user's own `sample` dimension."""
    return NamedArray(
        jnp.arange(4 * 3 * 5, dtype=jnp.float32).reshape(4, 3, 5), ("chain", "draw", "sample")
    )


# --------------------------------------------------------------------------- tests

def test_stacking_produces_two_dimensions_with_the_same_name():
    """Erroneous behaviour: the container accepts duplicate names without complaint."""
    assert stack_buggy(_posterior_with_a_user_sample_dim()).dims == ("sample", "sample")


def test_selection_by_name_now_picks_the_wrong_axis():
    """`mean_over_user_samples` was meant to average the user's samples. It averages draws."""
    stacked = stack_buggy(_posterior_with_a_user_sample_dim())
    assert stacked.axis_of("sample") == 0          # the stacked chain*draw axis
    averaged = mean_over_user_samples(stacked)
    assert averaged.data.shape == (5,)             # collapsed the wrong axis
    assert averaged.dims == ("sample",)            # and the survivor still claims the name


def test_fixed_keeps_the_two_dimensions_distinct():
    stacked = stack_fixed(_posterior_with_a_user_sample_dim())
    assert stacked.dims == ("__sample__", "sample")
    assert stacked.axis_of("sample") == 1
    assert mean_over_user_samples(stacked).dims == ("__sample__",)


def test_the_fix_is_a_convention_not_a_guarantee():
    """A user dimension literally called `__sample__` breaks it again."""
    posterior = NamedArray(jnp.zeros((4, 3, 5)), ("chain", "draw", "__sample__"))
    assert stack_fixed(posterior).dims == ("__sample__", "__sample__")


def test_jax_itself_never_sees_the_names():
    """The array and its dimension names are two independent objects."""
    stacked = stack_buggy(_posterior_with_a_user_sample_dim())
    assert stacked.data.shape == (12, 5)
    assert float(stacked.data.sum()) == float(_posterior_with_a_user_sample_dim().data.sum())


def test_a_missing_name_is_the_only_thing_that_raises():
    with pytest.raises(ValueError):
        _posterior_with_a_user_sample_dim().axis_of("no_such_dim")
