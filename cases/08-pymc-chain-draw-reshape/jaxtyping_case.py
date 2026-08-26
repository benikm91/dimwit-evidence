"""Case 08 under jaxtyping — `plain_jax.py` with the axis names written into annotations.

This is one of the cases where jaxtyping does fire, because the defect changes a shape that
an annotation can mention. Two conditions have to hold: someone has to write the output
annotation, and the two candidate rectangles have to differ.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype

SAMPLE_DIMS = ("chain", "draw")


def _transpose_to_front(data, dims, sample_dims):
    order = list(sample_dims) + [d for d in dims if d not in sample_dims]
    moved = jnp.transpose(data, [dims.index(d) for d in order])
    return moved, tuple(order)


@jaxtyped(typechecker=beartype)
def to_point_list_buggy(
    data: Float[Array, "team draw chain"],
    dims: tuple[str, ...],
    sample_dims: tuple[str, ...] = SAMPLE_DIMS,
) -> Float[Array, "chain_draw team"]:
    """As shipped: `rest` is read from the ORIGINAL array, not the transposed one."""
    moved, _ = _transpose_to_front(data, dims, sample_dims)
    rest = data.shape[len(sample_dims):]          # <- the defect
    return jnp.reshape(moved, (-1, *rest))


@jaxtyped(typechecker=beartype)
def to_point_list_fixed(
    data: Float[Array, "team draw chain"],
    dims: tuple[str, ...],
    sample_dims: tuple[str, ...] = SAMPLE_DIMS,
) -> Float[Array, "chain_draw team"]:
    """After PR #7180: transpose first, then read the shape."""
    moved, _ = _transpose_to_front(data, dims, sample_dims)
    rest = moved.shape[len(sample_dims):]
    return jnp.reshape(moved, (-1, *rest))


DIMS = ("team", "draw", "chain")


# --------------------------------------------------------------------------- tests

def test_the_buggy_output_violates_the_annotation():
    """RUN-TIME detection: `team` is bound to 5 by the input, but the output has 3 columns."""
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    with pytest.raises(Exception):
        to_point_list_buggy(data, DIMS)


def test_the_fixed_output_satisfies_it():
    data = jnp.arange(5 * 2 * 3, dtype=jnp.float32).reshape(5, 2, 3)
    assert to_point_list_fixed(data, DIMS).shape == (6, 5)


def test_it_is_missed_when_the_two_rectangles_coincide():
    """team == chain, so (chain*draw, chain) and (chain*draw, team) are the same shape."""
    data = jnp.arange(3 * 2 * 3, dtype=jnp.float32).reshape(3, 2, 3)  # team = chain = 3
    assert to_point_list_buggy(data, DIMS).shape == to_point_list_fixed(data, DIMS).shape == (6, 3)


def test_the_dims_tuple_itself_is_beyond_the_annotation():
    """The annotation can only describe one fixed layout, so `dims` is checked by nobody.

    Called with the sample dims already leading — a correct call — the very same annotation
    now rejects it, because `team` was bound to the first axis of the input.
    """
    data = jnp.arange(3 * 2 * 5, dtype=jnp.float32).reshape(3, 2, 5)
    with pytest.raises(Exception):
        to_point_list_fixed(data, ("chain", "draw", "team"))
