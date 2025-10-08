# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from functools import wraps
from typing import Callable, List, Sequence

from dataclasses import dataclass

from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from ttmlir.ir import *


class Marks:
    """
    Convenience class for adding pytest marks.

    Example
    -------
    >>> skip_test = Marks(pytest.mark.skip)
    >>> test_case | skip_test  # Marks test_case to be skipped
    """

    def __init__(self, *marks):
        """
        Initialize with pytest marks.

        Parameters
        ----------
        *marks : tuple
            Variable number of pytest.mark objects
        """
        self.marks = marks

    def __ror__(self, lhs):
        """
        Apply marks to a test parameter.

        Parameters
        ----------
        lhs : Any
            Test parameter to mark

        Returns
        -------
        pytest.param
            Marked test parameter
        """
        return pytest.param(lhs, marks=self.marks)


def shape_str(shape):
    """
    Converts shape tuple to string.

    Parameters
    ----------
    shape : *Union[Tuple[int, ...], List[int]]*
        Shape to convert to string

    Returns
    -------
    str
        String representation of the shape (e.g., '32x32' for shape (32, 32))
    """
    return "x".join(map(str, shape))


def make_shard_shape(
    tensor_rank: int,
    shard_dims: Sequence[int],
    mesh_shape: Sequence[int],
) -> List[int]:
    """
    Create a shard shape from a tensor rank, shard dimensions, and mesh shape.

    Parameters
    ----------
    tensor_rank : int
        Rank of the tensor
    shard_dims : Sequence[int]
        Shard dimensions, where -1 indicates no shard along that dimension
    mesh_shape : Sequence[int]
        Mesh shape, where each element represents the number of devices along that dimension

    Returns
    -------
    List[int]
        Shard shape, where each element represents the number of shards along that dimension
    """
    if len(shard_dims) != len(mesh_shape):
        raise RuntimeError("shard_dims and mesh_shape must have same length")
    shard_shape = [1] * tensor_rank
    for mesh_axis, tensor_dim in enumerate(shard_dims):
        if tensor_dim >= 0:
            shard_shape[tensor_dim] = mesh_shape[mesh_axis]
    return shard_shape


@dataclass
class ShardWrapperData:
    """Container for sharding wrapper results with better type safety."""

    input_shape: Shape
    test_fn: Callable


def shard_wrap_factory(
    test_shape: Sequence[int],
    mesh_shape: Sequence[int],
    test_fn: Callable,
) -> ShardWrapperData:
    """
    Creates a sharding wrapper for test functions.

    This function takes a test function and wraps it with automatic sharding and
    unsharding operations. The wrapper handles:
    1. Sharding the input tensor across the mesh
    2. Executing the original test function on the sharded tensor
    3. Unsharding the result back to the full tensor

    Args:
        test_shape: The shape of the input tensor to be sharded
        mesh_shape: The shape of the mesh (e.g., (2, 2) for 2x2 mesh)
        test_fn: The test function to wrap with sharding logic

    Returns:
        ShardWrapperData: A dataclass containing:
        - input_shape: The expanded input shape after sharding
        - test_fn: The test function wrapped with sharding logic

    Example:
        def my_test_fn(in0, builder):
            return builder.some_operation(in0)

        wrapper = shard_wrap_factory(
            test_shape=(1, 32, 32, 64),
            mesh_shape=(2, 2),
            test_fn=my_test_fn
        )
        # wrapper.input_shape will be (1, 32, 64, 128) - expanded for 2x2 sharding
        # wrapper.test_fn is the wrapped function
    """
    # Calculate sharding parameters
    rank_in = len(test_shape)
    rank_mesh = len(mesh_shape)

    if rank_mesh > rank_in:
        raise ValueError(
            f"Mesh shape {mesh_shape} has {rank_mesh} dimensions, but test shape "
            f"{test_shape} only has {rank_in} dimensions. Cannot shard more "
            f"dimensions than exist in the tensor."
        )

    # Take the last `rank_mesh` dims as sharded dims
    shard_dims = list(range(rank_in - rank_mesh, rank_in))
    shard_shape = make_shard_shape(rank_in, shard_dims, mesh_shape)

    full_input_shape = list(test_shape)
    for d, factor in zip(shard_dims, mesh_shape):
        full_input_shape[d] *= factor

    @wraps(test_fn)  # keep original name for debugging
    def wrapped_fn(in0: Operand, builder: TTIRBuilder):
        try:
            # sharding
            in_shard = builder.mesh_shard(
                in0,
                shard_direction="#ttcore.shard_direction<full_to_shard>",
                shard_type="#ttcore.shard_type<devices>",
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
            # op under test
            out_shard = test_fn(in_shard, builder)
            # unsharding
            return builder.mesh_shard(
                out_shard,
                shard_direction="#ttcore.shard_direction<shard_to_full>",
                shard_type="#ttcore.shard_type<devices>",
                shard_shape=shard_shape,
                shard_dims=shard_dims,
            )
        except Exception as e:
            # Provide context about the sharding operation that failed
            raise RuntimeError(
                f"Sharding operation failed for test function {test_fn.__name__} "
                f"with mesh_shape={mesh_shape}, test_shape={test_shape}: {e}"
            ) from e

    return ShardWrapperData(input_shape=full_input_shape, test_fn=wrapped_fn)
