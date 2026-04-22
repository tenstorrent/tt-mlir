# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from functools import wraps
from typing import Callable, List, Sequence

from dataclasses import dataclass

from builder.base.builder_utils import Operand, Shape
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
        if not isinstance(lhs, tuple):
            lhs = (lhs,)
        return pytest.param(*lhs, marks=self.marks)


class SkipIf:
    """
    Convenience class for adding pytest skip_config.

    Example
    -------
    # Marks ttmetal target as skip OR (ttnn AND sim) as skip
    >>> test_case | SkipConfig("ttmetal", ["ttnn", "sim"])
    """

    def __init__(self, *marks_groups, mark_fn=pytest.mark.skip_config, reason=None):
        self.marks_groups = [g if isinstance(g, list) else [g] for g in marks_groups]
        self.mark_fn = mark_fn
        self.reason = reason

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
        kwargs = {"reason": self.reason} if self.reason is not None else {}
        return pytest.param(
            lhs, marks=[self.mark_fn(g, **kwargs) for g in self.marks_groups]
        )


class OnlyIf(SkipIf):
    def __init__(self, *marks_groups):
        super().__init__(*marks_groups, mark_fn=pytest.mark.only_config)


class SkipExecIf(SkipIf):
    def __init__(self, *marks_groups):
        super().__init__(*marks_groups, mark_fn=pytest.mark.skip_exec)


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


def shapes_list_str(shapes):
    """
    Converts a list of shapes to string, joined by "-".
    Parameters
    ----------
    shapes : Sequence[*Union[Tuple[int, ...], List[int]]*]
        Shapes to convert to string
    Returns
    -------
    str
        String representation of the shapes (e.g., '1x2-3x4' for input [(1, 2), (3, 4)])
    """
    return "-".join(shape_str(s) for s in shapes)


def sharding_str(sharding):
    """
    Converts shape tuple to string.

    Parameters
    ----------
    sharding : *List[Tuple[str, bool]]*
        Sharding annotation config to convert to string

    Returns
    -------
    str
        String representation of the sharding config (e.g., '32x32' for shape (32, 32))
    """
    return "".join(map(str, sharding))


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


class SystemDesc:
    def __init__(self, system_desc):
        self._system_desc = system_desc

    def __getitem__(self, key):
        return self._system_desc[key]

    def __iter__(self):
        return iter(self._system_desc)

    def __len__(self):
        return len(self._system_desc)

    def _readonly(self, *_, **__) -> None:
        raise TypeError(f"{type(self).__name__!r} is read-only")

    __setitem__ = __delitem__ = _readonly

    def get_arch(self, chip_id=0):
        return self._system_desc["chip_descs"][chip_id]["arch"]

    def get_grid_shape(self, chip_id=0):
        grid_shape = self._system_desc["chip_descs"][chip_id]["grid_size"]
        return (grid_shape["y"], grid_shape["x"])

    def get_num_cores(self, chip_id=0):
        y, x = self.get_grid_shape(chip_id=chip_id)
        return y * x

    def calc_fpu_tops(
        self, num_fpu_ops, dtype="bf16", units="ns", format_as_string=True
    ):
        arch_tensix_tops = {
            "Wormhole_b0": {
                "bfp4": 5.45,
                "bfp8": 2.72,
                "fp16": 1.36,  # maps to tf32
                "bf16": 1.36,  # maps to tf32
                "fp32": 1.36,  # maps to tf32
                "tf32": 1.36,
                "int8": 1.36,
            },
            "Blackhole": {
                "bfp4": 6.14,
                "bfp8": 3.07,
                "fp16": 1.54,  # maps to tf32
                "bf16": 1.54,  # maps to tf32
                "fp32": 1.54,  # maps to tf32
                "tf32": 1.54,
                "int8": 1.54,
            },
        }[self.get_arch()][dtype]

        unit_scale = {
            "ns": 9,
            "us": 6,
            "ms": 3,
            "s": 0,
        }[units]

        arch_tops = arch_tensix_tops * self.get_num_cores()
        compute_tops = num_fpu_ops / (10**12)
        sec = compute_tops / arch_tops
        result = sec * 10**unit_scale
        if format_as_string:
            return f"{result:.4g}{units}"
        else:
            return result
