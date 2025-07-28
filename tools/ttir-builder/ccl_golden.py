# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List
from ttmlir.ir import Attribute
import torch

# We cannot inspect the intermediate buffer on a multi-device.
# Therefore, we only support Graph Level golden.
# Although generating an Op level golden is not needed,
# we return a random torch.Tensor with the correct output shape and type for TTIR.

# ToDo(hongseok): Develop support for generating Op Level golden for CCL Ops.


def mesh_shard_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    shard_type: Attribute,
    shard_direction: Attribute,
    shard_shape: Tuple[int, int],
    shard_dims: List[int],
) -> torch.Tensor:
    # Return a random torch.Tensor which has the correct shape and type after doing mesh_shard on the input.

    out_shape = list(input.shape)
    if "devices" in str(shard_type).lower():
        for shard_dim in shard_dims:
            if shard_dim == -1:
                continue
            if "shard_to_full" in str(shard_direction).lower():
                out_shape[shard_dim] *= shard_shape[shard_dim]
            elif "full_to_shard" in str(shard_direction).lower():
                out_shape[shard_dim] //= shard_shape[shard_dim]
    return torch.randn(out_shape, dtype=input.dtype)


def all_gather_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
) -> torch.Tensor:
    # Return a random torch.Tensor which has the correct shape and type after doing all_gather on the input.
    out_shape = list(input.shape)
    out_shape[all_gather_dim] *= mesh_shape[cluster_axis]
    return torch.randn(out_shape, dtype=input.dtype)


def all_reduce_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    reduce_type: Attribute,
) -> torch.Tensor:
    # Return a random torch.Tensor which has the correct shape and type after doing all_reduce on the input.
    return torch.randn(input.shape, dtype=input.dtype)


def reduce_scatter_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    reduce_type: Attribute,
    scatter_dim: int,
    cluster_axis: int,
) -> torch.Tensor:
    # Return a random torch.Tensor which has the correct shape and type after doing reduce_scatter on the input.
    out_shape = list(input.shape)
    out_shape[scatter_dim] //= mesh_shape[cluster_axis]
    return torch.randn(out_shape, dtype=input.dtype)


def collective_permute_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    source_target_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    # Return a random torch.Tensor which has the correct shape and type after doing collective_permute on the input.
    return torch.randn(input.shape, dtype=input.dtype)


def all_to_all_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    split_dim: int,
    concat_dim: int,
    split_count: int,
    replica_groups: List[List[int]],
) -> torch.Tensor:
    # Return a random torch.Tensor which has the correct shape and type after doing all_gather on the input.
    out_shape = list(input.shape)
    out_shape[split_dim] //= split_count
    out_shape[concat_dim] *= split_count
    return torch.randn(out_shape, dtype=input.dtype)
