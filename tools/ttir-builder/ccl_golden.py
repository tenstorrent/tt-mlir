# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List, Union
from ttmlir.ir import Attribute
import torch
from sharded_tensor import ShardedTensor, TensorLike
import itertools

# We cannot inspect the intermediate buffer on a multi-device.
# Therefore, we only support Graph Level golden.
# Although generating an Op level golden is not needed,
# we return a random torch.Tensor with the correct output shape and type for TTIR.

# ToDo(hongseok): Develop support for generating Op Level golden for CCL Ops.


def shard_tensor(
    tensor: torch.Tensor, mesh_shape: Tuple[int], shard_dims: Tuple[Union[int, None]]
) -> List[torch.Tensor]:
    """
    Shards or replicates a tensor based on mesh shape and shard dimensions.

    Args:
        tensor (torch.Tensor): Tensor to shard.
        mesh_shape (Tuple[int]): Number of shards or replicas per dimension.
        shard_dims (Tuple[int or None]): Shard dimension for each level, or -1/None to replicate.

    Returns:
        List[torch.Tensor]: List of resulting tensor shards or replicas.
    """
    assert len(mesh_shape) == len(
        shard_dims
    ), "mesh_shape and shard_dims must have the same length"

    shards = [tensor]
    for dim_size, shard_dim in zip(mesh_shape, shard_dims):
        temp_shards = []
        if shard_dim is None or shard_dim == -1:
            # replicate each tensor dim_size times
            for shard in shards:
                temp_shards.extend([shard.clone() for _ in range(dim_size)])
        else:
            # split tensor into dim_size chunks along shard_dim
            for shard in shards:
                temp_shards.extend(torch.chunk(shard, dim_size, dim=shard_dim))
        shards = temp_shards
    return shards


def unshard_tensor(
    shards: List[torch.Tensor],
    mesh_shape: Tuple[int],
    shard_dims: Tuple[Union[int, None]],
) -> torch.Tensor:
    """
    Reconstructs the original tensor from shards. Supports both sharding and replication.

    Args:
        shards (List[torch.Tensor]): Sharded or replicated tensor list.
        mesh_shape (Tuple[int]): Number of shards or replicas per dimension.
        shard_dims (Tuple[int or None]): Dimensions along which the tensor was sharded,
                                         or -1/None if it was replicated.

    Returns:
        torch.Tensor: The reconstructed tensor.
    """
    assert len(mesh_shape) == len(
        shard_dims
    ), "mesh_shape and shard_dims must have the same length"

    for dim_size, shard_dim in zip(reversed(mesh_shape), reversed(shard_dims)):
        if shard_dim is None or shard_dim == -1:
            # It was replication: only keep one copy per group
            shards = shards[::dim_size]
        else:
            # It was sharding: group and concatenate
            temp_shards = []
            for i in range(0, len(shards), dim_size):
                concat_shard = torch.cat(shards[i : i + dim_size], dim=shard_dim)
                temp_shards.append(concat_shard)
            shards = temp_shards

    assert len(shards) == 1, "Unsharding failed to reduce to a single tensor"
    return shards[0]


def mesh_shard_golden(
    input: TensorLike,
    mesh_shape: Tuple[int],
    shard_type: Attribute,
    shard_direction: Attribute,
    shard_shape: Tuple[int],
    shard_dims: List[int],
) -> TensorLike:
    shard_direction_str = str(shard_direction).lower()
    shard_type_str = str(shard_type).lower()
    if "full_to_shard" in shard_direction_str:
        assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"
        if "replicate" in shard_type_str:
            shard_dims = [None] * len(mesh_shape)
        shards = shard_tensor(input, mesh_shape, shard_dims)
        return ShardedTensor(shards, mesh_shape)
    elif "shard_to_full" in shard_direction_str:
        assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
        full = unshard_tensor(input.shards, mesh_shape, shard_dims)
        return full


def _ravel_nd(idx: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    """Convert N-D index to row-major flat index."""
    out, mult = 0, 1
    for size, i in zip(reversed(shape), reversed(idx)):
        out += i * mult
        mult *= size
    return out


def _replica_groups(shape: Tuple[int, ...], cluster_axis: int) -> List[List[int]]:
    """Groups that vary cluster_axis while fixing all other axes."""
    other_axes = [ax for ax in range(len(shape)) if ax != cluster_axis]
    groups: List[List[int]] = []
    for fixed in itertools.product(*[range(shape[ax]) for ax in other_axes]):
        group: List[int] = []
        for v in range(shape[cluster_axis]):
            full = list(fixed)
            full.insert(cluster_axis, v)
            group.append(_ravel_nd(tuple(full), shape))
        groups.append(group)
    return groups


def all_gather_golden(
    input: ShardedTensor,
    mesh_shape: Tuple[int],
    all_gather_dim: int,
    cluster_axis: int,
) -> torch.Tensor:
    assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
    output = input.clone()
    replica_groups = _replica_groups(mesh_shape, cluster_axis)
    for group in replica_groups:
        group_tensors = [input.get_shard(i) for i in group]
        group_tensor = torch.cat(group_tensors, dim=all_gather_dim)
        for i in group:
            output.shards[i] = group_tensor
    return output


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
