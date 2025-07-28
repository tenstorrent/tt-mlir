# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple, List, Union
from ttmlir.ir import Attribute
import torch
from .sharded_tensor import ShardedTensor, TensorLike
from functools import reduce


def _sharding(
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


def _unsharding(
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

    # assert len(shards) == 1, "Unsharding failed to reduce to a single tensor"
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
        shards = _sharding(input, mesh_shape, shard_dims)
        return ShardedTensor(shards, mesh_shape)
    elif "shard_to_full" in shard_direction_str:
        assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
        if "replicate" in shard_type_str:
            full = _unsharding(input.shards, [1], [1])
        else:
            full = _unsharding(input.shards, mesh_shape, shard_dims)
        return full


def all_gather_golden(
    input: ShardedTensor,
    mesh_shape: Tuple[int],
    all_gather_dim: int,
    cluster_axis: int,
) -> ShardedTensor:
    assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
    output = [None] * len(input.shards)
    grouped_tensors = input.replica_groups(cluster_axis)
    for group in grouped_tensors:
        gathered_tensor = torch.cat(list(group.values()), dim=all_gather_dim)
        for id in group.keys():
            output[id] = gathered_tensor.clone()
    assert None not in output, "Not all shards are gathered"
    return ShardedTensor(output, mesh_shape)


def _reduce(inputs: List[torch.Tensor], reduce_type: Attribute) -> torch.Tensor:
    reduce_type_str = str(reduce_type).lower()
    if "sum" in reduce_type_str:
        reduced_tensor = reduce(torch.add, inputs)
    elif "mean" in reduce_type_str:
        reduced_tensor = reduce(torch.add, inputs) / len(inputs)
    elif "max" in reduce_type_str:
        reduced_tensor = reduce(torch.max, inputs)
    elif "min" in reduce_type_str:
        reduced_tensor = reduce(torch.min, inputs)
    elif "std" in reduce_type_str:
        reduced_tensor = torch.std(torch.stack(inputs), dim=0, unbiased=False)
    elif "var" in reduce_type_str:
        reduced_tensor = torch.var(torch.stack(inputs), dim=0, unbiased=False)
    else:
        raise ValueError(f"Unsupported reduce type: {reduce_type_str}")
    return reduced_tensor


def all_reduce_golden(
    input: ShardedTensor,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    reduce_type: Attribute,
) -> ShardedTensor:
    assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
    output = [None] * len(input.shards)
    grouped_tensors = input.replica_groups(cluster_axis)
    for group in grouped_tensors:
        group_tensors = list(group.values())
        reduced_tensor = _reduce(group_tensors, reduce_type)
        for id in group.keys():
            output[id] = reduced_tensor.clone()
    assert None not in output, "Not all shards are reduced"
    return ShardedTensor(output, mesh_shape)


def reduce_scatter_golden(
    input: ShardedTensor,
    mesh_shape: Tuple[int, int],
    reduce_type: Attribute,
    scatter_dim: int,
    cluster_axis: int,
) -> ShardedTensor:
    assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
    output = [None] * len(input.shards)
    grouped_tensors = input.replica_groups(cluster_axis)
    for group in grouped_tensors:
        group_tensors = list(group.values())
        reduced_tensor = _reduce(group_tensors, reduce_type)
        scattered_tensor = torch.chunk(
            reduced_tensor, mesh_shape[cluster_axis], dim=scatter_dim
        )
        for index, id in enumerate(group.keys()):
            output[id] = scattered_tensor[index].clone()
    assert None not in output, "Not all shards are reduced"
    return ShardedTensor(output, mesh_shape)


def collective_permute_golden(
    input: ShardedTensor,
    mesh_shape: Tuple[int, int],
    source_target_pairs: List[Tuple[int, int]],
) -> ShardedTensor:
    assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"
    output_shards = [torch.zeros_like(shard) for shard in input.shards]
    for src, tgt in source_target_pairs:
        output_shards[tgt] = input.shards[src].clone()
    return ShardedTensor(output_shards, mesh_shape)


def all_to_all_golden(
    input: ShardedTensor,
    mesh_shape: Tuple[int, int],
    split_dim: int,
    concat_dim: int,
    split_count: int,
    replica_groups: List[List[int]],
) -> ShardedTensor:
    assert isinstance(input, ShardedTensor), "Input must be a ShardedTensor"

    # Pre-allocate the output list
    output_shards = [None] * len(input.shards)

    for group in replica_groups:
        assert len(group) == split_count, "group size must equal split_count"
        # Split every source tensor into N = split_count chunks
        splits_per_src: List[Tuple[torch.Tensor, ...]] = [
            torch.chunk(input.shards[dev_id], split_count, dim=split_dim)
            for dev_id in group
        ]

        # Reassemble chunks: dst_idx == slice index to receive
        for dst_idx in range(split_count):
            output_shards[group[dst_idx]] = torch.cat(
                [splits_per_src[src_idx][dst_idx] for src_idx in range(split_count)],
                dim=concat_dim,
            )

    assert None not in output_shards, "Some shards were not written"
    return ShardedTensor(output_shards, mesh_shape)
