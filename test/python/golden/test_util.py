# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch, math, itertools
from typing import List, Tuple, Iterable, Sequence, Union


def all_sharding_configs(
    input_rank: int, num_devices: int  # e.g. 4-D tensor -> 4  # mesh size (rows*cols)
) -> Iterable[
    Tuple[
        Tuple[int, int],  # mesh_shape
        Tuple[int, int],  # shard_dims
        List[int],  # shard_shape
        int,  # cluster_axis
        List[List[int]],
    ]
]:  # replica_groups
    """
    Yield every (mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups)
    that represents a *valid* sharding of `input_rank` across `num_devices`.
    """
    # all 2-D mesh factorizations (row, col) – both orientations
    mesh_shapes = {
        (r, num_devices // r)
        for r in range(1, math.isqrt(num_devices) + 1)
        if num_devices % r == 0
    }
    mesh_shapes |= {(c, r) for r, c in mesh_shapes}

    # every pair of shard dims (-1 = not sharded)
    shard_dim_pairs = itertools.product(range(-1, input_rank), repeat=2)

    for mesh_shape, shard_dims in itertools.product(
        sorted(mesh_shapes), shard_dim_pairs
    ):
        dim_row, dim_col = shard_dims
        shard_shape = [
            mesh_shape[0] if i == dim_row else mesh_shape[1] if i == dim_col else 1
            for i in range(input_rank)
        ]

        # — skip invalids: duplicate dim or “no sharding at all”
        if dim_row == dim_col or all(s == 1 for s in shard_shape):
            continue

        rows, cols = mesh_shape
        for cluster_axis in (0, 1):
            replica_groups = (
                [
                    [r * cols + c for r in range(rows)] for c in range(cols)
                ]  # across rows
                if cluster_axis == 0
                else [
                    [r * cols + c for c in range(cols)] for r in range(rows)
                ]  # across cols
            )
            yield mesh_shape, shard_dims, shard_shape, cluster_axis, replica_groups


def shape_divisible(tensor_shape: Sequence[int], shard_shape: Sequence[int]) -> bool:
    return all(t % s == 0 for t, s in zip(tensor_shape, shard_shape))


def shardTensor2dMesh(
    tensor: torch.Tensor, mesh_shape: Tuple[int, int], shard_dims: Tuple[int, int]
):
    rows, cols = mesh_shape
    row_dim, col_dim = shard_dims

    # Shard along rows
    row_tensors = (
        [tensor.clone() for _ in range(rows)]
        if row_dim == -1
        else torch.chunk(tensor, rows, dim=row_dim)
    )

    # Shard along columns
    if col_dim == -1:
        return [t.clone() for t in row_tensors for _ in range(cols)]
    tensor_shards = [
        tt for t in row_tensors for tt in torch.chunk(t, cols, dim=col_dim)
    ]
    return tensor_shards


def concatMesh2dToTensor(
    device_shards: List[torch.Tensor],
    mesh_shape: Tuple[int, int],
    shard_dims: Tuple[int, int],
):
    rows, cols = mesh_shape
    row_dim, col_dim = shard_dims

    # Reshape the list of shards into a 2D list representing the device mesh
    mesh_shape = [
        device_shards[i : i + cols] for i in range(0, len(device_shards), cols)
    ]

    if col_dim == -1:
        row_concatenated = [row[0] for row in mesh_shape]
    else:
        row_concatenated = [torch.cat(row, dim=col_dim) for row in mesh_shape]

    # Then concatenate the resulting tensors along rows
    if row_dim == -1:
        return row_concatenated[0]
    else:
        return torch.cat(row_concatenated, dim=row_dim)
