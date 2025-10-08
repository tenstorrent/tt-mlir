# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn

from enum import Enum


class DeviceGetter:
    _instance = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    @classmethod
    def get_device(cls):
        if cls._instance == None:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(1, 2),
                # **updated_device_params,
                # device_id=0, l1_small_size=cls.l1_small_size
            )
            print(f"Device: {cls._instance}")
        return cls._instance


class MeshShardDirection(Enum):
    FullToShard = 0
    ShardToFull = 1


class MeshShardType(Enum):
    Identity = 0
    Replicate = 1
    Maximal = 2
    Devices = 3


ttnn.MeshShardDirection = MeshShardDirection
ttnn.MeshShardType = MeshShardType


def mesh_shard(
    input: ttnn.Tensor,
    mesh_device,
    shard_direction,
    shard_type,
    shard_shape,
    shard_dims,
):
    # TODO: implement

    if shard_type == ttnn.MeshShardType.Identity:
        return

    meshShape = mesh_device.shape
    meshDims = meshShape.dims()

    if shard_direction == ttnn.MeshShardDirection.FullToShard:
        placements = [ttnn.PlacementReplicate() for _ in range(meshDims)]
        meshMapperConfig = ttnn.MeshMapperConfig(placements, meshShape)

        sharded_input = ttnn.distribute_tensor(
            input,
            ttnn.create_mesh_mapper(
                mesh_device,
                meshMapperConfig,
            ),
        )
    elif shard_direction == ttnn.MeshShardDirection.ShardToFull:
        output = ttnn.aggregate_tensor(
            sharded_input,
            ttnn.create_mesh_composer(
                mesh_device, ttnn.MeshComposerConfig([-1, 1], ttnn.MeshShape(1, 8))
            ),
        )
    else:
        raise Exception(f"Unexpected shard_direction: {shard_direction}")

    return


ttnn.mesh_shard = mesh_shard
