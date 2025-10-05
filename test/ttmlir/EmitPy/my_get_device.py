# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import ttnn


class DeviceGetter:
    _instance = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    @classmethod
    def get_device(cls):
        if cls._instance == None:
            cls._instance = ttnn.open_device(
                device_id=0, l1_small_size=cls.l1_small_size
            )
        return cls._instance


def mesh_shard(
    input: ttnn.Tensor,
    mesh_device,
    shard_direction,
    shard_type,
    shard_shape,
    shard_dims,
):
    # TODO: Convert all strings to enums

    if shard_type == "Identity":
        return

    if shard_direction == "FullToShard":
        sharded_input = ttnn.distribute_tensor(
            input,
            ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(
                    [ttnn.PlacementReplicate(), ttnn.PlacementShard(1)],
                    ttnn.MeshShape(1, 8),
                ),
            ),
        )
    elif shard_direction == "ShardToFull":
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
