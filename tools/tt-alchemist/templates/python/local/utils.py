# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn
import ttnn_supplemental


# Monkey-patch ttnn with ttnn_supplemental objects
ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
ttnn.MeshShardType = ttnn_supplemental.MeshShardType
ttnn.mesh_shard = ttnn_supplemental.mesh_shard
ttnn.all_gather = ttnn_supplemental.all_gather
ttnn.reduce_scatter = ttnn_supplemental.reduce_scatter
ttnn.collective_permute = ttnn_supplemental.collective_permute
ttnn.point_to_point = ttnn_supplemental.point_to_point


class DeviceGetter:
    _instance = None
    _mesh_shape = None
    l1_small_size = 1 << 15

    def __init__(self):
        raise RuntimeError("This is Singleton, invoke get_device() instead.")

    @classmethod
    def get_device(cls, mesh_shape):
        if cls._instance == None:
            if (
                not isinstance(mesh_shape, (list, tuple))
                or len(mesh_shape) == 0
                or not all(isinstance(x, int) and x > 0 for x in mesh_shape)
            ):
                raise ValueError(
                    f"mesh_shape must be a non-empty list or tuple of positive integers, got {mesh_shape}"
                )
            cls._mesh_shape = mesh_shape
            cls._instance = ttnn.open_mesh_device(
                mesh_shape=ttnn.MeshShape(mesh_shape),
                l1_small_size=cls.l1_small_size,
            )
            print(f"Device: {cls._instance}")

        # Compare requested mesh_shape with _mesh_shape used to initialize the device
        if tuple(cls._mesh_shape) != tuple(mesh_shape):
            raise ValueError(
                f"Device already initialized with mesh_shape={cls._mesh_shape}, but got mesh_shape={mesh_shape}"
            )

        return cls._instance


def get_scalar_from_tensor(tensor: ttnn.Tensor) -> int:
    assert tensor.logical_volume() == 1, "expected scalar tensor"
    assert tensor.dtype == ttnn.DataType.UINT32, "expected uint32 tensor"

    host_tensor = ttnn.from_device(tensor)
    return host_tensor.item()
