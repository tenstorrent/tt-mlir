import ttnn
import ttnn_supplemental

# Add ttnn_supplemental exports to ttnn namespace
ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
ttnn.MeshShardType = ttnn_supplemental.MeshShardType
ttnn.mesh_shard = ttnn_supplemental.mesh_shard

a = ttnn.ones([1024])

# print(ttnn_supplemental.add(3, 5))

meshDevice = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 2),
)

ttnn_supplemental.mesh_shard(
    a,
    meshDevice,
    ttnn.MeshShardDirection.FullToShard,
    ttnn.MeshShardType.Devices,
    [2],
    [-1, 0],
)
