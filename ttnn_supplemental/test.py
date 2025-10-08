import ttnn
import ttnn_supplemental

# Inject all ttnn_supplemental CCL operations into ttnn namespace
ttnn.MeshShardDirection = ttnn_supplemental.MeshShardDirection
ttnn.MeshShardType = ttnn_supplemental.MeshShardType
ttnn.mesh_shard = ttnn_supplemental.mesh_shard
ttnn.all_gather = ttnn_supplemental.all_gather
ttnn.reduce_scatter = ttnn_supplemental.reduce_scatter
ttnn.collective_permute = ttnn_supplemental.collective_permute
ttnn.point_to_point = ttnn_supplemental.point_to_point

a = ttnn.ones([1024])

meshDevice = ttnn.open_mesh_device(
    mesh_shape=ttnn.MeshShape(1, 2),
)

ttnn.mesh_shard(
    a,
    meshDevice,
    ttnn.MeshShardDirection.FullToShard,
    ttnn.MeshShardType.Devices,
    [2],
    [-1, 0],
)
