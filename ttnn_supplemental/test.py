import ttnn
import ttnn_supplemental

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
