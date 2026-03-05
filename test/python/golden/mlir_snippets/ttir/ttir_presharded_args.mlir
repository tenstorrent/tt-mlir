module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @model(%arg0: tensor<1x1x256x512xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x1x256x256xf32>>}) -> tensor<1x1x256x512xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x256xf32>
    %1 = "ttir.exp"(%0) : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x512xf32>
    return %2 : tensor<1x1x256x512xf32>
  }
}
