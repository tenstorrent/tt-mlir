module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 2x4>]>} {
  func.func @forward2(%arg0: tensor<1x1x256x512xf32>) -> tensor<1x1x256x128xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x128x128xf32>
    %1 = "ttir.all_reduce"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x128x128xf32>) -> tensor<1x1x128x128xf32>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: 2, -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 2, 1>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x128x128xf32>) -> tensor<1x1x256x128xf32>
    return %2 : tensor<1x1x256x128xf32>
  }
}
