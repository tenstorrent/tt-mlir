module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @model(%arg0: tensor<1x1x256x256xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> tensor<1x1x256x512xf32> {
    %0 = "ttir.exp"(%arg0) : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
    %1 = "ttir.mesh_shard"(%0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x512xf32>
    return %1 : tensor<1x1x256x512xf32>
  }
}
