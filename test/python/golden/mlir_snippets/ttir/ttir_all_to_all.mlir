module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @all_to_all_wrapper(%arg0: tensor<8x8x64x128xbf16>) -> tensor<8x8x64x128xbf16> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<8x8x64x128xbf16>) -> tensor<8x8x64x64xbf16>
    %1 = "ttir.all_to_all"(%0) <{concat_dim = 0 : si32, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : si32, split_dim = 0 : si32}> : (tensor<8x8x64x64xbf16>) -> tensor<8x8x64x64xbf16>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: 2, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<8x8x64x64xbf16>) -> tensor<8x8x64x128xbf16>
    return %2 : tensor<8x8x64x128xbf16>
  }
}
