module {
  func.func @collective_permute_wrapper(%arg0: tensor<64x256xf32>) -> tensor<64x256xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<64x256xf32>) -> tensor<32x64xf32>
    %1 = "ttir.collective_permute"(%0) <{source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0]]> : tensor<4x2xi64>}> : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 4>, shard_type = #ttcore.shard_type<devices>}> : (tensor<32x64xf32>) -> tensor<64x256xf32>
    return %2 : tensor<64x256xf32>
  }
}
