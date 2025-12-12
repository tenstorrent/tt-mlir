module {
  func.func @collective_permute_wrapper(%arg0: tensor<32x128xbf16>) -> tensor<32x128xbf16> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<32x128xbf16>) -> tensor<32x64xbf16>
    %1 = "ttir.collective_permute"(%0) <{source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<32x64xbf16>) -> tensor<32x64xbf16>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<32x64xbf16>) -> tensor<32x128xbf16>
    return %2 : tensor<32x128xbf16>
  }
}
