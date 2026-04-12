module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @selective_reduce_combine_wrapper(%arg0: tensor<16x256x1x2880xbf16>, %arg1: tensor<16x256x1x2880xbf16>, %arg2: tensor<1x256x1x4xi64>, %arg3: tensor<1x256x1x1xi64>) -> tensor<16x256x1x2880xbf16> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<16x256x1x2880xbf16>) -> tensor<16x128x1x2880xbf16>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<16x256x1x2880xbf16>) -> tensor<16x128x1x2880xbf16>
    %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x256x1x4xi64>) -> tensor<1x128x1x4xi64>
    %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x256x1x1xi64>) -> tensor<1x128x1x1xi64>
    %4 = "ttir.selective_reduce_combine"(%0, %1, %2, %3) <{hidden_size = 2880 : ui32, batch_size = 128 : ui32, seq_size = 1 : ui32, select_experts_k = 4 : ui32, experts = 32 : ui32, axis = 1 : ui32}> : (tensor<16x128x1x2880xbf16>, tensor<16x128x1x2880xbf16>, tensor<1x128x1x4xi64>, tensor<1x128x1x1xi64>) -> tensor<16x128x1x2880xbf16>
    %5 = "ttir.mesh_shard"(%4) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<16x128x1x2880xbf16>) -> tensor<16x256x1x2880xbf16>
    return %5 : tensor<16x256x1x2880xbf16>
  }
}
