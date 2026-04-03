module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @forward(%arg0: tensor<1x1x32x256xbf16>, %arg1: tensor<256xbf16>, %arg2: tensor<256xbf16>, %arg3: tensor<1x1x32x256xbf16>) -> tensor<1x1x32x256xbf16> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x256xbf16>) -> tensor<1x1x32x128xbf16>
    %1 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<256xbf16>) -> tensor<128xbf16>
    %2 = "ttir.mesh_shard"(%arg2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<256xbf16>) -> tensor<128xbf16>
    %3 = "ttir.mesh_shard"(%arg3) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x256xbf16>) -> tensor<1x1x32x128xbf16>
    %4 = "ttir.distributed_layer_norm"(%0, %1, %2, %3) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<1x1x32x128xbf16>, tensor<128xbf16>, tensor<128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    %5 = "ttir.mesh_shard"(%4) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x256xbf16>
    return %5 : tensor<1x1x32x256xbf16>
  }
}
