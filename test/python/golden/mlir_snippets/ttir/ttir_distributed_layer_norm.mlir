module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @forward(%arg0: tensor<1x1x32x256xbf16>) -> tensor<1x1x32x256xbf16> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x256xbf16>) -> tensor<1x1x32x128xbf16>
    %1 = "ttir.distributed_layer_norm"(%0) <{cluster_axis = 1 : ui32, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1, 3>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1, 1, 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x256xbf16>
    return %2 : tensor<1x1x32x256xbf16>
  }
}
