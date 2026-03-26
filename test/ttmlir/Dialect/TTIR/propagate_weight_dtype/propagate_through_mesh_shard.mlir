// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through a mesh_shard op to the matmul.
// MeshShardOp does not have the TensorManipulation trait, but should still
// be traced through since it commonly appears on weight paths in tensor
// parallel models.

module {
  // CHECK-LABEL: func.func @propagate_through_mesh_shard
  func.func @propagate_through_mesh_shard(
    %arg0: tensor<32x64xbf16>,
    %arg1: tensor<64x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf8"}
  ) -> tensor<32x64xbf16> {
    %0 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<identity>}> : (tensor<64x128xbf16>) -> tensor<64x64xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf8"
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x64xbf16>, tensor<64x64xbf16>) -> tensor<32x64xbf16>
    return %1 : tensor<32x64xbf16>
  }
}
