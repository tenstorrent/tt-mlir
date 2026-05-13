// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through mesh_shard and mesh_partition ops
// to the sparse_matmul. This is the typical pattern for MoE expert weights in
// tensor-parallel models.

module {
  // CHECK-LABEL: func.func @propagate_through_mesh_partition
  func.func @propagate_through_mesh_partition(
    %arg0: tensor<4x4x32x2880xbf16>,
    %arg1: tensor<128x2880x5760xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"},
    %arg2: tensor<4x4x1x32xbf16>
  ) -> tensor<4x4x1x32x32x5760xbf16> {
    %0 = "ttir.mesh_shard"(%arg1) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<128x2880x5760xbf16>) -> tensor<128x2880x5760xbf16>
    %1 = "ttir.mesh_partition"(%0) <{cluster_axis = 1 : ui32, dim = 0 : si32}> : (tensor<128x2880x5760xbf16>) -> tensor<32x2880x5760xbf16>
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 32 : i32, 2880 : i32, 5760 : i32]}> : (tensor<32x2880x5760xbf16>) -> tensor<1x32x2880x5760xbf16>
    // CHECK: "ttir.sparse_matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    %3 = "ttir.sparse_matmul"(%arg0, %2, %arg2) <{is_input_a_sparse = false, is_input_b_sparse = true, nnz = 0 : i64}> : (tensor<4x4x32x2880xbf16>, tensor<1x32x2880x5760xbf16>, tensor<4x4x1x32xbf16>) -> tensor<4x4x1x32x32x5760xbf16>
    return %3 : tensor<4x4x1x32x32x5760xbf16>
  }
}
