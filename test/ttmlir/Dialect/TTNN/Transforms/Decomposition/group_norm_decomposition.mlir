// RUN: ttmlir-opt --ttcore-register-device --ttnn-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 9216 + d1 * 9216 + d2, d3), <1x1>, memref<73728x128xf32, #dram>, <interleaved>>
#ttnn_layout_affine = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x128xf32, #dram>, <interleaved>>
#ttnn_layout_non_aligned = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 50 + d1 * 50 + d2, d3), <1x1>, memref<50x480xf32, #dram>, <interleaved>>
// N=2, per-sample H*W=16 (N*H*W=32 looks aligned, but H*W=16 is not).
#ttnn_layout_n2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 16 + d2, d3), <1x1>, memref<32x256xf32, #dram>, <interleaved>>

// Test: full group_norm with weight and bias decomposes to primitives.
// CHECK-LABEL: func.func @group_norm_weight_bias
// CHECK-NOT: "ttnn.group_norm"
// Reshape input from [N, 1, S, C] to [N, S, G, Cpg].
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [8 : i32, 9216 : i32, 32 : i32, 4 : i32]
// Per-group mean over (S, Cpg) = dims [1, 3].
// CHECK: "ttnn.mean"
// CHECK-SAME: dim_arg = [1 : i32, 3 : i32], keep_dim = true
// Centering.
// CHECK: "ttnn.subtract"
// Square and variance.
// CHECK: "ttnn.multiply"
// CHECK: "ttnn.mean"
// CHECK-SAME: dim_arg = [1 : i32, 3 : i32], keep_dim = true
// eps tensor + (var + eps) + rsqrt.
// CHECK: "ttnn.full"
// CHECK: "ttnn.add"
// CHECK: "ttnn.rsqrt"
// Normalize and restore canonical [N, 1, S, C] shape.
// CHECK: "ttnn.multiply"
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [8 : i32, 1 : i32, 9216 : i32, 128 : i32]
// Affine tail: weight reshape [C] -> [1, 1, 1, C], multiply.
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]
// CHECK: "ttnn.multiply"
// Affine tail: bias reshape [C] -> [1, 1, 1, C], add.
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]
// CHECK: "ttnn.add"
func.func @group_norm_weight_bias(
    %arg0: tensor<8x1x9216x128xf32, #ttnn_layout>,
    %arg1: tensor<128xf32, #ttnn_layout_affine>,
    %arg2: tensor<128xf32, #ttnn_layout_affine>) -> tensor<8x1x9216x128xf32, #ttnn_layout> {
  %0 = "ttnn.group_norm"(%arg0, %arg1, %arg2) <{num_groups = 32 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 1, 1>}> : (tensor<8x1x9216x128xf32, #ttnn_layout>, tensor<128xf32, #ttnn_layout_affine>, tensor<128xf32, #ttnn_layout_affine>) -> tensor<8x1x9216x128xf32, #ttnn_layout>
  return %0 : tensor<8x1x9216x128xf32, #ttnn_layout>
}

// Test: group_norm without weight or bias still decomposes; no affine tail
// is emitted.
// CHECK-LABEL: func.func @group_norm_no_affine
// CHECK-NOT: "ttnn.group_norm"
// Group-split reshape, two means (mean and variance), and the normalize tail.
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [8 : i32, 9216 : i32, 32 : i32, 4 : i32]
// CHECK: "ttnn.mean"
// CHECK: "ttnn.subtract"
// CHECK: "ttnn.multiply"
// CHECK: "ttnn.mean"
// CHECK: "ttnn.full"
// CHECK: "ttnn.add"
// CHECK: "ttnn.rsqrt"
// CHECK: "ttnn.multiply"
// Restoring reshape is the LAST reshape - there must be no affine multiply or
// affine reshape after it.
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [8 : i32, 1 : i32, 9216 : i32, 128 : i32]
// CHECK-NOT: "ttnn.multiply"
// CHECK-NOT: "ttnn.add"
// CHECK: return
func.func @group_norm_no_affine(
    %arg0: tensor<8x1x9216x128xf32, #ttnn_layout>) -> tensor<8x1x9216x128xf32, #ttnn_layout> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 32 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<8x1x9216x128xf32, #ttnn_layout>) -> tensor<8x1x9216x128xf32, #ttnn_layout>
  return %0 : tensor<8x1x9216x128xf32, #ttnn_layout>
}

// Non-tile-aligned flattened height (N*S = 1*50 = 50): now representable (the
// verifier no longer rejects it) and decomposed into primitives by this pass.
// CHECK-LABEL: func.func @group_norm_non_tile_aligned
// CHECK-NOT: "ttnn.group_norm"
// Group-split reshape [N, 1, S, C] -> [N, S, G, Cpg] = [1, 50, 8, 60].
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [1 : i32, 50 : i32, 8 : i32, 60 : i32]
// CHECK: "ttnn.mean"
// CHECK: "ttnn.subtract"
// CHECK: "ttnn.multiply"
// CHECK: "ttnn.mean"
// CHECK: "ttnn.rsqrt"
// CHECK: "ttnn.multiply"
// Restore canonical [N, 1, S, C] shape.
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [1 : i32, 1 : i32, 50 : i32, 480 : i32]
func.func @group_norm_non_tile_aligned(
    %arg0: tensor<1x1x50x480xf32, #ttnn_layout_non_aligned>) -> tensor<1x1x50x480xf32, #ttnn_layout_non_aligned> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<1x1x50x480xf32, #ttnn_layout_non_aligned>) -> tensor<1x1x50x480xf32, #ttnn_layout_non_aligned>
  return %0 : tensor<1x1x50x480xf32, #ttnn_layout_non_aligned>
}

// N=2, per-sample H*W=16 (N*H*W=32 is aligned but H*W=16 is not): decompose.
// CHECK-LABEL: func.func @group_norm_n2_non_tile_aligned
// CHECK-NOT: "ttnn.group_norm"
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [2 : i32, 16 : i32, 8 : i32, 32 : i32]
// CHECK: "ttnn.mean"
// CHECK: "ttnn.reshape"
// CHECK-SAME: shape = [2 : i32, 1 : i32, 16 : i32, 256 : i32]
func.func @group_norm_n2_non_tile_aligned(
    %arg0: tensor<2x1x16x256xf32, #ttnn_layout_n2>) -> tensor<2x1x16x256xf32, #ttnn_layout_n2> {
  %0 = "ttnn.group_norm"(%arg0) <{num_groups = 8 : i64, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<2x1x16x256xf32, #ttnn_layout_n2>) -> tensor<2x1x16x256xf32, #ttnn_layout_n2>
  return %0 : tensor<2x1x16x256xf32, #ttnn_layout_n2>
}
