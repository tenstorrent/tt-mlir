// RUN: ttmlir-opt --ttir-decompose-complex-reshape %s | FileCheck %s

// ============================================================================
// Singleton Transpose: Sub-case B (input has trailing 1, output doesn't)
// Permute swaps last two dims of input, then reshape to output if needed.
// ============================================================================

// [32, 1] -> [1, 32]: swap last two -> done
// CHECK-LABEL: @singleton_transpose_swap_trailing_1_2d
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<32x1xf32>) -> tensor<1x32xf32>
// CHECK-NOT: "ttir.reshape"
func.func @singleton_transpose_swap_trailing_1_2d(%arg0: tensor<32x1xf32>) -> tensor<1x32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}> : (tensor<32x1xf32>) -> tensor<1x32xf32>
  return %0 : tensor<1x32xf32>
}

// [1, 64, 1] -> [1, 1, 64]: swap last two -> done
// CHECK-LABEL: @singleton_transpose_swap_trailing_1_3d
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1>}>
// CHECK-SAME: (tensor<1x64x1xf32>) -> tensor<1x1x64xf32>
// CHECK-NOT: "ttir.reshape"
func.func @singleton_transpose_swap_trailing_1_3d(%arg0: tensor<1x64x1xf32>) -> tensor<1x1x64xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<1x64x1xf32>) -> tensor<1x1x64xf32>
  return %0 : tensor<1x1x64xf32>
}

// ============================================================================
// Singleton Transpose: Sub-case B fallback (input swap identity, use output swap)
// When input's last two dims are both 1, reshape to output-swapped then permute.
// ============================================================================

// [64, 1, 1] -> [1, 1, 64]: reshape to [1, 64, 1], permute swap
// CHECK-LABEL: @singleton_transpose_trailing_ones_identity_swap
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 1 : i32]}>
// CHECK-SAME: (tensor<64x1x1xf32>) -> tensor<1x64x1xf32>
// CHECK: "ttir.permute"({{%.*}}) <{permutation = array<i64: 0, 2, 1>}>
// CHECK-SAME: (tensor<1x64x1xf32>) -> tensor<1x1x64xf32>
func.func @singleton_transpose_trailing_ones_identity_swap(%arg0: tensor<64x1x1xf32>) -> tensor<1x1x64xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 64 : i32]}> : (tensor<64x1x1xf32>) -> tensor<1x1x64xf32>
  return %0 : tensor<1x1x64xf32>
}

// ============================================================================
// Singleton Transpose: Sub-case A (output has trailing 1, input doesn't)
// Reshape to output-with-last-two-swapped, then permute swap last two.
// ============================================================================

// [1, 128] -> [128, 1]: input already equals output-swapped, just permute
// CHECK-LABEL: @singleton_transpose_add_trailing_1_2d
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<1x128xf32>) -> tensor<128x1xf32>
// CHECK-NOT: "ttir.reshape"
func.func @singleton_transpose_add_trailing_1_2d(%arg0: tensor<1x128xf32>) -> tensor<128x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [128 : i32, 1 : i32]}> : (tensor<1x128xf32>) -> tensor<128x1xf32>
  return %0 : tensor<128x1xf32>
}

// [1, 1, 64] -> [1, 64, 1]: input equals output-swapped, just permute
// CHECK-LABEL: @singleton_transpose_add_trailing_1_3d
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 0, 2, 1>}>
// CHECK-SAME: (tensor<1x1x64xf32>) -> tensor<1x64x1xf32>
// CHECK-NOT: "ttir.reshape"
func.func @singleton_transpose_add_trailing_1_3d(%arg0: tensor<1x1x64xf32>) -> tensor<1x64x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<1x1x64xf32>) -> tensor<1x64x1xf32>
  return %0 : tensor<1x64x1xf32>
}

// [32, 8, 17] -> [32, 8, 17, 1]: reshape to [32, 8, 1, 17], permute swap
// CHECK-LABEL: @singleton_transpose_add_trailing_1_rank_increase
// CHECK: "ttir.reshape"(%arg0) <{shape = [32 : i32, 8 : i32, 1 : i32, 17 : i32]}>
// CHECK-SAME: (tensor<32x8x17xf32>) -> tensor<32x8x1x17xf32>
// CHECK: "ttir.permute"({{%.*}}) <{permutation = array<i64: 0, 1, 3, 2>}>
// CHECK-SAME: (tensor<32x8x1x17xf32>) -> tensor<32x8x17x1xf32>
func.func @singleton_transpose_add_trailing_1_rank_increase(%arg0: tensor<32x8x17xf32>) -> tensor<32x8x17x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 8 : i32, 17 : i32, 1 : i32]}> : (tensor<32x8x17xf32>) -> tensor<32x8x17x1xf32>
  return %0 : tensor<32x8x17x1xf32>
}

// ============================================================================
// Singleton Transpose: Sub-case A fallback (output swap identity, use input swap)
// When output's last two dims are both 1, swap input's last two then reshape.
// ============================================================================

// [1, 16] -> [1, 16, 1, 1]: permute [1, 0] -> [16, 1], reshape to output
// CHECK-LABEL: @singleton_transpose_output_swap_identity
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<1x16xf32>) -> tensor<16x1xf32>
// CHECK: "ttir.reshape"({{%.*}}) <{shape = [1 : i32, 16 : i32, 1 : i32, 1 : i32]}>
// CHECK-SAME: (tensor<16x1xf32>) -> tensor<1x16x1x1xf32>
func.func @singleton_transpose_output_swap_identity(%arg0: tensor<1x16xf32>) -> tensor<1x16x1x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 16 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16xf32>) -> tensor<1x16x1x1xf32>
  return %0 : tensor<1x16x1x1xf32>
}

// ============================================================================
// Flatten and Swap: trailing 1s become leading 1s with a genuine reshape
// Strategy: flatten to (X, 1), swap to (1, X), reshape to output.
// ============================================================================

// [128, 4, 1] -> [1, 512]: reshape to [512, 1], permute [1, 0]
// CHECK-LABEL: @flatten_and_swap_collapse
// CHECK: "ttir.reshape"(%arg0) <{shape = [512 : i32, 1 : i32]}>
// CHECK-SAME: (tensor<128x4x1xf32>) -> tensor<512x1xf32>
// CHECK: "ttir.permute"({{%.*}}) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<512x1xf32>) -> tensor<1x512xf32>
func.func @flatten_and_swap_collapse(%arg0: tensor<128x4x1xf32>) -> tensor<1x512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<128x4x1xf32>) -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// ============================================================================
// No Transform: reshapes that should NOT be decomposed
// ============================================================================

// Adding leading 1 - trailing status unchanged
// CHECK-LABEL: @no_transform_add_leading_one
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_add_leading_one(%arg0: tensor<32xf32>) -> tensor<1x32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}> : (tensor<32xf32>) -> tensor<1x32xf32>
  return %0 : tensor<1x32xf32>
}

// Adding middle 1s - trailing status unchanged
// CHECK-LABEL: @no_transform_add_middle_ones
// CHECK: "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 1 : i32, 360 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_add_middle_ones(%arg0: tensor<32x360xf32>) -> tensor<32x1x1x360xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 1 : i32, 360 : i32]}> : (tensor<32x360xf32>) -> tensor<32x1x1x360xf32>
  return %0 : tensor<32x1x1x360xf32>
}

// Moving singleton between non-1 dims, trailing status same (both trailing 1)
// CHECK-LABEL: @no_transform_move_singleton_same_trailing
// CHECK: "ttir.reshape"(%arg0) <{shape = [64 : i32, 1 : i32, 1 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_move_singleton_same_trailing(%arg0: tensor<1x64x1xf32>) -> tensor<64x1x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x1xf32>) -> tensor<64x1x1xf32>
  return %0 : tensor<64x1x1xf32>
}

// 4d: moving leading 1, trailing status same (both trailing 1)
// CHECK-LABEL: @no_transform_4d_same_trailing
// CHECK: "ttir.reshape"(%arg0) <{shape = [2 : i32, 1 : i32, 3 : i32, 1 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_4d_same_trailing(%arg0: tensor<1x2x3x1xf32>) -> tensor<2x1x3x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 1 : i32, 3 : i32, 1 : i32]}> : (tensor<1x2x3x1xf32>) -> tensor<2x1x3x1xf32>
  return %0 : tensor<2x1x3x1xf32>
}

// Non-1 dims differ (true reshape)
// CHECK-LABEL: @no_transform_true_reshape
// CHECK: "ttir.reshape"(%arg0) <{shape = [32 : i32, 2 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_true_reshape(%arg0: tensor<2x32xf32>) -> tensor<32x2xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 2 : i32]}> : (tensor<2x32xf32>) -> tensor<32x2xf32>
  return %0 : tensor<32x2xf32>
}

// Collapse all dims
// CHECK-LABEL: @no_transform_collapse_all
// CHECK: "ttir.reshape"(%arg0) <{shape = [24 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_collapse_all(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [24 : i32]}> : (tensor<2x3x4xf32>) -> tensor<24xf32>
  return %0 : tensor<24xf32>
}

// Expand to multiple dims
// CHECK-LABEL: @no_transform_expand_all
// CHECK: "ttir.reshape"(%arg0) <{shape = [2 : i32, 3 : i32, 4 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_expand_all(%arg0: tensor<24xf32>) -> tensor<2x3x4xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 3 : i32, 4 : i32]}> : (tensor<24xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// Leading 1 preserved, no trailing 1 movement
// CHECK-LABEL: @no_transform_leading_one_preserved
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_leading_one_preserved(%arg0: tensor<1x128x4xf32>) -> tensor<1x512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<1x128x4xf32>) -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// No singletons involved
// CHECK-LABEL: @no_transform_no_singletons
// CHECK: "ttir.reshape"(%arg0) <{shape = [512 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_no_singletons(%arg0: tensor<128x4xf32>) -> tensor<512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [512 : i32]}> : (tensor<128x4xf32>) -> tensor<512xf32>
  return %0 : tensor<512xf32>
}

// Identity reshape
// CHECK-LABEL: @no_transform_identity
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32, 1 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_identity(%arg0: tensor<1x32x1xf32>) -> tensor<1x32x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  return %0 : tensor<1x32x1xf32>
}

// Adding trailing 1 to shape that already has trailing 1 (same status)
// CHECK-LABEL: @no_transform_add_trailing_one_to_trailing
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 1 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_add_trailing_one_to_trailing(%arg0: tensor<64x1xf32>) -> tensor<1x64x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 64 : i32, 1 : i32]}> : (tensor<64x1xf32>) -> tensor<1x64x1xf32>
  return %0 : tensor<1x64x1xf32>
}
