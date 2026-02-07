// RUN: ttmlir-opt --d2m-decompose-complex-reshape %s | FileCheck %s

// ============================================================================
// Pure Permute Cases - reshape is equivalent to permute (only moving singletons)
// When input and output have the same rank, the reshape is replaced with
// just a permute operation.
// ============================================================================

// Test: [32, 1] -> [1, 32]
// Non-1 dims: [32] vs [32] - same order, but singleton positions differ
// Same rank -> just permute
// CHECK-LABEL: @pure_permute_2d_swap_singleton
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<32x1xf32>) -> tensor<1x32xf32>
// CHECK-NOT: "ttir.reshape"
func.func @pure_permute_2d_swap_singleton(%arg0: tensor<32x1xf32>) -> tensor<1x32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}> : (tensor<32x1xf32>) -> tensor<1x32xf32>
  return %0 : tensor<1x32xf32>
}

// Test: [1, 64, 1] -> [64, 1, 1]
// Non-1 dims: [64] vs [64] - same order, but singleton positions differ
// CHECK-LABEL: @pure_permute_3d_move_leading_one
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}>
// CHECK-SAME: (tensor<1x64x1xf32>) -> tensor<64x1x1xf32>
// CHECK-NOT: "ttir.reshape"
func.func @pure_permute_3d_move_leading_one(%arg0: tensor<1x64x1xf32>) -> tensor<64x1x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32, 1 : i32, 1 : i32]}> : (tensor<1x64x1xf32>) -> tensor<64x1x1xf32>
  return %0 : tensor<64x1x1xf32>
}

// Test: [1, 2, 32] -> [2, 1, 32]
// Non-1 dims: [2, 32] vs [2, 32] - same order, singleton moved
// CHECK-LABEL: @pure_permute_3d_swap_leading_one
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}>
// CHECK-SAME: (tensor<1x2x32xf32>) -> tensor<2x1x32xf32>
// CHECK-NOT: "ttir.reshape"
func.func @pure_permute_3d_swap_leading_one(%arg0: tensor<1x2x32xf32>) -> tensor<2x1x32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 1 : i32, 32 : i32]}> : (tensor<1x2x32xf32>) -> tensor<2x1x32xf32>
  return %0 : tensor<2x1x32xf32>
}

// Test: [1, 1, 64] -> [64, 1, 1]
// Non-1 dims: [64] vs [64] - same order, multiple singletons move
// CHECK-LABEL: @pure_permute_3d_multiple_ones
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 1>}>
// CHECK-SAME: (tensor<1x1x64xf32>) -> tensor<64x1x1xf32>
// CHECK-NOT: "ttir.reshape"
func.func @pure_permute_3d_multiple_ones(%arg0: tensor<1x1x64xf32>) -> tensor<64x1x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [64 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x64xf32>) -> tensor<64x1x1xf32>
  return %0 : tensor<64x1x1xf32>
}

// Test: [32] -> [1, 32]
// Just adding a leading 1 - not a pure permute, should remain as reshape
// CHECK-LABEL: @no_transform_add_leading_one
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_add_leading_one(%arg0: tensor<32xf32>) -> tensor<1x32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}> : (tensor<32xf32>) -> tensor<1x32xf32>
  return %0 : tensor<1x32xf32>
}

// Test: [32, 1] -> [32]
// Removing a trailing 1 involves singleton position change
// After normalization: [32, 1] vs [1, 32] - differs, so permute needed
// Decomposed to: permute [1, 0] -> reshape to [32]
// CHECK-LABEL: @decompose_remove_trailing_one
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<32x1xf32>) -> tensor<1x32xf32>
// CHECK: "ttir.reshape"({{%.*}}) <{shape = [32 : i32]}>
// CHECK-SAME: (tensor<1x32xf32>) -> tensor<32xf32>
func.func @decompose_remove_trailing_one(%arg0: tensor<32x1xf32>) -> tensor<32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32]}> : (tensor<32x1xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}

// ============================================================================
// Singleton Permute Cases - trailing 1s need to become leading 1s before
// the actual reshape can happen
// ============================================================================

// Test: [128, 4, 1] -> [1, 512]
// Trailing 1 needs to become leading 1 before reshape
// Decomposed to: permute [2, 0, 1] -> reshape to [1, 512]
// CHECK-LABEL: @singleton_permute_trailing_to_leading
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 1>}>
// CHECK-SAME: (tensor<128x4x1xf32>) -> tensor<1x128x4xf32>
// CHECK: "ttir.reshape"({{%.*}}) <{shape = [1 : i32, 512 : i32]}>
// CHECK-SAME: (tensor<1x128x4xf32>) -> tensor<1x512xf32>
func.func @singleton_permute_trailing_to_leading(%arg0: tensor<128x4x1xf32>) -> tensor<1x512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<128x4x1xf32>) -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Test: [64, 8, 1] -> [1, 512]
// Same pattern as above
// CHECK-LABEL: @singleton_permute_trailing_to_leading_2
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 1>}>
// CHECK-SAME: (tensor<64x8x1xf32>) -> tensor<1x64x8xf32>
// CHECK: "ttir.reshape"({{%.*}}) <{shape = [1 : i32, 512 : i32]}>
// CHECK-SAME: (tensor<1x64x8xf32>) -> tensor<1x512xf32>
func.func @singleton_permute_trailing_to_leading_2(%arg0: tensor<64x8x1xf32>) -> tensor<1x512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<64x8x1xf32>) -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Test: [32, 1, 1] -> [1, 1, 32]
// Two trailing 1s need to become leading 1s
// Same rank, so just permute
// CHECK-LABEL: @singleton_permute_two_trailing_ones
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 2, 0>}>
// CHECK-SAME: (tensor<32x1x1xf32>) -> tensor<1x1x32xf32>
// CHECK-NOT: "ttir.reshape"
func.func @singleton_permute_two_trailing_ones(%arg0: tensor<32x1x1xf32>) -> tensor<1x1x32xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 32 : i32]}> : (tensor<32x1x1xf32>) -> tensor<1x1x32xf32>
  return %0 : tensor<1x1x32xf32>
}

// ============================================================================
// Cases that should NOT be transformed - true reshapes that change data layout
// ============================================================================

// Test: [2, 32] -> [32, 2]
// Non-1 dims: [2, 32] vs [32, 2] - DIFFERENT order (true reshape, not permute)
// CHECK-LABEL: @no_transform_true_reshape
// CHECK: "ttir.reshape"(%arg0) <{shape = [32 : i32, 2 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_true_reshape(%arg0: tensor<2x32xf32>) -> tensor<32x2xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [32 : i32, 2 : i32]}> : (tensor<2x32xf32>) -> tensor<32x2xf32>
  return %0 : tensor<32x2xf32>
}

// Test: [2, 3, 4] -> [24]
// Collapsing all dims (true reshape)
// CHECK-LABEL: @no_transform_collapse_all
// CHECK: "ttir.reshape"(%arg0) <{shape = [24 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_collapse_all(%arg0: tensor<2x3x4xf32>) -> tensor<24xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [24 : i32]}> : (tensor<2x3x4xf32>) -> tensor<24xf32>
  return %0 : tensor<24xf32>
}

// Test: [24] -> [2, 3, 4]
// Expanding to multiple dims (true reshape)
// CHECK-LABEL: @no_transform_expand_all
// CHECK: "ttir.reshape"(%arg0) <{shape = [2 : i32, 3 : i32, 4 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_expand_all(%arg0: tensor<24xf32>) -> tensor<2x3x4xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 3 : i32, 4 : i32]}> : (tensor<24xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>
}

// Test: [1, 128, 4] -> [1, 512]
// Leading 1 stays leading (just a reshape, no permute needed)
// CHECK-LABEL: @no_transform_leading_one_preserved
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_leading_one_preserved(%arg0: tensor<1x128x4xf32>) -> tensor<1x512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<1x128x4xf32>) -> tensor<1x512xf32>
  return %0 : tensor<1x512xf32>
}

// Test: [128, 4] -> [512]
// No singletons involved (true reshape)
// CHECK-LABEL: @no_transform_no_singletons
// CHECK: "ttir.reshape"(%arg0) <{shape = [512 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_no_singletons(%arg0: tensor<128x4xf32>) -> tensor<512xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [512 : i32]}> : (tensor<128x4xf32>) -> tensor<512xf32>
  return %0 : tensor<512xf32>
}

// Test: [1, 32, 1] -> [1, 32, 1]
// Identity reshape - should not be transformed
// CHECK-LABEL: @no_transform_identity
// CHECK: "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32, 1 : i32]}>
// CHECK-NOT: "ttir.permute"
func.func @no_transform_identity(%arg0: tensor<1x32x1xf32>) -> tensor<1x32x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32, 1 : i32]}> : (tensor<1x32x1xf32>) -> tensor<1x32x1xf32>
  return %0 : tensor<1x32x1xf32>
}

// ============================================================================
// Higher-dimensional tests
// ============================================================================

// Test: [1, 2, 3, 1] -> [2, 1, 3, 1]
// Swap leading 1 with next dim (same rank -> just permute)
// CHECK-LABEL: @pure_permute_4d_swap
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2, 3>}>
// CHECK-SAME: (tensor<1x2x3x1xf32>) -> tensor<2x1x3x1xf32>
// CHECK-NOT: "ttir.reshape"
func.func @pure_permute_4d_swap(%arg0: tensor<1x2x3x1xf32>) -> tensor<2x1x3x1xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 1 : i32, 3 : i32, 1 : i32]}> : (tensor<1x2x3x1xf32>) -> tensor<2x1x3x1xf32>
  return %0 : tensor<2x1x3x1xf32>
}

// Test: [4, 8, 1, 1] -> [1, 1, 4, 8]
// Multiple trailing 1s move to leading (same rank -> just permute)
// CHECK-LABEL: @singleton_permute_4d_trailing_to_leading
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 2, 3, 0, 1>}>
// CHECK-SAME: (tensor<4x8x1x1xf32>) -> tensor<1x1x4x8xf32>
// CHECK-NOT: "ttir.reshape"
func.func @singleton_permute_4d_trailing_to_leading(%arg0: tensor<4x8x1x1xf32>) -> tensor<1x1x4x8xf32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 4 : i32, 8 : i32]}> : (tensor<4x8x1x1xf32>) -> tensor<1x1x4x8xf32>
  return %0 : tensor<1x1x4x8xf32>
}

// ============================================================================
// Tests with different element types
// ============================================================================

// Test with bf16: [32, 1] -> [1, 32]
// CHECK-LABEL: @pure_permute_bf16
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0>}>
// CHECK-SAME: (tensor<32x1xbf16>) -> tensor<1x32xbf16>
// CHECK-NOT: "ttir.reshape"
func.func @pure_permute_bf16(%arg0: tensor<32x1xbf16>) -> tensor<1x32xbf16> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 32 : i32]}> : (tensor<32x1xbf16>) -> tensor<1x32xbf16>
  return %0 : tensor<1x32xbf16>
}

// Test with i32: [128, 4, 1] -> [1, 512]
// CHECK-LABEL: @singleton_permute_i32
// CHECK: "ttir.permute"(%arg0) <{permutation = array<i64: 2, 0, 1>}>
// CHECK-SAME: (tensor<128x4x1xi32>) -> tensor<1x128x4xi32>
// CHECK: "ttir.reshape"({{%.*}}) <{shape = [1 : i32, 512 : i32]}>
// CHECK-SAME: (tensor<1x128x4xi32>) -> tensor<1x512xi32>
func.func @singleton_permute_i32(%arg0: tensor<128x4x1xi32>) -> tensor<1x512xi32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 512 : i32]}> : (tensor<128x4x1xi32>) -> tensor<1x512xi32>
  return %0 : tensor<1x512xi32>
}
