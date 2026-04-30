// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// Folds: scalar constant -> reshape -> broadcast -> shift, the exact chain
// produced by the StableHLO->TTIR conversion of JAX's iota_2x32_shape.
// CHECK-LABEL: @fold_jax_threefry_pattern
func.func @fold_jax_threefry_pattern(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %c = "ttir.constant"() <{value = dense<32> : tensor<i32>}> : () -> tensor<i32>
  %r = "ttir.reshape"(%c) <{shape = [1 : i32]}> : (tensor<i32>) -> tensor<1xi32>
  %b = "ttir.broadcast"(%r) <{broadcast_dimensions = array<i64: 2>}> : (tensor<1xi32>) -> tensor<2xi32>
  // CHECK-NOT: ttir.logical_right_shift
  // CHECK: ttir.constant
  // CHECK-SAME: dense<0>
  %0 = "ttir.logical_right_shift"(%arg0, %b) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// Folds: rank-2 1x10 result via reshape (rank-0 -> 1x1) + broadcast (1x1 -> 1x10).
// Mirrors the iota_2x32_shape lowering for `jax.random.normal((1, 10))`.
// CHECK-LABEL: @fold_2d_via_reshape_broadcast
func.func @fold_2d_via_reshape_broadcast(%arg0: tensor<1x10xui32>) -> tensor<1x10xui32> {
  %c = "ttir.constant"() <{value = dense<32> : tensor<ui32>}> : () -> tensor<ui32>
  %r = "ttir.reshape"(%c) <{shape = [1 : i32, 1 : i32]}> : (tensor<ui32>) -> tensor<1x1xui32>
  %b = "ttir.broadcast"(%r) <{broadcast_dimensions = array<i64: 1, 10>}> : (tensor<1x1xui32>) -> tensor<1x10xui32>
  // CHECK-NOT: ttir.logical_right_shift
  // CHECK: ttir.constant
  // CHECK-SAME: dense<0>
  %0 = "ttir.logical_right_shift"(%arg0, %b) : (tensor<1x10xui32>, tensor<1x10xui32>) -> tensor<1x10xui32>
  return %0 : tensor<1x10xui32>
}

// Folds: shift amount = 63 (still >= 32) via reshape -> broadcast.
// CHECK-LABEL: @fold_by_63_via_reshape_broadcast
func.func @fold_by_63_via_reshape_broadcast(%arg0: tensor<4xi32>) -> tensor<4xi32> {
  %c = "ttir.constant"() <{value = dense<63> : tensor<i32>}> : () -> tensor<i32>
  %r = "ttir.reshape"(%c) <{shape = [1 : i32]}> : (tensor<i32>) -> tensor<1xi32>
  %b = "ttir.broadcast"(%r) <{broadcast_dimensions = array<i64: 4>}> : (tensor<1xi32>) -> tensor<4xi32>
  // CHECK-NOT: ttir.logical_right_shift
  // CHECK: ttir.constant
  // CHECK-SAME: dense<0>
  %0 = "ttir.logical_right_shift"(%arg0, %b) : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
  return %0 : tensor<4xi32>
}

// Does NOT fold: shift amount < 32 even via the reshape -> broadcast chain.
// CHECK-LABEL: @no_fold_below_32_via_reshape_broadcast
func.func @no_fold_below_32_via_reshape_broadcast(%arg0: tensor<2xi32>) -> tensor<2xi32> {
  %c = "ttir.constant"() <{value = dense<31> : tensor<i32>}> : () -> tensor<i32>
  %r = "ttir.reshape"(%c) <{shape = [1 : i32]}> : (tensor<i32>) -> tensor<1xi32>
  %b = "ttir.broadcast"(%r) <{broadcast_dimensions = array<i64: 2>}> : (tensor<1xi32>) -> tensor<2xi32>
  // CHECK: ttir.logical_right_shift
  %0 = "ttir.logical_right_shift"(%arg0, %b) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// Does NOT fold: non-constant shift amount.
// CHECK-LABEL: @no_fold_dynamic
func.func @no_fold_dynamic(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> tensor<2xi32> {
  // CHECK: ttir.logical_right_shift
  %0 = "ttir.logical_right_shift"(%arg0, %arg1) : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}
