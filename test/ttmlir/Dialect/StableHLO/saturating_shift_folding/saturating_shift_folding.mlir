// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-saturating-shift-folding %s | FileCheck %s

// Folds: ui64 shift_right_logical by splat-32.
// CHECK-LABEL: @fold_ui64_by_32
func.func @fold_ui64_by_32(%arg0: tensor<2xui64>) -> tensor<2xui64> {
  %c = stablehlo.constant dense<32> : tensor<2xui64>
  // CHECK-NOT: stablehlo.shift_right_logical
  // CHECK: stablehlo.constant dense<0> : tensor<2xui64>
  %0 = stablehlo.shift_right_logical %arg0, %c : tensor<2xui64>
  return %0 : tensor<2xui64>
}

// Folds: i64 (signed/signless) shift_right_logical by splat-63.
// CHECK-LABEL: @fold_i64_by_63
func.func @fold_i64_by_63(%arg0: tensor<4xi64>) -> tensor<4xi64> {
  %c = stablehlo.constant dense<63> : tensor<4xi64>
  // CHECK-NOT: stablehlo.shift_right_logical
  // CHECK: stablehlo.constant dense<0> : tensor<4xi64>
  %0 = stablehlo.shift_right_logical %arg0, %c : tensor<4xi64>
  return %0 : tensor<4xi64>
}

// Folds: ui64 shift_right_logical by scalar-32 broadcast via broadcast_in_dim.
// CHECK-LABEL: @fold_ui64_by_32_broadcast
func.func @fold_ui64_by_32_broadcast(%arg0: tensor<2xui64>) -> tensor<2xui64> {
  %c = stablehlo.constant dense<32> : tensor<ui64>
  %bcast = stablehlo.broadcast_in_dim %c, dims = [] : (tensor<ui64>) -> tensor<2xui64>
  // CHECK-NOT: stablehlo.shift_right_logical
  // CHECK: stablehlo.constant dense<0> : tensor<2xui64>
  %0 = stablehlo.shift_right_logical %arg0, %bcast : tensor<2xui64>
  return %0 : tensor<2xui64>
}

// Does NOT fold: shift amount < 32.
// CHECK-LABEL: @no_fold_below_32
func.func @no_fold_below_32(%arg0: tensor<2xui64>) -> tensor<2xui64> {
  %c = stablehlo.constant dense<31> : tensor<2xui64>
  // CHECK: stablehlo.shift_right_logical
  %0 = stablehlo.shift_right_logical %arg0, %c : tensor<2xui64>
  return %0 : tensor<2xui64>
}

// Does NOT fold: result is 32-bit, not 64-bit.
// CHECK-LABEL: @no_fold_ui32
func.func @no_fold_ui32(%arg0: tensor<2xui32>) -> tensor<2xui32> {
  %c = stablehlo.constant dense<31> : tensor<2xui32>
  // CHECK: stablehlo.shift_right_logical
  %0 = stablehlo.shift_right_logical %arg0, %c : tensor<2xui32>
  return %0 : tensor<2xui32>
}

// Does NOT fold: shift amount is non-constant.
// CHECK-LABEL: @no_fold_non_constant
func.func @no_fold_non_constant(%arg0: tensor<2xui64>, %arg1: tensor<2xui64>) -> tensor<2xui64> {
  // CHECK: stablehlo.shift_right_logical
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<2xui64>
  return %0 : tensor<2xui64>
}
