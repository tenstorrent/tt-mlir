// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @eq_with_full_rhs
func.func @eq_with_full_rhs(%arg0: tensor<32x32xf32>) -> tensor<32x32xi1> {
  // CHECK: ttir.equal_scalar
  // CHECK-NOT: ttir.eq
  %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 2.0 : f32}> : () -> tensor<32x32xf32>
  %1 = "ttir.eq"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xi1>
  return %1 : tensor<32x32xi1>
}

// CHECK-LABEL: @eq_with_full_lhs
func.func @eq_with_full_lhs(%arg0: tensor<32x32xf32>) -> tensor<32x32xi1> {
  // CHECK: ttir.equal_scalar
  // CHECK-NOT: ttir.eq
  %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 3.0 : f32}> : () -> tensor<32x32xf32>
  %1 = "ttir.eq"(%0, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xi1>
  return %1 : tensor<32x32xi1>
}

// CHECK-LABEL: @eq_with_zeros
func.func @eq_with_zeros(%arg0: tensor<32x32xf32>) -> tensor<32x32xi1> {
  // CHECK: ttir.equal_scalar
  // CHECK-NOT: ttir.eq
  %0 = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xf32>
  %1 = "ttir.eq"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xi1>
  return %1 : tensor<32x32xi1>
}

// CHECK-LABEL: @eq_with_ones
func.func @eq_with_ones(%arg0: tensor<32x32xf32>) -> tensor<32x32xi1> {
  // CHECK: ttir.equal_scalar
  // CHECK-NOT: ttir.eq
  %0 = "ttir.ones"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xf32>
  %1 = "ttir.eq"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xi1>
  return %1 : tensor<32x32xi1>
}

// Test that the pattern does NOT apply when both operands are tensors (not constants)
// CHECK-LABEL: @eq_tensor_tensor
func.func @eq_tensor_tensor(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xi1> {
  // CHECK: ttir.eq
  // CHECK-NOT: ttir.equal_scalar
  %1 = "ttir.eq"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xi1>
  return %1 : tensor<32x32xi1>
}

// Test that the pattern DOES apply even when the full op has multiple uses
// (canonicalization converts to scalar op regardless of other uses)
// CHECK-LABEL: @eq_full_multiple_uses
func.func @eq_full_multiple_uses(%arg0: tensor<32x32xf32>) -> (tensor<32x32xi1>, tensor<32x32xf32>) {
  // CHECK: ttir.full
  // CHECK: ttir.equal_scalar
  // CHECK-NOT: ttir.eq
  %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 2.0 : f32}> : () -> tensor<32x32xf32>
  %1 = "ttir.eq"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xi1>
  return %1, %0 : tensor<32x32xi1>, tensor<32x32xf32>
}
