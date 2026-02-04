// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @pow_with_full_exponent
func.func @pow_with_full_exponent(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ttir.pow_scalar
  // CHECK-NOT: ttir.pow
  %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 2.0 : f32}> : () -> tensor<32x32xf32>
  %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: @pow_with_zeros_exponent
func.func @pow_with_zeros_exponent(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ttir.pow_scalar
  // CHECK-NOT: ttir.pow
  %0 = "ttir.zeros"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xf32>
  %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// CHECK-LABEL: @pow_with_ones_exponent
func.func @pow_with_ones_exponent(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ttir.pow_scalar
  // CHECK-NOT: ttir.pow
  %0 = "ttir.ones"() <{shape = array<i32: 32, 32>}> : () -> tensor<32x32xf32>
  %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Test that the pattern does NOT apply when both operands are tensors (not constants)
// CHECK-LABEL: @pow_tensor_tensor
func.func @pow_tensor_tensor(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ttir.pow
  // CHECK-NOT: ttir.pow_scalar
  %1 = "ttir.pow"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Test that the pattern does NOT apply when base is a constant
// (only scalar exponent is supported, not scalar base)
// CHECK-LABEL: @pow_full_base
func.func @pow_full_base(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ttir.pow
  // CHECK-NOT: ttir.pow_scalar
  %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 2.0 : f32}> : () -> tensor<32x32xf32>
  %1 = "ttir.pow"(%0, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}

// Test that the pattern DOES apply even when the full op has multiple uses
// (canonicalization converts to scalar op regardless of other uses)
// CHECK-LABEL: @pow_full_multiple_uses
func.func @pow_full_multiple_uses(%arg0: tensor<32x32xf32>) -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  // CHECK: ttir.full
  // CHECK: ttir.pow_scalar
  // CHECK-NOT: ttir.pow{{[^_]}}
  %0 = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 2.0 : f32}> : () -> tensor<32x32xf32>
  %1 = "ttir.pow"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1, %0 : tensor<32x32xf32>, tensor<32x32xf32>
}
