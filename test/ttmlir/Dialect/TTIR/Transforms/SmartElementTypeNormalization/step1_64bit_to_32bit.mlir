// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-smart-element-type-normalization %s | FileCheck %s

// =============================================================================
// Step 1: i64 and f64 tensors and constants are converted to i32 and f32.
// =============================================================================

// -----
// CHECK-LABEL: func.func @phase1_constant_i64_to_i32
// CHECK: "ttir.constant"() <{value = dense<42> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
// CHECK: return
func.func @phase1_constant_i64_to_i32() -> tensor<2x2xi64> {
  %0 = "ttir.constant"() <{value = dense<42> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
  return %0 : tensor<2x2xi64>
}

// -----
// CHECK-LABEL: func.func @phase1_constant_f64_to_f32
// CHECK: "ttir.constant"() <{value = dense<1.500000e+00> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>
// CHECK: return
func.func @phase1_constant_f64_to_f32() -> tensor<2x2xf64> {
  %0 = "ttir.constant"() <{value = dense<1.5> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  return %0 : tensor<2x2xf64>
}

// -----
// CHECK-LABEL: func.func @phase1_arg_and_ops_i64_f64
// CHECK-SAME: (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xf32>)
// CHECK: "ttir.add"(%arg0, %{{.*}}) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.add"(%arg1, %{{.*}}) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
func.func @phase1_arg_and_ops_i64_f64(%arg0: tensor<4x4xi64>, %arg1: tensor<4x4xf64>) -> (tensor<4x4xi64>, tensor<4x4xf64>) {
  %cst = "ttir.constant"() <{value = dense<1> : tensor<4x4xi64>}> : () -> tensor<4x4xi64>
  %cst_0 = "ttir.constant"() <{value = dense<1.0> : tensor<4x4xf64>}> : () -> tensor<4x4xf64>
  %0 = "ttir.add"(%arg0, %cst) : (tensor<4x4xi64>, tensor<4x4xi64>) -> tensor<4x4xi64>
  %1 = "ttir.add"(%arg1, %cst_0) : (tensor<4x4xf64>, tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0, %1 : tensor<4x4xi64>, tensor<4x4xf64>
}

// -----
// CHECK-LABEL: func.func @phase1_i32_f32_unchanged
// CHECK: "ttir.constant"() <{value = dense<7> : tensor<8x8xi32>}> : () -> tensor<8x8xi32>
// CHECK: return %0 : tensor<8x8xi32>
func.func @phase1_i32_f32_unchanged() -> tensor<8x8xi32> {
  %0 = "ttir.constant"() <{value = dense<7> : tensor<8x8xi32>}> : () -> tensor<8x8xi32>
  return %0 : tensor<8x8xi32>
}

// -----
// Step 1 edge case: clamp_scalar with i64 input. Attrs stay i32 while input and
// result are normalized to i32 by type conversion.
// CHECK-LABEL: func.func @step1_clamp_scalar_i64_to_i32
// CHECK-SAME: (%arg0: tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.clamp_scalar"(%arg0) <{max = 3 : i32, min = 0 : i32}> : (tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return
func.func @step1_clamp_scalar_i64_to_i32(%arg0: tensor<4x4xi64>) -> tensor<4x4xi64> {
  %0 = "ttir.clamp_scalar"(%arg0) <{max = 3 : i32, min = 0 : i32}> : (tensor<4x4xi64>) -> tensor<4x4xi64>
  return %0 : tensor<4x4xi64>
}
