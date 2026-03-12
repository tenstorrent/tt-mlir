// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 2: Comparison ops (eq, ne, gt, ge, lt, le) with i1 result get result
// type changed to match lhs element type.
// =============================================================================

// -----
// CHECK-LABEL: func.func @phase2_eq_result_i32
// CHECK: "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_eq_result_i32(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi1> {
  %0 = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// -----
// Step 2 changes comparison result to lhs type; use i32 so step 3 (i1->i32) keeps
// function return type and value type consistent.
// CHECK-LABEL: func.func @phase2_ne_result_i32
// CHECK: "ttir.ne"(%arg0, %arg1) : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
func.func @phase2_ne_result_i32(%arg0: tensor<2x8xi32>, %arg1: tensor<2x8xi32>) -> tensor<2x8xi1> {
  %0 = "ttir.ne"(%arg0, %arg1) : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
  return %0 : tensor<2x8xi1>
}

// -----
// CHECK-LABEL: func.func @phase2_gt_ge_lt_le_result_type
// CHECK: "ttir.gt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
// CHECK: "ttir.ge"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
// CHECK: "ttir.lt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
// CHECK: "ttir.le"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
func.func @phase2_gt_ge_lt_le_result_type(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>) -> (tensor<3x3xi1>, tensor<3x3xi1>, tensor<3x3xi1>, tensor<3x3xi1>) {
  %a = "ttir.gt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %b = "ttir.ge"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %c = "ttir.lt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %d = "ttir.le"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  return %a, %b, %c, %d : tensor<3x3xi1>, tensor<3x3xi1>, tensor<3x3xi1>, tensor<3x3xi1>
}
