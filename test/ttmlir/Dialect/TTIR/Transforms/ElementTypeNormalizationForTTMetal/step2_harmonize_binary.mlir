// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 2: logical_and(i1, i32) (and similar) — operands and result harmonized
// to the non-i1 type (i32) by inserting typecasts. Step 3 then converts
// remaining i1 to i32, so the final output has i32 everywhere; typecasts
// may appear as identity (i32 -> i32).
// =============================================================================

// -----
// CHECK-LABEL: func.func @phase2_logical_and_i1_i32_harmonize
// CHECK-SAME: (%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.typecast"(%arg0)
// CHECK: "ttir.logical_and"(%{{.*}}, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_logical_and_i1_i32_harmonize(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi1> {
  %0 = "ttir.logical_and"(%arg0, %arg1) : (tensor<4x4xi1>, tensor<4x4xi32>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// -----
// CHECK-LABEL: func.func @phase2_logical_or_i32_i1_harmonize
// CHECK-SAME: (%arg0: tensor<2x8xi32>, %arg1: tensor<2x8xi32>) -> tensor<2x8xi32>
// CHECK: "ttir.typecast"(%arg1)
// CHECK: "ttir.logical_or"(%arg0, %{{.*}}) : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
func.func @phase2_logical_or_i32_i1_harmonize(%arg0: tensor<2x8xi32>, %arg1: tensor<2x8xi1>) -> tensor<2x8xi1> {
  %0 = "ttir.logical_or"(%arg0, %arg1) : (tensor<2x8xi32>, tensor<2x8xi1>) -> tensor<2x8xi1>
  return %0 : tensor<2x8xi1>
}
