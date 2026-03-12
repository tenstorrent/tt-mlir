// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Combined: All steps applied in one function — 64-bit, comparisons, where,
// logical_and mix, then remaining i1 to i32.
// =============================================================================

// -----
// CHECK-LABEL: func.func @combined_all_steps
// Step 1: i64 -> i32 (f64 constant unused in this function)
// CHECK: "ttir.constant"() <{value = dense<1> : tensor<2x2xi32>}> : () -> tensor<2x2xi32>
// Step 2: eq result i1 -> i32, where condition cast to i32 (after step 3 no i1
// remains, so condition typecast may be identity i32 -> i32).
// CHECK: "ttir.eq"(%{{.*}}, %{{.*}}) : (tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK: "ttir.typecast"(%{{.*}})
// CHECK: "ttir.where"(%{{.*}}, %{{.*}}, %{{.*}}) : (tensor<2x2xi32>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
// CHECK: return
func.func @combined_all_steps(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xf64>, %arg2: tensor<2x2xi1>) -> tensor<2x2xi32> {
  %c64 = "ttir.constant"() <{value = dense<1> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
  %cf64 = "ttir.constant"() <{value = dense<1.0> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %eq = "ttir.eq"(%arg0, %c64) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi1>
  %sel = "ttir.where"(%eq, %arg0, %c64) : (tensor<2x2xi1>, tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
  %cast = "ttir.typecast"(%sel) : (tensor<2x2xi64>) -> tensor<2x2xi32>
  return %cast : tensor<2x2xi32>
}

// -----
// Combined: comparison -> logical_and(i1, i32) -> step 3 i1 constant.
// CHECK-LABEL: func.func @combined_compare_and_then_i1_const
// Order in output: constant then eq (SSA renumbered).
// CHECK: "ttir.constant"() <{value = dense<1> : tensor<4x4xi32>}> : () -> tensor<4x4xi32>
// CHECK: "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.logical_and"(%{{.*}}, %{{.*}}) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return
func.func @combined_compare_and_then_i1_const(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %eq = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
  %one = "ttir.constant"() <{value = dense<true> : tensor<4x4xi1>}> : () -> tensor<4x4xi1>
  %and = "ttir.logical_and"(%eq, %one) : (tensor<4x4xi1>, tensor<4x4xi1>) -> tensor<4x4xi1>
  %cast = "ttir.typecast"(%and) : (tensor<4x4xi1>) -> tensor<4x4xi32>
  return %cast : tensor<4x4xi32>
}
