// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 2: where(condition, true_val, false_val) — condition tensor is
// typecast so its element type matches the true/false value type.
// =============================================================================

// -----
// Step 2 inserts typecast(condition) to match value type. Step 3 converts i1
// (including function args) to i32, so the condition arg becomes i32.
// CHECK-LABEL: func.func @phase2_where_condition_cast_to_i32
// CHECK: "ttir.typecast"(%arg0) {{.*}} : (tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.where"(%{{.*}}, %arg1, %arg2) : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_where_condition_cast_to_i32(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi32>, %arg2: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<4x4xi1>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_where_condition_cast_to_f32
// CHECK: "ttir.typecast"(%arg0) {{.*}} : (tensor<2x8xi32>) -> tensor<2x8xf32>
// CHECK: "ttir.where"(%{{.*}}, %arg1, %arg2) : (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
func.func @phase2_where_condition_cast_to_f32(%arg0: tensor<2x8xi1>, %arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
