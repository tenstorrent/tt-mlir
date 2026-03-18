// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-predicate-type-alignment %s | FileCheck %s

// =============================================================================
// ttir-predicate-type-alignment: comparisons, logical_not, reduce_or, where,
// mixed i1/non-i1 elementwise binaries, unary tensor-manip + gather.
// =============================================================================

// -----
// CHECK-LABEL: func.func @phase2_eq_result_i32
// CHECK: "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_eq_result_i32(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi1>
  %1 = "ttir.typecast"(%0) : (tensor<4x4xi1>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_ne_result_i32
// CHECK: "ttir.ne"(%arg0, %arg1) : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
func.func @phase2_ne_result_i32(%arg0: tensor<2x8xi32>, %arg1: tensor<2x8xi32>) -> tensor<2x8xi32> {
  %0 = "ttir.ne"(%arg0, %arg1) : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi1>
  %1 = "ttir.typecast"(%0) : (tensor<2x8xi1>) -> tensor<2x8xi32>
  return %1 : tensor<2x8xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_gt_ge_lt_le_result_type
// CHECK: "ttir.gt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
// CHECK: "ttir.ge"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
// CHECK: "ttir.lt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
// CHECK: "ttir.le"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi32>
func.func @phase2_gt_ge_lt_le_result_type(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>) -> (tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>) {
  %a = "ttir.gt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %b = "ttir.ge"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %c = "ttir.lt"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %d = "ttir.le"(%arg0, %arg1) : (tensor<3x3xi32>, tensor<3x3xi32>) -> tensor<3x3xi1>
  %a_i32 = "ttir.typecast"(%a) : (tensor<3x3xi1>) -> tensor<3x3xi32>
  %b_i32 = "ttir.typecast"(%b) : (tensor<3x3xi1>) -> tensor<3x3xi32>
  %c_i32 = "ttir.typecast"(%c) : (tensor<3x3xi1>) -> tensor<3x3xi32>
  %d_i32 = "ttir.typecast"(%d) : (tensor<3x3xi1>) -> tensor<3x3xi32>
  return %a_i32, %b_i32, %c_i32, %d_i32 : tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>, tensor<3x3xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_logical_not_result_i32
// CHECK: "ttir.logical_not"(%arg0) : (tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_logical_not_result_i32(%arg0: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.logical_not"(%arg0) : (tensor<4x4xi32>) -> tensor<4x4xi1>
  %1 = "ttir.typecast"(%0) : (tensor<4x4xi1>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_reduce_or_result_i32
// CHECK: "ttir.reduce_or"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<1x8xi32>
func.func @phase2_reduce_or_result_i32(%arg0: tensor<4x8xi32>) -> tensor<1x8xi32> {
  %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<1x8xi1>
  %1 = "ttir.typecast"(%0) : (tensor<1x8xi1>) -> tensor<1x8xi32>
  return %1 : tensor<1x8xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_where_condition_cast_to_i32
// CHECK-SAME: (%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi32>, %arg2: tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.typecast"(%arg0) {{.*}} : (tensor<4x4xi1>) -> tensor<4x4xi32>
// CHECK: "ttir.where"(%{{.*}}, %arg1, %arg2) : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_where_condition_cast_to_i32(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi32>, %arg2: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<4x4xi1>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_where_condition_cast_to_f32
// CHECK-SAME: (%arg0: tensor<2x8xi1>, %arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK: "ttir.typecast"(%arg0) {{.*}} : (tensor<2x8xi1>) -> tensor<2x8xf32>
// CHECK: "ttir.where"(%{{.*}}, %arg1, %arg2) : (tensor<2x8xf32>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
func.func @phase2_where_condition_cast_to_f32(%arg0: tensor<2x8xi1>, %arg1: tensor<2x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = "ttir.where"(%arg0, %arg1, %arg2) : (tensor<2x8xi1>, tensor<2x8xf32>, tensor<2x8xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}

// -----
// CHECK-LABEL: func.func @phase2_logical_and_i1_i32_align
// CHECK-SAME: (%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.typecast"(%arg0) {{.*}} : (tensor<4x4xi1>) -> tensor<4x4xi32>
// CHECK: "ttir.logical_and"(%{{.*}}, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_logical_and_i1_i32_align(%arg0: tensor<4x4xi1>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.logical_and"(%arg0, %arg1) : (tensor<4x4xi1>, tensor<4x4xi32>) -> tensor<4x4xi1>
  %1 = "ttir.typecast"(%0) : (tensor<4x4xi1>) -> tensor<4x4xi32>
  return %1 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_logical_or_i32_i1_align
// CHECK-SAME: (%arg0: tensor<2x8xi32>, %arg1: tensor<2x8xi1>) -> tensor<2x8xi32>
// CHECK: "ttir.typecast"(%arg1) {{.*}} : (tensor<2x8xi1>) -> tensor<2x8xi32>
// CHECK: "ttir.logical_or"(%arg0, %{{.*}}) : (tensor<2x8xi32>, tensor<2x8xi32>) -> tensor<2x8xi32>
func.func @phase2_logical_or_i32_i1_align(%arg0: tensor<2x8xi32>, %arg1: tensor<2x8xi1>) -> tensor<2x8xi32> {
  %0 = "ttir.logical_or"(%arg0, %arg1) : (tensor<2x8xi32>, tensor<2x8xi1>) -> tensor<2x8xi1>
  %1 = "ttir.typecast"(%0) : (tensor<2x8xi1>) -> tensor<2x8xi32>
  return %1 : tensor<2x8xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_reshape_result_matches_input
// CHECK: "ttir.reshape"(%arg0) <{shape = [8 : i32, 4 : i32]}> : (tensor<4x8xi32>) -> tensor<8x4xi32>
func.func @phase2_reshape_result_matches_input(%arg0: tensor<4x8xi32>) -> tensor<8x4xi32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [8 : i32, 4 : i32]}> : (tensor<4x8xi32>) -> tensor<8x4xi32>
  return %0 : tensor<8x4xi32>
}

// -----
// CHECK-LABEL: func.func @phase2_broadcast_result_matches_input
// CHECK: "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 16>}> : (tensor<1x1xi32>) -> tensor<16x16xi32>
func.func @phase2_broadcast_result_matches_input(%arg0: tensor<1x1xi32>) -> tensor<16x16xi32> {
  %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 16>}> : (tensor<1x1xi32>) -> tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}

// -----
// CHECK-LABEL: func.func @edge_comparison_already_non_i1
// CHECK: "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return
func.func @edge_comparison_already_non_i1(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----
// CHECK-LABEL: func.func @edge_typecast_boundary
// CHECK: "ttir.typecast"(%arg0){{.*}}: (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK: "ttir.reshape"(%{{.*}}) <{shape = [2 : i32, 8 : i32]}> : (tensor<4x4xf32>) -> tensor<2x8xf32>
func.func @edge_typecast_boundary(%arg0: tensor<4x4xi32>) -> tensor<2x8xf32> {
  %cast = "ttir.typecast"(%arg0) : (tensor<4x4xi32>) -> tensor<4x4xf32>
  %0 = "ttir.reshape"(%cast) <{shape = [2 : i32, 8 : i32]}> : (tensor<4x4xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
