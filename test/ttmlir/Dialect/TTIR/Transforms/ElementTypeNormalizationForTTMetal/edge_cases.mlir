// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Edge case: Comparison with non-i1 result is left unchanged (no rewrite).
// =============================================================================

// -----
// CHECK-LABEL: func.func @edge_comparison_already_non_i1
// CHECK: "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return
func.func @edge_comparison_already_non_i1(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
  %0 = "ttir.eq"(%arg0, %arg1) : (tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
  return %0 : tensor<4x4xi32>
}

// -----
// Edge case: Already normalized graph is unchanged (idempotent).
// CHECK-LABEL: func.func @edge_idempotent
// CHECK: tensor<8x8xi32>
// CHECK: tensor<8x8xf32>
// CHECK: return
func.func @edge_idempotent() -> (tensor<8x8xi32>, tensor<8x8xf32>) {
  %c32 = "ttir.constant"() <{value = dense<0> : tensor<8x8xi32>}> : () -> tensor<8x8xi32>
  %cf32 = "ttir.constant"() <{value = dense<0.0> : tensor<8x8xf32>}> : () -> tensor<8x8xf32>
  return %c32, %cf32 : tensor<8x8xi32>, tensor<8x8xf32>
}

// -----
// Edge case: Unary propagation stops at typecast (reshape after typecast
// keeps its result type; we only check that the pass runs without error).
// CHECK-LABEL: func.func @edge_typecast_boundary
// CHECK: "ttir.typecast"(%arg0){{.*}}: (tensor<4x4xi32>) -> tensor<4x4xf32>
// CHECK: "ttir.reshape"(%{{.*}}) <{shape = [2 : i32, 8 : i32]}> : (tensor<4x4xf32>) -> tensor<2x8xf32>
func.func @edge_typecast_boundary(%arg0: tensor<4x4xi32>) -> tensor<2x8xf32> {
  %cast = "ttir.typecast"(%arg0) : (tensor<4x4xi32>) -> tensor<4x4xf32>
  %0 = "ttir.reshape"(%cast) <{shape = [2 : i32, 8 : i32]}> : (tensor<4x4xf32>) -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
