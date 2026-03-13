// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 2: Unary tensor-manipulation ops (reshape, broadcast) get result
// element type set to input element type. Propagation stops at typecast.
// =============================================================================

// -----
// Reshape: result element type propagates from input (i1 -> becomes i32 in phase 3;
// here we test that reshape output is updated to match input when input is i32).
// CHECK-LABEL: func.func @phase2_reshape_result_matches_input
// CHECK: "ttir.reshape"(%arg0) <{shape = [8 : i32, 4 : i32]}> : (tensor<4x8xi32>) -> tensor<8x4xi32>
func.func @phase2_reshape_result_matches_input(%arg0: tensor<4x8xi32>) -> tensor<8x4xi32> {
  %0 = "ttir.reshape"(%arg0) <{shape = [8 : i32, 4 : i32]}> : (tensor<4x8xi32>) -> tensor<8x4xi32>
  return %0 : tensor<8x4xi32>
}

// -----
// Broadcast: result element type set to input type.
// CHECK-LABEL: func.func @phase2_broadcast_result_matches_input
// CHECK: "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 16>}> : (tensor<1x1xi32>) -> tensor<16x16xi32>
func.func @phase2_broadcast_result_matches_input(%arg0: tensor<1x1xi32>) -> tensor<16x16xi32> {
  %0 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 16, 16>}> : (tensor<1x1xi32>) -> tensor<16x16xi32>
  return %0 : tensor<16x16xi32>
}
