// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 2: logical_not and reduce_or with i1 result get result type changed to
// match input element type.
// =============================================================================

// -----
// CHECK-LABEL: func.func @phase2_logical_not_result_i32
// CHECK: "ttir.logical_not"(%arg0) : (tensor<4x4xi32>) -> tensor<4x4xi32>
func.func @phase2_logical_not_result_i32(%arg0: tensor<4x4xi32>) -> tensor<4x4xi1> {
  %0 = "ttir.logical_not"(%arg0) : (tensor<4x4xi32>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// -----
// CHECK-LABEL: func.func @phase2_reduce_or_result_i32
// CHECK: "ttir.reduce_or"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<1x8xi32>
func.func @phase2_reduce_or_result_i32(%arg0: tensor<4x8xi32>) -> tensor<1x8xi1> {
  %0 = "ttir.reduce_or"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xi32>) -> tensor<1x8xi1>
  return %0 : tensor<1x8xi1>
}
