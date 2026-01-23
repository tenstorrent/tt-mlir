// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-reduction-keep-dim %s | FileCheck %s

// Test: Pattern 2 - Reshape that adds size-1 dimension, then reduction with
// keep_dim=false. Should fuse to reduction on original input with keep_dim=true
// and adjusted dim_arg.
module {
  func.func @reshape_then_reduction_sum(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
    // CHECK-LABEL: func.func @reshape_then_reduction_sum
    // CHECK-NOT: ttir.reshape
    // CHECK: "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}>
    // CHECK-SAME: (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
    %0 = "ttir.reshape"(%arg0) <{shape = [2 : i32, 1 : i32, 3 : i32, 4 : i32]}> : (tensor<2x3x4xf32>) -> tensor<2x1x3x4xf32>
    %1 = "ttir.sum"(%0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x1x3x4xf32>) -> tensor<2x1x4xf32>
    return %1 : tensor<2x1x4xf32>
  }
}

// -----
module {
  func.func @reshape_then_reduction_mean(%arg0: tensor<4x8x16xf32>) -> tensor<4x1x16xf32> {
    // CHECK-LABEL: func.func @reshape_then_reduction_mean
    // CHECK-NOT: ttir.reshape
    // CHECK: "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}>
    // CHECK-SAME: (tensor<4x8x16xf32>) -> tensor<4x1x16xf32>
    %0 = "ttir.reshape"(%arg0) <{shape = [4 : i32, 1 : i32, 8 : i32, 16 : i32]}> : (tensor<4x8x16xf32>) -> tensor<4x1x8x16xf32>
    %1 = "ttir.mean"(%0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<4x1x8x16xf32>) -> tensor<4x1x16xf32>
    return %1 : tensor<4x1x16xf32>
  }
}
