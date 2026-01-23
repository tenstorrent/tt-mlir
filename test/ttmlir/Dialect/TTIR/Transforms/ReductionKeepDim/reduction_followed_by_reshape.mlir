// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-reduction-keep-dim %s | FileCheck %s

// Test: Pattern 1 - Reduction (keep_dim=false) followed by reshape that adds
// back the reduced dimension. Should fuse to single reduction with keep_dim=true.
module {
  func.func @reduction_then_reshape_sum(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
    // CHECK-LABEL: func.func @reduction_then_reshape_sum
    // CHECK-NOT: ttir.reshape
    // CHECK: "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}>
    // CHECK-SAME: (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<2x4xf32>
    %1 = "ttir.reshape"(%0) <{shape = [2 : i32, 1 : i32, 4 : i32]}> : (tensor<2x4xf32>) -> tensor<2x1x4xf32>
    return %1 : tensor<2x1x4xf32>
  }
}

// -----
module {
  func.func @reduction_then_reshape_mean(%arg0: tensor<4x8x16xf32>) -> tensor<4x1x16xf32> {
    // CHECK-LABEL: func.func @reduction_then_reshape_mean
    // CHECK-NOT: ttir.reshape
    // CHECK: "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}>
    // CHECK-SAME: (tensor<4x8x16xf32>) -> tensor<4x1x16xf32>
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8x16xf32>) -> tensor<4x16xf32>
    %1 = "ttir.reshape"(%0) <{shape = [4 : i32, 1 : i32, 16 : i32]}> : (tensor<4x16xf32>) -> tensor<4x1x16xf32>
    return %1 : tensor<4x1x16xf32>
  }
}
