// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-reduction-force-keep-dim %s | FileCheck %s

// Verify that reductions with keep_dim=false are converted to keep_dim=true
// with a reshape inserted after to restore the original output shape.

module {
  func.func @force_keep_dim_sum(%arg0: tensor<2x3x4xf32>) -> tensor<2x4xf32> {
    // CHECK-LABEL: func.func @force_keep_dim_sum
    // CHECK: "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: (tensor<2x1x4xf32>) -> tensor<2x4xf32>
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

// -----
module {
  func.func @force_keep_dim_mean(%arg0: tensor<4x8x16xf32>) -> tensor<4x16xf32> {
    // CHECK-LABEL: func.func @force_keep_dim_mean
    // CHECK: "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8x16xf32>) -> tensor<4x1x16xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: (tensor<4x1x16xf32>) -> tensor<4x16xf32>
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8x16xf32>) -> tensor<4x16xf32>
    return %0 : tensor<4x16xf32>
  }
}

// -----
module {
  func.func @force_keep_dim_max_last_dim(%arg0: tensor<2x3x4xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @force_keep_dim_max_last_dim
    // CHECK: "ttir.max"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<2x3x1xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: (tensor<2x3x1xf32>) -> tensor<2x3xf32>
    %0 = "ttir.max"(%arg0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}

// -----
module {
  func.func @force_keep_dim_min_first_dim(%arg0: tensor<2x3x4xf32>) -> tensor<3x4xf32> {
    // CHECK-LABEL: func.func @force_keep_dim_min_first_dim
    // CHECK: "ttir.min"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<1x3x4xf32>
    // CHECK: "ttir.reshape"
    // CHECK-SAME: (tensor<1x3x4xf32>) -> tensor<3x4xf32>
    %0 = "ttir.min"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<3x4xf32>
    return %0 : tensor<3x4xf32>
  }
}
