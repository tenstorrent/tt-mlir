// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-reduction-force-keep-dim %s | FileCheck %s

// Verify reductions that already have keep_dim=true are not modified.
module {
  func.func @already_keep_dim_sum(%arg0: tensor<2x3x4xf32>) -> tensor<2x1x4xf32> {
    // CHECK-LABEL: func.func @already_keep_dim_sum
    // CHECK: "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
    // CHECK-NEXT: return
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<2x3x4xf32>) -> tensor<2x1x4xf32>
    return %0 : tensor<2x1x4xf32>
  }
}

// -----
module {
  func.func @already_keep_dim_mean(%arg0: tensor<4x8x16xf32>) -> tensor<4x1x16xf32> {
    // CHECK-LABEL: func.func @already_keep_dim_mean
    // CHECK: "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8x16xf32>) -> tensor<4x1x16xf32>
    // CHECK-NEXT: return
    %0 = "ttir.mean"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8x16xf32>) -> tensor<4x1x16xf32>
    return %0 : tensor<4x1x16xf32>
  }
}
