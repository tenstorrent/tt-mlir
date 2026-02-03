// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-reduction-keep-dim %s | FileCheck %s

// Test: No transform when reshape output shape does not match keep_dim=true shape
// (reshape to [1,2,4] instead of [2,1,4] so pattern does not apply)
module {
  func.func @no_transform_reshape_mismatch(%arg0: tensor<2x3x4xf32>) -> tensor<1x2x4xf32> {
    // CHECK-LABEL: func.func @no_transform_reshape_mismatch
    // CHECK: "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}>
    // CHECK: "ttir.reshape"
    %0 = "ttir.sum"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<2x3x4xf32>) -> tensor<2x4xf32>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 2 : i32, 4 : i32]}> : (tensor<2x4xf32>) -> tensor<1x2x4xf32>
    return %1 : tensor<1x2x4xf32>
  }
}
