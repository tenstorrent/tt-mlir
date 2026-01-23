// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fold-constant-reshape-broadcast %s | FileCheck %s

// Test: Non-splat constant should NOT be folded
module {
  func.func @test_no_fold_non_splat(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
    // CHECK-LABEL: func.func @test_no_fold_non_splat
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<{{.*}}> : tensor<1x2x3xf32>}>
    // CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[CONST]])
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[RESHAPE]])
    // CHECK: return %[[ADD]]
    %cst = "ttir.constant"() <{value = dense<[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]> : tensor<1x2x3xf32>}> : () -> tensor<1x2x3xf32>
    %cst_reshaped = "ttir.reshape"(%cst) <{shape = [2 : i32, 3 : i32]}> : (tensor<1x2x3xf32>) -> tensor<2x3xf32>
    %result = "ttir.add"(%arg0, %cst_reshaped) : (tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    return %result : tensor<2x3xf32>
  }
}

// Test: Reshape not consumed by elementwise binary op should NOT be folded
module {
  func.func @test_no_fold_not_consumed_by_binary(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    // CHECK-LABEL: func.func @test_no_fold_not_consumed_by_binary
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[CONST]])
    // CHECK: %[[EXP:.*]] = "ttir.exp"(%[[RESHAPE]])
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[EXP]])
    // CHECK: return %[[ADD]]
    %cst = "ttir.constant"() <{value = dense<1.0> : tensor<1x32x64xf32>}> : () -> tensor<1x32x64xf32>
    %cst_reshaped = "ttir.reshape"(%cst) <{shape = [32 : i32, 64 : i32]}> : (tensor<1x32x64xf32>) -> tensor<32x64xf32>
    %exp = "ttir.exp"(%cst_reshaped) : (tensor<32x64xf32>) -> tensor<32x64xf32>
    %result = "ttir.add"(%arg0, %exp) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %result : tensor<32x64xf32>
  }
}
