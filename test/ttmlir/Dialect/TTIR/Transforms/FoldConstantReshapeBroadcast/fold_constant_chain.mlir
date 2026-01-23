// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fold-constant-reshape-broadcast %s | FileCheck %s

// Test: Chain of reshape -> broadcast -> reshape from splat constant
module {
  func.func @test_fold_reshape_broadcast_reshape_chain(%arg0: tensor<128x360xf32>) -> tensor<128x360xf32> {
    // CHECK-LABEL: func.func @test_fold_reshape_broadcast_reshape_chain
    // The entire chain should be folded into a single constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<2.{{0*}}e+00> : tensor<128x360xf32>}>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[POW:.*]] = "ttir.pow"(%arg0, %[[CONST]])
    // CHECK: return %[[POW]]
    %cst = "ttir.constant"() <{value = dense<2.0> : tensor<f32>}> : () -> tensor<f32>
    %cst_reshape1 = "ttir.reshape"(%cst) <{shape = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<f32>) -> tensor<1x1x1xf32>
    %cst_broadcast = "ttir.broadcast"(%cst_reshape1) <{broadcast_dimensions = array<i64: 1, 128, 360>}> : (tensor<1x1x1xf32>) -> tensor<1x128x360xf32>
    %cst_reshape2 = "ttir.reshape"(%cst_broadcast) <{shape = [128 : i32, 360 : i32]}> : (tensor<1x128x360xf32>) -> tensor<128x360xf32>
    %result = "ttir.pow"(%arg0, %cst_reshape2) : (tensor<128x360xf32>, tensor<128x360xf32>) -> tensor<128x360xf32>
    return %result : tensor<128x360xf32>
  }
}

// Test: Multiple reshapes in chain with broadcast
module {
  func.func @test_fold_multiple_reshapes(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    // CHECK-LABEL: func.func @test_fold_multiple_reshapes
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<5.{{0*}}e-01> : tensor<32x64xf32>}>
    // CHECK-NOT: ttir.reshape
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[CONST]])
    // CHECK: return %[[ADD]]
    %cst = "ttir.constant"() <{value = dense<0.5> : tensor<1xf32>}> : () -> tensor<1xf32>
    %cst_reshape1 = "ttir.reshape"(%cst) <{shape = [1 : i32, 1 : i32]}> : (tensor<1xf32>) -> tensor<1x1xf32>
    %cst_broadcast = "ttir.broadcast"(%cst_reshape1) <{broadcast_dimensions = array<i64: 32, 64>}> : (tensor<1x1xf32>) -> tensor<32x64xf32>
    %result = "ttir.add"(%arg0, %cst_broadcast) : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %result : tensor<32x64xf32>
  }
}
