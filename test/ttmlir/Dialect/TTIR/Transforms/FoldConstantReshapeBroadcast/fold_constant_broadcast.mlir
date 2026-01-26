// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fold-constant-reshape-broadcast %s | FileCheck %s

// Test: Basic broadcast of splat constant
module {
  func.func @test_fold_broadcast_splat(%arg0: tensor<128x360xf32>) -> tensor<128x360xf32> {
    // CHECK-LABEL: func.func @test_fold_broadcast_splat
    // The broadcast should be folded into the constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<2.{{0*}}e+00> : tensor<128x360xf32>}>
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[MUL:.*]] = "ttir.multiply"(%arg0, %[[CONST]])
    // CHECK: return %[[MUL]]
    %cst = "ttir.constant"() <{value = dense<2.0> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
    %cst_broadcast = "ttir.broadcast"(%cst) <{broadcast_dimensions = array<i64: 128, 360>}> : (tensor<1x1xf32>) -> tensor<128x360xf32>
    %result = "ttir.multiply"(%arg0, %cst_broadcast) : (tensor<128x360xf32>, tensor<128x360xf32>) -> tensor<128x360xf32>
    return %result : tensor<128x360xf32>
  }
}

// Test: Integer splat broadcast
module {
  func.func @test_fold_broadcast_integer_splat(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    // CHECK-LABEL: func.func @test_fold_broadcast_integer_splat
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<7> : tensor<64x128xi32>}>
    // CHECK-NOT: ttir.broadcast
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[CONST]])
    // CHECK: return %[[ADD]]
    %cst = "ttir.constant"() <{value = dense<7> : tensor<1x128xi32>}> : () -> tensor<1x128xi32>
    %cst_broadcast = "ttir.broadcast"(%cst) <{broadcast_dimensions = array<i64: 64, 1>}> : (tensor<1x128xi32>) -> tensor<64x128xi32>
    %result = "ttir.add"(%arg0, %cst_broadcast) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %result : tensor<64x128xi32>
  }
}
