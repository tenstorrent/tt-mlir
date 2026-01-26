// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-move-reshape-to-constant %s | FileCheck %s

// Test: No transform when both operands are activations (neither is constant)
module {
  func.func @test_no_constant(%arg0: tensor<32x2560xf32>, %arg1: tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32> {
    // CHECK-LABEL: func.func @test_no_constant
    // CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%arg0)
    // CHECK: %[[ADD:.*]] = "ttir.add"(%[[RESHAPE]], %arg1)
    // CHECK: return %[[ADD]]
    %reshaped = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2560 : i32]}> : (tensor<32x2560xf32>) -> tensor<32x1x2560xf32>
    %result = "ttir.add"(%reshaped, %arg1) : (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32>
    return %result : tensor<32x1x2560xf32>
  }
}

// Test: No transform when reshape has multiple uses
module {
  func.func @test_reshape_multiple_uses(%arg0: tensor<32x2560xf32>) -> (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) {
    // CHECK-LABEL: func.func @test_reshape_multiple_uses
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%arg0)
    // CHECK: %[[POW:.*]] = "ttir.pow"(%[[RESHAPE]], %[[CONST]])
    // CHECK: %[[MUL:.*]] = "ttir.multiply"(%[[RESHAPE]], %[[CONST]])
    // CHECK: return %[[POW]], %[[MUL]]
    %cst = "ttir.constant"() <{value = dense<2.0> : tensor<32x1x2560xf32>}> : () -> tensor<32x1x2560xf32>
    %reshaped = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2560 : i32]}> : (tensor<32x2560xf32>) -> tensor<32x1x2560xf32>
    %result1 = "ttir.pow"(%reshaped, %cst) : (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32>
    %result2 = "ttir.multiply"(%reshaped, %cst) : (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32>
    return %result1, %result2 : tensor<32x1x2560xf32>, tensor<32x1x2560xf32>
  }
}
