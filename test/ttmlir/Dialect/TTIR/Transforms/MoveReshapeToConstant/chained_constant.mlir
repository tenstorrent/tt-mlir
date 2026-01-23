// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-move-reshape-to-constant %s | FileCheck %s

// Test: Constant followed by reshape is still recognized as "from constant"
// The reshapes on constant path cancel out (32x2560 -> 32x1x2560 -> 32x2560)
module {
  func.func @test_reshaped_constant(%arg0: tensor<32x2560xf32>) -> tensor<32x1x2560xf32> {
    // CHECK-LABEL: func.func @test_reshaped_constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[CONST]])
    // CHECK: %[[OUT_RESHAPE:.*]] = "ttir.reshape"(%[[ADD]])
    // CHECK: return %[[OUT_RESHAPE]]
    %cst = "ttir.constant"() <{value = dense<1.0> : tensor<32x2560xf32>}> : () -> tensor<32x2560xf32>
    %cst_reshaped = "ttir.reshape"(%cst) <{shape = [32 : i32, 1 : i32, 2560 : i32]}> : (tensor<32x2560xf32>) -> tensor<32x1x2560xf32>
    %reshaped = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2560 : i32]}> : (tensor<32x2560xf32>) -> tensor<32x1x2560xf32>
    %result = "ttir.add"(%reshaped, %cst_reshaped) : (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32>
    return %result : tensor<32x1x2560xf32>
  }
}

// Test: Constant followed by broadcast is still recognized as "from constant"
module {
  func.func @test_broadcast_constant(%arg0: tensor<32x2560xf32>) -> tensor<32x1x2560xf32> {
    // CHECK-LABEL: func.func @test_broadcast_constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[BCAST:.*]] = "ttir.broadcast"(%[[CONST]])
    // CHECK: %[[CONST_RESHAPE:.*]] = "ttir.reshape"(%[[BCAST]])
    // CHECK-SAME: shape = [32 : i32, 2560 : i32]
    // CHECK: %[[MUL:.*]] = "ttir.multiply"(%arg0, %[[CONST_RESHAPE]])
    // CHECK: %[[OUT_RESHAPE:.*]] = "ttir.reshape"(%[[MUL]])
    // CHECK: return %[[OUT_RESHAPE]]
    %cst = "ttir.constant"() <{value = dense<2.0> : tensor<1x1x2560xf32>}> : () -> tensor<1x1x2560xf32>
    %cst_bcast = "ttir.broadcast"(%cst) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x1x2560xf32>) -> tensor<32x1x2560xf32>
    %reshaped = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2560 : i32]}> : (tensor<32x2560xf32>) -> tensor<32x1x2560xf32>
    %result = "ttir.multiply"(%reshaped, %cst_bcast) : (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32>
    return %result : tensor<32x1x2560xf32>
  }
}
