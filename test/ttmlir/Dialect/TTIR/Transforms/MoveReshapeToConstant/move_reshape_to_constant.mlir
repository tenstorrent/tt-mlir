// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-move-reshape-to-constant %s | FileCheck %s

// Test: Reshape on LHS activation, constant on RHS
// The reshape should move to the constant path
module {
  func.func @test_reshape_lhs_const_rhs(%arg0: tensor<32x2560xf32>) -> tensor<32x1x2560xf32> {
    // CHECK-LABEL: func.func @test_reshape_lhs_const_rhs
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[CONST_RESHAPE:.*]] = "ttir.reshape"(%[[CONST]])
    // CHECK-SAME: shape = [32 : i32, 2560 : i32]
    // CHECK: %[[POW:.*]] = "ttir.pow"(%arg0, %[[CONST_RESHAPE]])
    // CHECK-SAME: tensor<32x2560xf32>
    // CHECK: %[[OUT_RESHAPE:.*]] = "ttir.reshape"(%[[POW]])
    // CHECK-SAME: shape = [32 : i32, 1 : i32, 2560 : i32]
    // CHECK: return %[[OUT_RESHAPE]]
    %cst = "ttir.constant"() <{value = dense<2.0> : tensor<32x1x2560xf32>}> : () -> tensor<32x1x2560xf32>
    %reshaped = "ttir.reshape"(%arg0) <{shape = [32 : i32, 1 : i32, 2560 : i32]}> : (tensor<32x2560xf32>) -> tensor<32x1x2560xf32>
    %result = "ttir.pow"(%reshaped, %cst) : (tensor<32x1x2560xf32>, tensor<32x1x2560xf32>) -> tensor<32x1x2560xf32>
    return %result : tensor<32x1x2560xf32>
  }
}

// Test: Works with add operation
module {
  func.func @test_add_reshape_to_constant(%arg0: tensor<64x128xf32>) -> tensor<64x1x128xf32> {
    // CHECK-LABEL: func.func @test_add_reshape_to_constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[CONST_RESHAPE:.*]] = "ttir.reshape"(%[[CONST]])
    // CHECK-SAME: shape = [64 : i32, 128 : i32]
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[CONST_RESHAPE]])
    // CHECK: %[[OUT_RESHAPE:.*]] = "ttir.reshape"(%[[ADD]])
    // CHECK: return %[[OUT_RESHAPE]]
    %cst = "ttir.constant"() <{value = dense<1.0> : tensor<64x1x128xf32>}> : () -> tensor<64x1x128xf32>
    %reshaped = "ttir.reshape"(%arg0) <{shape = [64 : i32, 1 : i32, 128 : i32]}> : (tensor<64x128xf32>) -> tensor<64x1x128xf32>
    %result = "ttir.add"(%reshaped, %cst) : (tensor<64x1x128xf32>, tensor<64x1x128xf32>) -> tensor<64x1x128xf32>
    return %result : tensor<64x1x128xf32>
  }
}

// Test: Works with subtract operation
module {
  func.func @test_sub_reshape_to_constant(%arg0: tensor<16x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK-LABEL: func.func @test_sub_reshape_to_constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"
    // CHECK: %[[CONST_RESHAPE:.*]] = "ttir.reshape"(%[[CONST]])
    // CHECK-SAME: shape = [16 : i32, 32 : i32]
    // CHECK: %[[SUB:.*]] = "ttir.subtract"(%arg0, %[[CONST_RESHAPE]])
    // CHECK: %[[OUT_RESHAPE:.*]] = "ttir.reshape"(%[[SUB]])
    // CHECK: return %[[OUT_RESHAPE]]
    %cst = "ttir.constant"() <{value = dense<0.5> : tensor<1x16x32xf32>}> : () -> tensor<1x16x32xf32>
    %reshaped = "ttir.reshape"(%arg0) <{shape = [1 : i32, 16 : i32, 32 : i32]}> : (tensor<16x32xf32>) -> tensor<1x16x32xf32>
    %result = "ttir.subtract"(%reshaped, %cst) : (tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %result : tensor<1x16x32xf32>
  }
}
