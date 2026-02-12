// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fold-constant-reshape-broadcast %s | FileCheck %s

// Test: Basic reshape of splat constant consumed by elementwise binary op
module {
  func.func @test_fold_reshape_splat(%arg0: tensor<32x2560xf32>) -> tensor<32x2560xf32> {
    // CHECK-LABEL: func.func @test_fold_reshape_splat
    // The reshape should be folded into the constant
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<2.{{0*}}e+00> : tensor<32x2560xf32>}>
    // CHECK-NOT: ttir.reshape
    // CHECK: %[[POW:.*]] = "ttir.pow"(%arg0, %[[CONST]])
    // CHECK: return %[[POW]]
    %cst = "ttir.constant"() <{value = dense<2.0> : tensor<32x1x2560xf32>}> : () -> tensor<32x1x2560xf32>
    %cst_reshaped = "ttir.reshape"(%cst) <{shape = [32 : i32, 2560 : i32]}> : (tensor<32x1x2560xf32>) -> tensor<32x2560xf32>
    %result = "ttir.pow"(%arg0, %cst_reshaped) : (tensor<32x2560xf32>, tensor<32x2560xf32>) -> tensor<32x2560xf32>
    return %result : tensor<32x2560xf32>
  }
}

// Test: Integer splat constant
module {
  func.func @test_fold_reshape_integer_splat(%arg0: tensor<32x64xi32>) -> tensor<32x64xi32> {
    // CHECK-LABEL: func.func @test_fold_reshape_integer_splat
    // CHECK: %[[CONST:.*]] = "ttir.constant"() <{value = dense<42> : tensor<32x64xi32>}>
    // CHECK-NOT: ttir.reshape
    // CHECK: %[[ADD:.*]] = "ttir.add"(%arg0, %[[CONST]])
    // CHECK: return %[[ADD]]
    %cst = "ttir.constant"() <{value = dense<42> : tensor<1x32x64xi32>}> : () -> tensor<1x32x64xi32>
    %cst_reshaped = "ttir.reshape"(%cst) <{shape = [32 : i32, 64 : i32]}> : (tensor<1x32x64xi32>) -> tensor<32x64xi32>
    %result = "ttir.add"(%arg0, %cst_reshaped) : (tensor<32x64xi32>, tensor<32x64xi32>) -> tensor<32x64xi32>
    return %result : tensor<32x64xi32>
  }
}
