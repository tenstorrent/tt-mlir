// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 3: Remaining i1 tensors are converted to bf16.
// Constants true->1.0, false->0.0.
// =============================================================================

// -----
// CHECK-LABEL: func.func @step3_i1_constant_true_false
// CHECK: "ttir.constant"() <{value = dense<{{.*}}> : tensor<2x2xbf16>}> : () -> tensor<2x2xbf16>
// CHECK: "ttir.constant"() <{value = dense<{{.*}}> : tensor<2x2xbf16>}> : () -> tensor<2x2xbf16>
func.func @step3_i1_constant_true_false() -> (tensor<2x2xi1>, tensor<2x2xi1>) {
  %t = "ttir.constant"() <{value = dense<true> : tensor<2x2xi1>}> : () -> tensor<2x2xi1>
  %f = "ttir.constant"() <{value = dense<false> : tensor<2x2xi1>}> : () -> tensor<2x2xi1>
  return %t, %f : tensor<2x2xi1>, tensor<2x2xi1>
}

// -----
// Remaining i1 (e.g. reshape of i1) is converted to bf16 in step 3.
// CHECK-LABEL: func.func @step3_remaining_i1_to_bf16
// CHECK-SAME: (%arg0: tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: "ttir.reshape"(%arg0) <{shape = [4 : i32, 4 : i32]}> : (tensor<4x4xbf16>) -> tensor<4x4xbf16>
// CHECK: return
func.func @step3_remaining_i1_to_bf16(%arg0: tensor<4x4xi1>) -> tensor<4x4xi1> {
  %0 = "ttir.reshape"(%arg0) <{shape = [4 : i32, 4 : i32]}> : (tensor<4x4xi1>) -> tensor<4x4xi1>
  return %0 : tensor<4x4xi1>
}

// -----
// Standalone i1 constant (no step 2 rewrite) gets converted in step 3.
// CHECK-LABEL: func.func @step3_standalone_i1_const
// CHECK: "ttir.constant"() <{value = dense<{{.*}}> : tensor<8x8xbf16>}> : () -> tensor<8x8xbf16>
// CHECK: return
func.func @step3_standalone_i1_const() -> tensor<8x8xi1> {
  %0 = "ttir.constant"() <{value = dense<true> : tensor<8x8xi1>}> : () -> tensor<8x8xi1>
  return %0 : tensor<8x8xi1>
}
