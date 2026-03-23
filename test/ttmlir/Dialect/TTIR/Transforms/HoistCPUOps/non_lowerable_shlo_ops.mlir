// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --ttcore-wrap-device-module --cpu-hoist-non-lowerable-shlo-ops -o %t %s
// RUN: FileCheck %s --input-file=%t

// --- Test 1: Allowlisted ops are hoisted ---

// CHECK-LABEL: func.func @einsum
// CHECK: call @cpu_hoisted_stablehlo_einsum_{{.*}}
// CHECK: return
func.func @einsum(%lhs: tensor<32x64xf32>, %rhs: tensor<64x48xf32>) -> tensor<32x48xf32> {
  %result = "stablehlo.einsum"(%lhs, %rhs) {
    einsum_config = "ij,jk->ik"
  } : (tensor<32x64xf32>, tensor<64x48xf32>) -> tensor<32x48xf32>
  return %result : tensor<32x48xf32>
}

// --- Test 2: Non-allowlisted ops are not hoisted ---

// CHECK-LABEL: func.func @non_allowlisted_op
// CHECK-NOT: call @cpu_hoisted
// CHECK: stablehlo.dynamic_reshape
// CHECK: return
func.func @non_allowlisted_op(%operand: tensor<2x3xi64>,
                               %shape: tensor<2xi64>) -> tensor<3x2xi64> {
  %result = "stablehlo.dynamic_reshape"(%operand, %shape)
    : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<3x2xi64>
  return %result : tensor<3x2xi64>
}

// dynamic_update_slice is now lowered to ttir.slice_write, not hoisted.
// CHECK-LABEL: func.func @dynamic_update_slice_not_hoisted
// CHECK-NOT: call @cpu_hoisted
// CHECK: stablehlo.dynamic_update_slice
// CHECK: return
func.func @dynamic_update_slice_not_hoisted(%base: tensor<4xi32>, %update: tensor<2xi32>,
                                            %start: tensor<i32>) -> tensor<4xi32> {
  %result = stablehlo.dynamic_update_slice %base, %update, %start
    : (tensor<4xi32>, tensor<2xi32>, tensor<i32>) -> tensor<4xi32>
  return %result : tensor<4xi32>
}

// Verify hoisted functions exist in CPU module.
// CHECK: ttcore.cpu_module
// CHECK: func.func @cpu_hoisted_stablehlo_einsum_{{.*}}
// CHECK: stablehlo.einsum
