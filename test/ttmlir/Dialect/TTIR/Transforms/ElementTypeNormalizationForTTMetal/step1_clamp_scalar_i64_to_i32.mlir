// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-element-type-normalization-for-ttmetal %s | FileCheck %s

// =============================================================================
// Step 1: clamp_scalar with i64 input: min/max stay i32, input/result become i32
// via type converter (no typecast).
// =============================================================================

// -----
// CHECK-LABEL: func.func @step1_clamp_scalar_i64_to_i32
// CHECK-SAME: (%arg0: tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: "ttir.clamp_scalar"(%arg0) <{max = 3 : i32, min = 0 : i32}> : (tensor<4x4xi32>) -> tensor<4x4xi32>
// CHECK: return
func.func @step1_clamp_scalar_i64_to_i32(%arg0: tensor<4x4xi64>) -> tensor<4x4xi64> {
  %0 = "ttir.clamp_scalar"(%arg0) <{max = 3 : i32, min = 0 : i32}> : (tensor<4x4xi64>) -> tensor<4x4xi64>
  return %0 : tensor<4x4xi64>
}
