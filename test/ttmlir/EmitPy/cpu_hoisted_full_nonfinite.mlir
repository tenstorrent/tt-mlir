// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-emitpy-pipeline -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir
// RUN: FileCheck %s --input-file=%t.py

// Full op with non-finite fill_value. Scalar FloatAttr holding inf/-inf/nan
// must emit valid Python literals (float('inf'), float('-inf'), float('nan')),
// not MLIR's textual form (Inf, -Inf, NaN), which would raise NameError at
// import time.

// CHECK-LABEL: def cpu_hoisted_ttir_full_{{.*}}
// CHECK: ttir_cpu.full(
// CHECK-SAME: fill_value=float('-inf')
// CHECK-LABEL: def full_neg_inf_validation
// CHECK: cpu_hoisted_ttir_full_{{.*}}
func.func @full_neg_inf_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 0xFF800000 : f32}> {ttir.should_hoist} : () -> tensor<32x32xf32>
  %out = "ttir.add"(%cpu_result, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %out : tensor<32x32xf32>
}

// CHECK-LABEL: def cpu_hoisted_ttir_full_{{.*}}
// CHECK: ttir_cpu.full(
// CHECK-SAME: fill_value=float('inf')
// CHECK-LABEL: def full_pos_inf_validation
// CHECK: cpu_hoisted_ttir_full_{{.*}}
func.func @full_pos_inf_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 0x7F800000 : f32}> {ttir.should_hoist} : () -> tensor<32x32xf32>
  %out = "ttir.add"(%cpu_result, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %out : tensor<32x32xf32>
}

// CHECK-LABEL: def cpu_hoisted_ttir_full_{{.*}}
// CHECK: ttir_cpu.full(
// CHECK-SAME: fill_value=float('nan')
// CHECK-LABEL: def full_nan_validation
// CHECK: cpu_hoisted_ttir_full_{{.*}}
func.func @full_nan_validation(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.full"() <{shape = array<i32: 32, 32>, fill_value = 0x7FC00000 : f32}> {ttir.should_hoist} : () -> tensor<32x32xf32>
  %out = "ttir.add"(%cpu_result, %arg0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %out : tensor<32x32xf32>
}
