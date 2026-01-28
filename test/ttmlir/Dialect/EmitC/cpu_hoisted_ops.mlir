// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that operations marked with {ttir.should_hoist} are properly hoisted to the CPU module
// and the EmitC pipeline generates appropriate structure for dylib-based CPU execution.
//
// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Verify that extern "C" declarations are generated for CPU-hoisted functions.
// CHECK: emitc.verbatim "extern {{.*}}cpu_hoisted_ttir_add_{{.*}}"

// Verify the device functions are still present in the output.
// CHECK-LABEL: func.func @add_validation

// Verify the CPU module contains LLVM functions for hoisted operations.
// CHECK: ttcore.cpu_module {
// CHECK: builtin.module {
// CHECK: llvm.func @cpu_hoisted_ttir_add_{{.*}}(

// Test: Binary add operation with CPU hoisting
// The hoisted op should generate a CPU function, while the device op goes through normal path.
func.func @add_validation(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %cpu_result = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %device_result = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %diff : tensor<32x32xf32>
}
