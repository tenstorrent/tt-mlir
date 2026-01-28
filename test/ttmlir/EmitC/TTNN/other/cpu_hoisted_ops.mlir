// E2E test for single-op CPU-hoisting with EmitC dylib compilation.
// This test verifies operations marked with {ttir.should_hoist} attribute are
// properly hoisted to a CPU module with LLVM IR.
//
// Note: C++ code generation and dylib compilation are handled by tt-alchemist,
// not by ttmlir-translate. This test verifies the MLIR output structure.
//
// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file %t.mlir

module {
  // Verify extern "C" declarations are generated
  // CHECK: emitc.verbatim "extern

  // Verify the device functions are still present
  // CHECK-LABEL: func.func @add_hoisted

  // Test: Binary add operation with CPU hoisting
  func.func @add_hoisted(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // This op is hoisted to CPU
    %cpu_result = "ttir.add"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // This op runs on device
    %device_result = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Compute difference to validate
    %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %diff : tensor<32x32xf32>
  }

  // Test: Unary relu operation with CPU hoisting
  // CHECK-LABEL: func.func @relu_hoisted
  func.func @relu_hoisted(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %cpu_result = "ttir.relu"(%arg0) {ttir.should_hoist} : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %device_result = "ttir.relu"(%arg0) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %diff : tensor<32x32xf32>
  }

  // Test: Matmul operation with CPU hoisting
  // CHECK-LABEL: func.func @matmul_hoisted
  func.func @matmul_hoisted(%arg0: tensor<32x64xf32>, %arg1: tensor<64x32xf32>) -> tensor<32x32xf32> {
    %cpu_result = "ttir.matmul"(%arg0, %arg1) {ttir.should_hoist} : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<32x32xf32>
    %device_result = "ttir.matmul"(%arg0, %arg1) : (tensor<32x64xf32>, tensor<64x32xf32>) -> tensor<32x32xf32>
    %diff = "ttir.subtract"(%cpu_result, %device_result) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %diff : tensor<32x32xf32>
  }

  // Verify the CPU module contains LLVM functions for hoisted ops (at the end)
  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @cpu_hoisted_ttir_add
}
