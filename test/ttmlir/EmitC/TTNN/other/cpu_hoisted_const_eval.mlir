// E2E test for CPU-hoisted const-eval with EmitC dylib compilation.
// This test verifies the MLIR IR structure after the ttir-to-emitc-pipeline.
//
// Note: C++ code generation and dylib compilation are handled by tt-alchemist,
// not by ttmlir-translate. This test verifies the MLIR output structure.
//
// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file %t.mlir

module {
  // Verify extern "C" declarations are generated in the root module
  // CHECK: emitc.verbatim "extern

  // Verify the const-eval wrapper function calls the CPU-hoisted function
  // CHECK-LABEL: func.func private @forward_const_eval_0
  // CHECK: call @cpu_hoisted_forward_const_eval

  // Verify the forward function structure
  // CHECK-LABEL: func.func @forward

  // Verify the CPU module contains LLVM functions for hoisted const-eval ops
  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @cpu_hoisted_forward_const_eval

  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // Device operation: uses input
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CPU-hoisted operations: only use parameters/constants
    %1 = "ttir.subtract"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // Device operation: uses both
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %2 : tensor<32x32xbf16>
  }
}
