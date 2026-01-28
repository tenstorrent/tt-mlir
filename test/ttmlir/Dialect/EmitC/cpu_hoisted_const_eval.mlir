// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that const-eval operations are properly hoisted to the CPU module when
// enable-cpu-hoisted-const-eval=true and the EmitC pipeline generates appropriate
// structure for dylib-based CPU execution.
//
// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module {
  // Verify extern "C" declarations are generated for CPU-hoisted const-eval functions.
  // CHECK: emitc.verbatim "extern {{.*}}cpu_hoisted_forward_const_eval_{{.*}}"

  // Verify the const-eval wrapper function is generated.
  // CHECK-LABEL: func.func private @forward_const_eval_0
  // CHECK: call @cpu_hoisted_forward_const_eval_{{.*}}

  // Verify the main forward function.
  // CHECK-LABEL: func.func @forward
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // This op uses input and parameter - should stay on device.
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // This op uses only parameters/constants - should be hoisted to CPU.
    // CHECK-NOT: "ttnn.subtract"
    %1 = "ttir.subtract"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // This op uses both device and const-eval results.
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %2 : tensor<32x32xbf16>
  }

  // Test merged const-eval: connected const-eval ops should be merged into one CPU function.
  // CHECK-LABEL: func.func @forward_merged
  func.func @forward_merged(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                            %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                            %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                            %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // These connected ops should be merged into a single CPU function.
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.subtract"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %4 : tensor<32x32xbf16>
  }

  // Test const-eval with creation ops (zeros).
  // CHECK-LABEL: func.func @forward_zeros
  func.func @forward_zeros(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                           %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.multiply"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }

  // Verify the CPU module structure with LLVM functions (at the end).
  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @cpu_hoisted_forward_const_eval_{{.*}}(
}
