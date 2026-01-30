// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test multiple independent const-eval subgraphs.
// Each independent subgraph should create a separate CPU-hoisted function.

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // First const-eval function.
  // CHECK-LABEL: func.func private @forward_split_const_eval_0{{.*}} -> tensor<32x32xbf16
  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // Second const-eval function.
  // CHECK-LABEL: func.func private @forward_split_const_eval_1{{.*}} -> tensor<32x32xbf16
  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // CHECK-LABEL: func.func @forward_split
  func.func @forward_split(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                           %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                           %arg4: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: [[CE0:%[0-9]+]] = ttcore.load_cached{{.*}}%arg1, %arg2
    // CHECK: [[CE1:%[0-9]+]] = ttcore.load_cached{{.*}}%arg3, %arg4

    // CHECK: [[SUM:%[0-9]+]] = "ttnn.add"(%{{.*}}, %arg1)
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: "ttnn.add"(%arg1, %arg2)
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: "ttnn.multiply"(%arg3, %arg4)
    %2 = "ttir.multiply"(%arg3, %arg4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.subtract"([[SUM]], [[CE0]])
    %3 = "ttir.subtract"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.multiply"({{.*}}, [[CE1]])
    %4 = "ttir.multiply"(%3, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %4 : tensor<32x32xbf16>
  }

  // Should have declarations for multiple hoisted functions.
  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> tensor<32x32xf32
  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> tensor<32x32xf32

  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @cpu_hoisted_const_eval_{{.*}}(
  // CHECK: llvm.func @cpu_hoisted_const_eval_{{.*}}(
}
