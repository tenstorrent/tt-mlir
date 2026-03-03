// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test const-eval subgraphs that merge due to shared dependencies.
// When two const-eval operations share values, they should be merged into a single subgraph.

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // Single merged const-eval function (all const-eval ops merge).
  // CHECK-LABEL: func.func private @forward_merge_const_eval_0{{.*}} -> tensor<32x32xbf16
  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // CHECK-LABEL: func.func @forward_merge
  func.func @forward_merge(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                           %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // Only ONE load_cached since all const-eval ops are merged.
    // CHECK: [[CE:%[0-9]+]] = ttcore.load_cached{{.*}}%arg1, %arg2, %arg3

    // Input operation - stays in device module
    // CHECK: [[SUM:%[0-9]+]] = "ttnn.add"(%{{.*}}, %arg1)
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // All these ops are const-eval and merged into one subgraph.
    // CHECK-NOT: "ttnn.add"(%arg1, %arg2)
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: "ttnn.add"(%arg2, %arg3)
    %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // Uses results of above - causes merge.
    // CHECK-NOT: "ttnn.subtract"
    %3 = "ttir.subtract"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // Final operation uses input result and merged const-eval result.
    // CHECK: "ttnn.multiply"([[SUM]], [[CE]])
    %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %4 : tensor<32x32xbf16>
  }

  // Should have only ONE hoisted function declaration (merged subgraph).
  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> tensor<32x32xf32

  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // Should have only ONE hoisted function (merged).
  // CHECK: llvm.func @cpu_hoisted_const_eval_{{.*}}(
}
