// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test basic const-eval CPU hoisting with parameter + constant ops.
// The const-eval subgraph (subtract on parameters/constants) should be hoisted to CPU module.

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // CHECK-LABEL: func.func private @forward_const_eval_0{{.*}} -> tensor<32x32xbf16

  // CHECK: [[ARG0:%[0-9]+]] = "ttnn.typecast"(%arg0)
  // CHECK: "ttnn.from_device"([[ARG0]])
  // CHECK: [[ARG1:%[0-9]+]] = "ttnn.typecast"(%arg1)
  // CHECK: "ttnn.from_device"([[ARG1]])

  // CHECK: call @hoisted_forward_const_eval_0_decl

  // CHECK: [[RES:%[0-9]+]] = "ttnn.to_dtype"
  // CHECK: "ttnn.to_device"([[RES]]

  // CHECK-LABEL: func.func @forward
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: [[CONSTEVAL:%[0-9]+]] = ttcore.load_cached{{.*}}%arg2, %arg3

    // CHECK: [[SUM:%[0-9]+]] = "ttnn.add"(%arg0, %arg1)
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: "ttnn.subtract"
    %1 = "ttir.subtract"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.multiply"([[SUM]], [[CONSTEVAL]])
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %2 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func private @hoisted_forward_const_eval_0_decl{{.*}} -> tensor<32x32xf32

  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK-LABEL: llvm.func @hoisted_forward_const_eval_0{{.*}}
}
