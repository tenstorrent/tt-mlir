// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test creation ops (zeros, ones, full) in const-eval subgraphs.
// Creation ops consumed by const-eval operations should be included in the CPU-hoisted function.

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // Const-eval function with zeros op.
  // Zeros op shouldn't be CPU-hoisted, as it is returned from the const-eval function.
  // CHECK-LABEL: func.func private @forward_with_zeros_const_eval_0{{.*}} -> (tensor<1x1xbf16{{.*}}>, tensor<32x32xbf16{{.*}}>)
  // CHECK: ttnn.zeros
  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // CHECK-LABEL: func.func @forward_with_zeros
  func.func @forward_with_zeros(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // CHECK: [[CE:%[0-9]+]]:2 = ttcore.load_cached{{.*}}%arg1

    // CHECK-NOT: ttnn.zeros
    %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xbf16>
    // CHECK-NOT: "ttnn.add"(%arg1,
    %1 = "ttir.add"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.add"(%{{.*}}, [[CE]]#0)
    %2 = "ttir.add"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.multiply"({{.*}}, [[CE]]#1)
    %3 = "ttir.multiply"(%2, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }

  // Const-eval function with full op.
  // Full op should be CPU-hoisted.
  // CHECK-LABEL: func.func private @forward_with_full_const_eval_0{{.*}} -> tensor<32x32xbf16
  // CHECK-NOT: ttnn.full
  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // CHECK-LABEL: func.func @forward_with_full
  func.func @forward_with_full(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                               %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // CHECK: [[CE:%[0-9]+]] = ttcore.load_cached{{.*}}%arg1

    %0 = "ttir.full"() <{fill_value = 3.000000e+00 : f32, shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
    // CHECK-NOT: "ttnn.multiply"(%arg1,
    %1 = "ttir.multiply"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.add"(%{{.*}}, [[CE]])
    %2 = "ttir.add"(%arg0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> tensor<32x32xf32
  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> tensor<32x32xf32

  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK: llvm.func @cpu_hoisted_const_eval_{{.*}}(
  // CHECK: llvm.func @cpu_hoisted_const_eval_{{.*}}(
}
