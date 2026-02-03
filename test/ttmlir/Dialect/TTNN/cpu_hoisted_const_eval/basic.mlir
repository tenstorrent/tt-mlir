// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // CHECK-LABEL: func.func private @forward_const_eval_0{{.*}} -> tensor<32x32xbf16
  // Inputs should already be on the host.
  // CHECK-NOT: "ttnn.from_device"
  // CHECK: "ttnn.to_dtype"(%arg0)
  // CHECK: "ttnn.to_dtype"(%arg1)

  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // CHECK: "ttnn.to_device"(%{{.*}})

  // CHECK-LABEL: func.func @forward
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: [[CONSTEVAL:%[0-9]+]] = ttcore.load_cached{{.*}}%arg2, %arg3

    // CHECK: [[SUM:%[0-9]+]] = "ttnn.add"(%{{.*}}, %arg1)
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: "ttnn.subtract"
    %1 = "ttir.subtract"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.multiply"([[SUM]], [[CONSTEVAL]])
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %2 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func private @forward_merge_return_multiple_values_const_eval_0
  // CHECK-SAME: -> (tensor<32x32xbf16{{.*}}>, tensor<32x32xbf16{{.*}}>)
  // CHECK: call @cpu_hoisted_const_eval_{{.*}}

  // CHECK-LABEL: func.func @forward_merge_return_multiple_values
  func.func @forward_merge_return_multiple_values(
                    %arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                    %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                    %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                    %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: [[CONSTEVAL:%[0-9]+]]:2 = ttcore.load_cached{{.*}}%arg1, %arg2, %arg3

    // CHECK: [[ADD0:%[0-9]+]] = "ttnn.add"(%{{.*}}, %arg1)
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: "ttnn.add"
    %1 = "ttir.add"(%arg1, %arg2)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3)  : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: [[MUL0:%[0-9]+]] = "ttnn.multiply"([[ADD0]], [[CONSTEVAL]]#1)
    %3 = "ttir.multiply"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.multiply"([[MUL0]], [[CONSTEVAL]]#0)
    %5 = "ttir.multiply"(%4, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %5 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> tensor<32x32xf32
  // CHECK-LABEL: func.func private @cpu_hoisted_const_eval_{{.*}} -> (tensor<32x32xf32{{.*}}>, tensor<32x32xf32{{.*}}>)

  // CHECK: ttcore.cpu_module {
  // CHECK: builtin.module {
  // CHECK-LABEL: llvm.func @cpu_hoisted_const_eval_{{.*}}(
  // CHECK-LABEL: llvm.func @cpu_hoisted_const_eval_{{.*}}(
}
