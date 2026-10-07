// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// RUN: FileCheck %s --input-file=%t --check-prefix=HOISTCOUNT

module {
  // CHECK: ttcore.device_module {
  // CHECK: builtin.module

  // CHECK-LABEL: func.func private @forward_transpose_only_const_eval_0
  // CHECK: "ttnn.permute"
  // CHECK-NOT: call @cpu_hoisted_const_eval_

  // CHECK-LABEL: func.func @forward_transpose_only
  func.func @forward_transpose_only(%arg0: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                    %arg1: tensor<32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // The transpose of the weight %arg1 is const-eval'able but pure data movement.
    %0 = "ttir.permute"(%arg1) <{permutation = array<i64: 1, 0>}> : (tensor<32x64xbf16>) -> tensor<64x32xbf16>
    // %arg0 @ transpose(%arg1) : (32x64) @ (64x32) -> 32x32
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x64xbf16>, tensor<64x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func private @forward_with_arith_const_eval_0
  // CHECK: "ttnn.typecast"
  // CHECK: call @cpu_hoisted_const_eval_

  // CHECK-LABEL: func.func @forward_with_arith
  func.func @forward_with_arith(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                                %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.subtract"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.multiply"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }

  // CHECK: ttcore.cpu_module {

  // HOISTCOUNT-COUNT-1: ttir.cpu_hoist_call
  // HOISTCOUNT-NOT: ttir.cpu_hoist_call
}
