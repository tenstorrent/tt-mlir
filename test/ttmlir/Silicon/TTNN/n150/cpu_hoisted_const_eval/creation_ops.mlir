// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test creation ops (zeros, full) in const-eval subgraphs.
// Creation ops consumed by const-eval operations should be included in the CPU-hoisted function.

module {
  // CHECK: ttcore.device_module {

  // CHECK-LABEL: func.func @forward_with_zeros
  func.func @forward_with_zeros(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                                %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached

    // CHECK-NOT: ttnn.add
    %0 = "ttir.zeros"() <{shape = array<i32:32, 32>}> : () -> tensor<32x32xbf16>
    
    // CHECK-NOT: ttnn.add
    %1 = "ttir.add"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: ttnn.add
    %2 = "ttir.add"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: ttnn.multiply
    %3 = "ttir.multiply"(%2, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %3 : tensor<32x32xbf16>
  }

  // CHECK-LABEL: func.func @forward_with_full
  func.func @forward_with_full(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                               %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
    // CHECK: ttcore.load_cached

    %0 = "ttir.full"() <{fill_value = 3.000000e+00 : f32, shape = array<i32: 32, 32>}> : () -> tensor<32x32xbf16>
    // Parameter * full should be hoisted
    %1 = "ttir.multiply"(%arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: "ttnn.add"
    %2 = "ttir.add"(%arg0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}
