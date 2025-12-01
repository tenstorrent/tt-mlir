// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test multiple independent const-eval subgraphs.
// Each independent subgraph should create a separate CPU-hoisted function.

module {
  // CHECK: ttcore.device_module {

  // CHECK-LABEL: func.func @forward_split
  func.func @forward_split(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                           %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>},
                           %arg4: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // Two independent const-eval subgraphs should be created
    // CHECK: ttcore.load_cached
    // CHECK: ttcore.load_cached

    // CHECK: ttnn.add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: ttnn.add
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: ttnn.multiply
    %2 = "ttir.multiply"(%arg3, %arg4) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: ttnn.subtract
    %3 = "ttir.subtract"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK: ttnn.multiply 
    %4 = "ttir.multiply"(%3, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %4 : tensor<32x32xbf16>
  }
}
