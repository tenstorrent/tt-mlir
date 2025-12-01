// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-cpu-hoisted-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Test const-eval subgraphs that merge due to shared dependencies.
// When two const-eval operations share values, they should be merged into a single subgraph.

module {
  // CHECK: ttcore.device_module {

  // CHECK-LABEL: func.func @forward_merge
  func.func @forward_merge(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                           %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                           %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // Only ONE load_cached since all const-eval ops are merged
    // CHECK: ttcore.load_cached
    // CHECK-NOT: ttcore.load_cached

    // CHECK: ttnn.add
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: ttnn.add
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    // CHECK-NOT: ttnn.subtract
    %3 = "ttir.subtract"(%1, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %4 = "ttir.multiply"(%0, %3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %4 : tensor<32x32xbf16>
  }
}
