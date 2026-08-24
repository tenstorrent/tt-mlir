// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Only "silu" is supported, and only ttnn.multiply carries lhs_activation at
// all: the runtime plumbs it for no other eltwise binary op, so accepting it
// elsewhere would execute without the activation and silently change results.

#dram = #ttnn.buffer_type<dram>
#l = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // CHECK: lhs_activation must be "silu", but got "relu"
  func.func @multiply_unsupported_activation(%a: tensor<32x32xbf16, #l>, %b: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.multiply"(%a, %b) <{lhs_activation = "relu"}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %0 : tensor<32x32xbf16, #l>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#l = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // CHECK: invalid properties {lhs_activation = "silu"} for op ttnn.add: this operation does not support properties
  func.func @add_does_not_accept_activation(%a: tensor<32x32xbf16, #l>, %b: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.add"(%a, %b) <{lhs_activation = "silu"}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %0 : tensor<32x32xbf16, #l>
  }
}
