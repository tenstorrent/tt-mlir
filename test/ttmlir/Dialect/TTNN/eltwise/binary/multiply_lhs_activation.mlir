// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt %s | FileCheck %s

// lhs_activation names a unary the multiply kernel applies to operand A before
// the multiply, so a producer's activation costs no separate op -- SwiGLU's
// multiply(silu(gate), up).

#dram = #ttnn.buffer_type<dram>
#l = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  // CHECK-LABEL: func.func @multiply_silu
  // CHECK: lhs_activation = "silu"
  func.func @multiply_silu(%a: tensor<32x32xbf16, #l>, %b: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.multiply"(%a, %b) <{lhs_activation = "silu"}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %0 : tensor<32x32xbf16, #l>
  }

  // Every other multiply in the tree leaves it unset.
  // CHECK-LABEL: func.func @multiply_plain
  // CHECK-NOT: lhs_activation
  func.func @multiply_plain(%a: tensor<32x32xbf16, #l>, %b: tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l> {
    %0 = "ttnn.multiply"(%a, %b) : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %0 : tensor<32x32xbf16, #l>
  }
}
