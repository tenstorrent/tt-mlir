// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --convert-ttnn-to-emitc %s 2>&1 | FileCheck %s
// RUN: not ttmlir-opt --convert-ttnn-to-emitpy %s 2>&1 | FileCheck %s
// RUN: not ttmlir-opt --convert-ttnn-to-ttir %s 2>&1 | FileCheck %s

// Only the flatbuffer path can represent lhs_activation. EmitC and EmitPy emit
// a binary call with no activation argument, and TTNN-to-TTIR rebuilds the op
// from operands alone, so each would drop the activation and produce a plain
// multiply -- different numerics with no diagnostic. All three refuse instead.
//
// This is the behaviour that keeps the attribute safe to add to only one op, so
// it is worth pinning: a future conversion that silently drops it would still
// pass every other test in this directory.

#dram = #ttnn.buffer_type<dram>
#l = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @multiply_lhs_activation_not_convertible(
      %a: tensor<32x32xbf16, #l>, %b: tensor<32x32xbf16, #l>)
      -> tensor<32x32xbf16, #l> {
    // CHECK: failed to legalize operation 'ttnn.multiply'
    %0 = "ttnn.multiply"(%a, %b) <{lhs_activation = "silu"}> : (tensor<32x32xbf16, #l>, tensor<32x32xbf16, #l>) -> tensor<32x32xbf16, #l>
    return %0 : tensor<32x32xbf16, #l>
  }
}
