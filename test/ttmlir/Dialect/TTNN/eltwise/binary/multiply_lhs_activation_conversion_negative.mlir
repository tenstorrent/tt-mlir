// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --convert-ttnn-to-emitc %s 2>&1 | FileCheck %s
// RUN: not ttmlir-opt --convert-ttnn-to-emitpy %s 2>&1 | FileCheck %s

// Only the flatbuffer path can represent lhs_activation. EmitC and EmitPy emit a
// binary call with no activation argument, so either would drop the activation
// and produce a plain multiply -- different numerics with no diagnostic. Both
// refuse instead, and this pins that: a conversion that silently dropped it
// would still pass every other test in this directory.
//
// TTNN-to-TTIR refuses it too, but is not covered here. That pass marks the TTNN
// dialect only dynamically illegal -- ops inside a D2M subgraph or hoisted via
// isTTNNHoistGenericViaD2MOp -- so an ordinary ttnn.multiply is legal, never
// reaches the pattern, and the pass is a no-op on a module like this one.

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
