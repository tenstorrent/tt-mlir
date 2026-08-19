// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// The fused op's device kernel enforces shape constraints on the addcmul
// operands: with the matmul output being [M, N], the residual (ternary_a) must
// match it exactly and the gate (ternary_b) must be [1, N] or [M, N].

#dram = #ttnn.buffer_type<dram>
// Activation A[M=32, K=128].
#a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight B[K=128, N=256].
#w = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// [M=32, N=256] output/gate.
#o = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Row-broadcast [1, N=256].
#b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Scalar [1, 1].
#s = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Residual (ternary_a) must match the output [M, N] exactly; a row-broadcast
// [1, N] residual is illegal.
func.func @dit_residual_broadcast(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                                  %res: tensor<1x256xbf16, #b>, %gate: tensor<32x256xbf16, #o>)
    -> tensor<32x256xbf16, #o> {
  // expected-error @+1 {{Residual[-2:](1, 256) must match output [M, N] = (32, 256)}}
  %0 = "ttnn.dit_matmul_addcmul_fused"(%a, %w, %res, %gate) : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>, tensor<1x256xbf16, #b>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %0 : tensor<32x256xbf16, #o>
}

// -----

#dram = #ttnn.buffer_type<dram>
#a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#w = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#o = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#s = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Gate (ternary_b) must be [1, N] or [M, N]; a scalar [1, 1] gate whose last
// dim != N is illegal.
func.func @dit_gate_scalar(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                           %res: tensor<32x256xbf16, #o>, %gate: tensor<1x1xbf16, #s>)
    -> tensor<32x256xbf16, #o> {
  // expected-error @+1 {{Gate[-2:](1, 1) must be [1, N] or [M, N] = (1 or 32, 256)}}
  %0 = "ttnn.dit_matmul_addcmul_fused"(%a, %w, %res, %gate) : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>, tensor<32x256xbf16, #o>, tensor<1x1xbf16, #s>) -> tensor<32x256xbf16, #o>
  return %0 : tensor<32x256xbf16, #o>
}
