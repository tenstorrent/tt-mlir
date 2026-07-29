// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing %s | FileCheck %s

// Fuses the DiT adaLN gated-residual epilogue on TTNN ops
//   out = residual + gate * (matmul(a, b)[+bias])
// into a single ttnn.dit_matmul_addcmul_fused. The matmul/multiply/add are
// consumed and folded into the one op, mirroring tt-metal's experimental
// fused kernel.

#dram = #ttnn.buffer_type<dram>
// Activation A[M=32, K=128].
#a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight B[K=128, N=256].
#w = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight transposed B[N=256, K=128] for the transpose negative case.
#wt = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// [M=32, N=256] operands/output (proj, gate, residual, result).
#o = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Bias / row-broadcast [1, N=256].
#b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Scalar [1, 1] operand for the scalar-gate negative case.
#s = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// matmul + gate*proj + residual -> fused op. Multiply is the add's LHS operand
// here, exercising the primary (add-lhs) branch of the commutative match.
// CHECK-LABEL: func.func @dit_matmul_addcmul
func.func @dit_matmul_addcmul(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                              %gate: tensor<32x256xbf16, #o>, %res: tensor<32x256xbf16, #o>)
    -> tensor<32x256xbf16, #o> {
  // CHECK: "ttnn.dit_matmul_addcmul_fused"
  // CHECK-NOT: "ttnn.matmul"
  // CHECK-NOT: "ttnn.multiply"
  // CHECK-NOT: "ttnn.add"
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.multiply"(%0, %gate) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%1, %res) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %2 : tensor<32x256xbf16, #o>
}

// linear (with bias) variant fuses too; multiply/add operands in reversed order.
// The bias rides into the fused op as its optional 5th operand.
// CHECK-LABEL: func.func @dit_linear_addcmul
func.func @dit_linear_addcmul(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>, %bias: tensor<1x256xbf16, #b>,
                              %gate: tensor<32x256xbf16, #o>, %res: tensor<32x256xbf16, #o>)
    -> tensor<32x256xbf16, #o> {
  // CHECK: "ttnn.dit_matmul_addcmul_fused"
  // CHECK-NOT: "ttnn.linear"
  %0 = "ttnn.linear"(%a, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>, tensor<1x256xbf16, #b>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.multiply"(%gate, %0) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%res, %1) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %2 : tensor<32x256xbf16, #o>
}

// transpose_b set -> not fused (fused op does not model transpose).
// CHECK-LABEL: func.func @dit_no_fuse_transpose
func.func @dit_no_fuse_transpose(%a: tensor<32x128xbf16, #a>, %w: tensor<256x128xbf16, #wt>,
                                 %gate: tensor<32x256xbf16, #o>, %res: tensor<32x256xbf16, #o>)
    -> tensor<32x256xbf16, #o> {
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = true}> : (tensor<32x128xbf16, #a>, tensor<256x128xbf16, #wt>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.multiply"(%0, %gate) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%res, %1) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %2 : tensor<32x256xbf16, #o>
}

// matmul result has a second use -> not fused (would duplicate the matmul).
// CHECK-LABEL: func.func @dit_no_fuse_multiuse
func.func @dit_no_fuse_multiuse(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                                %gate: tensor<32x256xbf16, #o>, %res: tensor<32x256xbf16, #o>)
    -> (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) {
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.multiply"(%0, %gate) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%res, %1) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %0, %2 : tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>
}

// Broadcast residual [1, N] -> not fused: the fused op's ternary_a (residual)
// must match the output [M, N] exactly, so a row-broadcast residual is illegal.
// (Regressed a StableHLO autoencoder model on silicon.)
// CHECK-LABEL: func.func @dit_no_fuse_broadcast_residual
func.func @dit_no_fuse_broadcast_residual(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                                          %gate: tensor<32x256xbf16, #o>, %res: tensor<1x256xbf16, #b>)
    -> tensor<32x256xbf16, #o> {
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.multiply"(%0, %gate) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%1, %res) : (tensor<32x256xbf16, #o>, tensor<1x256xbf16, #b>) -> tensor<32x256xbf16, #o>
  return %2 : tensor<32x256xbf16, #o>
}

// Scalar gate [1, 1] -> not fused: the fused op's ternary_b (gate) must be
// [1, N] or [M, N], so a scalar gate whose last dim != N is illegal.
// (Regressed an SDPA optimizer fusing model.)
// CHECK-LABEL: func.func @dit_no_fuse_scalar_gate
func.func @dit_no_fuse_scalar_gate(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                                   %gate: tensor<1x1xbf16, #s>, %res: tensor<32x256xbf16, #o>)
    -> tensor<32x256xbf16, #o> {
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.multiply"(%0, %gate) : (tensor<32x256xbf16, #o>, tensor<1x1xbf16, #s>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%1, %res) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %2 : tensor<32x256xbf16, #o>
}
