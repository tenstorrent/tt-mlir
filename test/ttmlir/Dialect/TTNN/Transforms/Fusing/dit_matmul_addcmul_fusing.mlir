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
// Bias / row-broadcast [1, N=256]. Doubles as the [M=1, N=256] projection
// output in the row-broadcast negative cases.
#b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Activation A[M=1, K=128] for the row-broadcast negative cases.
#a1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Scalar [1, 1] operand for the scalar-gate negative case.
#s = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Rank-3 [1, M=32, N=256] operands/output, the shape the adaLN epilogue runs at
// while the projection stays 2D.
#o3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Rank-3 row-broadcast gate [1, 1, N=256].
#g3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// f32 counterparts, for the fp32-epilogue negative case.
#o3f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#g3f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 + d1, d2), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
// 2D f32 [M=32, N=256], the typecast result in the fp32-epilogue negative case.
#of32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
// Flattened [1, M*N = 8192] operands, for the element-reordering reshape case.
#flat = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x256x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

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

// A unit-outer-dim reshape between the projection and the multiply is folded
// away and replayed on the fused result. This is the shape WAN 2.2 DiT emits:
// `ttir.dot_general` lowers to a 2D matmul while the adaLN epilogue stays
// rank-3, so linear -> reshape([M, N] -> [1, M, N]) -> multiply -> add.
// CHECK-LABEL: func.func @dit_linear_addcmul_through_reshape
func.func @dit_linear_addcmul_through_reshape(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>, %bias: tensor<1x256xbf16, #b>,
                                              %gate: tensor<1x1x256xbf16, #g3>, %res: tensor<1x32x256xbf16, #o3>)
    -> tensor<1x32x256xbf16, #o3> {
  // The fused op keeps the projection's 2D shape (the device kernel derives the
  // output rank from operand `a`) and the reshape moves after it.
  // CHECK: %[[FUSED:.*]] = "ttnn.dit_matmul_addcmul_fused"
  // CHECK-SAME: -> tensor<32x256xbf16
  // CHECK: "ttnn.reshape"(%[[FUSED]])
  // CHECK-SAME: -> tensor<1x32x256xbf16
  // CHECK-NOT: "ttnn.linear"
  // CHECK-NOT: "ttnn.multiply"
  // CHECK-NOT: "ttnn.add"
  %0 = "ttnn.linear"(%a, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>, tensor<1x256xbf16, #b>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 32 : i32, 256 : i32]}> : (tensor<32x256xbf16, #o>) -> tensor<1x32x256xbf16, #o3>
  %2 = "ttnn.multiply"(%1, %gate) : (tensor<1x32x256xbf16, #o3>, tensor<1x1x256xbf16, #g3>) -> tensor<1x32x256xbf16, #o3>
  %3 = "ttnn.add"(%res, %2) : (tensor<1x32x256xbf16, #o3>, tensor<1x32x256xbf16, #o3>) -> tensor<1x32x256xbf16, #o3>
  return %3 : tensor<1x32x256xbf16, #o3>
}

// A reshape that redistributes elements ([32, 256] -> [1, 8192]) is not folded:
// it does not commute with the epilogue, so the addcmul operands would no longer
// line up with the matmul's [M, N] output.
// CHECK-LABEL: func.func @dit_no_fuse_reordering_reshape
func.func @dit_no_fuse_reordering_reshape(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>,
                                          %gate: tensor<1x8192xbf16, #flat>, %res: tensor<1x8192xbf16, #flat>)
    -> tensor<1x8192xbf16, #flat> {
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 8192 : i32]}> : (tensor<32x256xbf16, #o>) -> tensor<1x8192xbf16, #flat>
  %2 = "ttnn.multiply"(%1, %gate) : (tensor<1x8192xbf16, #flat>, tensor<1x8192xbf16, #flat>) -> tensor<1x8192xbf16, #flat>
  %3 = "ttnn.add"(%res, %2) : (tensor<1x8192xbf16, #flat>, tensor<1x8192xbf16, #flat>) -> tensor<1x8192xbf16, #flat>
  return %3 : tensor<1x8192xbf16, #flat>
}

// A multiply that broadcasts the projection's rows ([M=1, N] projection against
// an [M, N] gate) is not fused: [M, N] belongs to the projection, and the fused
// op cannot widen it to the add's shape.
// CHECK-LABEL: func.func @dit_no_fuse_row_broadcast_proj
func.func @dit_no_fuse_row_broadcast_proj(%a: tensor<1x128xbf16, #a1>, %w: tensor<128x256xbf16, #w>,
                                          %gate: tensor<32x256xbf16, #o>, %res: tensor<32x256xbf16, #o>)
    -> tensor<32x256xbf16, #o> {
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<1x128xbf16, #a1>, tensor<128x256xbf16, #w>) -> tensor<1x256xbf16, #b>
  %1 = "ttnn.multiply"(%0, %gate) : (tensor<1x256xbf16, #b>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  %2 = "ttnn.add"(%res, %1) : (tensor<32x256xbf16, #o>, tensor<32x256xbf16, #o>) -> tensor<32x256xbf16, #o>
  return %2 : tensor<32x256xbf16, #o>
}

// The same row broadcast behind a unit-outer-dim reshape. Folding the reshape
// here would build the fused op at the projection's [1, 256] type and then
// replay the reshape onto [1, 32, 256], which is not even the same number of
// elements.
// CHECK-LABEL: func.func @dit_no_fuse_row_broadcast_proj_through_reshape
func.func @dit_no_fuse_row_broadcast_proj_through_reshape(%a: tensor<1x128xbf16, #a1>, %w: tensor<128x256xbf16, #w>,
                                                          %gate: tensor<1x32x256xbf16, #o3>, %res: tensor<1x32x256xbf16, #o3>)
    -> tensor<1x32x256xbf16, #o3> {
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.reshape"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.matmul"(%a, %w) <{transpose_a = false, transpose_b = false}> : (tensor<1x128xbf16, #a1>, tensor<128x256xbf16, #w>) -> tensor<1x256xbf16, #b>
  %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 1 : i32, 256 : i32]}> : (tensor<1x256xbf16, #b>) -> tensor<1x1x256xbf16, #g3>
  %2 = "ttnn.multiply"(%1, %gate) : (tensor<1x1x256xbf16, #g3>, tensor<1x32x256xbf16, #o3>) -> tensor<1x32x256xbf16, #o3>
  %3 = "ttnn.add"(%res, %2) : (tensor<1x32x256xbf16, #o3>, tensor<1x32x256xbf16, #o3>) -> tensor<1x32x256xbf16, #o3>
  return %3 : tensor<1x32x256xbf16, #o3>
}

// An fp32 epilogue on a bf16 projection is not fused: the typecast sits between
// the projection and the reshape, so the reshape no longer lands on the matmul.
// This is what WAN 2.2 DiT emits today; enabling `_patch_adaln_modulation_bf16`
// on the model keeps the epilogue in bf16 and produces the fusable form above.
// CHECK-LABEL: func.func @dit_no_fuse_fp32_epilogue
func.func @dit_no_fuse_fp32_epilogue(%a: tensor<32x128xbf16, #a>, %w: tensor<128x256xbf16, #w>, %bias: tensor<1x256xbf16, #b>,
                                     %gate: tensor<1x1x256xf32, #g3f32>, %res: tensor<1x32x256xf32, #o3f32>)
    -> tensor<1x32x256xf32, #o3f32> {
  // CHECK: "ttnn.linear"
  // CHECK: "ttnn.typecast"
  // CHECK: "ttnn.multiply"
  // CHECK: "ttnn.add"
  // CHECK-NOT: dit_matmul_addcmul_fused
  %0 = "ttnn.linear"(%a, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x256xbf16, #w>, tensor<1x256xbf16, #b>) -> tensor<32x256xbf16, #o>
  %1 = "ttnn.typecast"(%0) : (tensor<32x256xbf16, #o>) -> tensor<32x256xf32, #of32>
  %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 32 : i32, 256 : i32]}> : (tensor<32x256xf32, #of32>) -> tensor<1x32x256xf32, #o3f32>
  %3 = "ttnn.multiply"(%2, %gate) : (tensor<1x32x256xf32, #o3f32>, tensor<1x1x256xf32, #g3f32>) -> tensor<1x32x256xf32, #o3f32>
  %4 = "ttnn.add"(%res, %3) : (tensor<1x32x256xf32, #o3f32>, tensor<1x32x256xf32, #o3f32>) -> tensor<1x32x256xf32, #o3f32>
  return %4 : tensor<1x32x256xf32, #o3f32>
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
