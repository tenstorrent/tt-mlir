// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s

// Fuses a reduce_scatter consuming a matmul/linear (optionally with a
// gated-residual addcmul epilogue) at the TTIR level into a
//   ttcore.composite "minimal_matmul_strided_reduce_scatter_async"
// whose decomposition body holds the primitive form. Promotion to
// ttnn.minimal_matmul_strided_reduce_scatter_async (or inlining the body)
// happens later via TTNNResolveComposites.
//
// composite_attributes print in sorted key order, so the CHECK-SAME lines are
// ordered: cluster_axis, has_addcmul, has_bias, scatter_dim[, scalar],
// composite_name.

// reduce_scatter(matmul(x, W)) -> composite. has_bias/has_addcmul both false.
// CHECK-LABEL: func.func @matmul_reduce_scatter
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = false
// CHECK-SAME: has_bias = false
// CHECK-SAME: composite_name = "minimal_matmul_strided_reduce_scatter_async"
func.func @matmul_reduce_scatter(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>)
    -> tensor<32x32xbf16> {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// -----

// reduce_scatter(linear(x, W, bias)) -> composite, bias carried through.
// CHECK-LABEL: func.func @linear_reduce_scatter
// CHECK: "ttcore.composite"
// CHECK-SAME: has_bias = true
// CHECK-SAME: composite_name = "minimal_matmul_strided_reduce_scatter_async"
func.func @linear_reduce_scatter(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>, %bias: tensor<1x64xbf16>)
    -> tensor<32x32xbf16> {
  %0 = "ttir.linear"(%x, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// -----

// transpose_b set -> not fused (fused op does not model transpose).
// CHECK-LABEL: func.func @no_fuse_transpose
// CHECK: "ttir.matmul"
// CHECK: "ttir.reduce_scatter"
// CHECK-NOT: ttcore.composite
func.func @no_fuse_transpose(%x: tensor<32x128xbf16>, %w: tensor<64x128xbf16>)
    -> tensor<32x32xbf16> {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = true}> : (tensor<32x128xbf16>, tensor<64x128xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// -----

// matmul result has a second use -> not fused (would duplicate the matmul).
// CHECK-LABEL: func.func @no_fuse_multiuse
// CHECK-NOT: ttcore.composite
func.func @no_fuse_multiuse(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>)
    -> (tensor<32x32xbf16>, tensor<32x64xbf16>) {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  return %1, %0 : tensor<32x32xbf16>, tensor<32x64xbf16>
}

// -----

// Gated-residual epilogue: residual + gate * reduce_scatter(matmul(x, W)) folds
// the whole thing (matmul + reduce_scatter + multiply + add) into one composite,
// with residual/gate mapped to the addcmul operands and scalar = 1.0.
// CHECK-LABEL: func.func @matmul_reduce_scatter_addcmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = true
// CHECK-SAME: scalar = 1.000000e+00 : f32
// CHECK-SAME: composite_name = "minimal_matmul_strided_reduce_scatter_async"
func.func @matmul_reduce_scatter_addcmul(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>,
                                         %gate: tensor<32x32xbf16>, %res: tensor<32x32xbf16>)
    -> tensor<32x32xbf16> {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  %2 = "ttir.multiply"(%1, %gate) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %3 = "ttir.add"(%res, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %3 : tensor<32x32xbf16>
}

// -----

// linear (with bias) + gated-residual epilogue, multiply operands reversed:
// exercises the linear path and both commutative branches.
// CHECK-LABEL: func.func @linear_reduce_scatter_addcmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = true
// CHECK-SAME: has_bias = true
// CHECK-SAME: composite_name = "minimal_matmul_strided_reduce_scatter_async"
func.func @linear_reduce_scatter_addcmul(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>, %bias: tensor<1x64xbf16>,
                                         %gate: tensor<32x32xbf16>, %res: tensor<32x32xbf16>)
    -> tensor<32x32xbf16> {
  %0 = "ttir.linear"(%x, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  %2 = "ttir.multiply"(%gate, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %3 = "ttir.add"(%2, %res) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %3 : tensor<32x32xbf16>
}

// -----

// The generated decomposition function is emitted and marked so fusing never
// recurses into it.
// CHECK: func.func private @minimal_matmul_strided_reduce_scatter_async_decomp
// CHECK-SAME: attributes {tt.composite_decomposition}
func.func @emit_decomp(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>)
    -> tensor<32x32xbf16> {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
