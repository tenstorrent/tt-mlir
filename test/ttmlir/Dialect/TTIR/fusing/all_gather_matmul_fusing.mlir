// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing="enable-all-gather-matmul-fusion=true" %s | FileCheck %s

// Fuses an all_gather feeding a matmul/linear (optionally with a gated-residual
// addcmul epilogue) at the TTIR level into a
//   ttcore.composite "all_gather_minimal_matmul_async"
// whose decomposition body holds the primitive form. Promotion to
// ttnn.all_gather_minimal_matmul_async (or inlining the body) happens later via
// TTNNResolveComposites.
//
// composite_attributes print in sorted key order, so the CHECK-SAME lines are
// ordered: all_gather_dim, cluster_axis, has_addcmul, has_bias[, scalar],
// composite_name.

// matmul(all_gather(x), W) -> composite. has_bias/has_addcmul both false.
// CHECK-LABEL: func.func @all_gather_matmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = false
// CHECK-SAME: has_bias = false
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
func.func @all_gather_matmul(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>) -> tensor<32x64xbf16>
  return %1 : tensor<32x64xbf16>
}

// linear(all_gather(x), W, bias) -> composite, bias carried through.
// CHECK-LABEL: func.func @all_gather_linear
// CHECK: "ttcore.composite"
// CHECK-SAME: has_bias = true
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
func.func @all_gather_linear(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>, %bias: tensor<1x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
  return %1 : tensor<32x64xbf16>
}

// transpose_b set -> not fused (fused op does not model transpose).
// CHECK-LABEL: func.func @no_fuse_transpose
// CHECK: "ttir.all_gather"
// CHECK: "ttir.matmul"
// CHECK-NOT: ttcore.composite
func.func @no_fuse_transpose(%x: tensor<32x128xbf16>, %w: tensor<64x512xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<64x512xbf16>) -> tensor<32x64xbf16>
  return %1 : tensor<32x64xbf16>
}

// all_gather result has a second use -> not fused (would duplicate the gather).
// CHECK-LABEL: func.func @no_fuse_multiuse
// CHECK-NOT: ttcore.composite
func.func @no_fuse_multiuse(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>)
    -> (tensor<32x64xbf16>, tensor<32x512xbf16>) {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>) -> tensor<32x64xbf16>
  return %1, %0 : tensor<32x64xbf16>, tensor<32x512xbf16>
}

// Gated-residual epilogue: residual + gate * matmul(all_gather(x), W) folds the
// whole thing (gather + matmul + multiply + add) into one composite, with
// residual/gate mapped to the addcmul operands and scalar = 1.0.
// CHECK-LABEL: func.func @all_gather_matmul_addcmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = true
// CHECK-SAME: scalar = 1.000000e+00 : f32
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
func.func @all_gather_matmul_addcmul(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>,
                                     %gate: tensor<32x64xbf16>, %res: tensor<32x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>) -> tensor<32x64xbf16>
  %2 = "ttir.multiply"(%1, %gate) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  %3 = "ttir.add"(%res, %2) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %3 : tensor<32x64xbf16>
}

// linear (with bias) + gated-residual epilogue, multiply operands reversed:
// exercises the linear path and both commutative branches.
// CHECK-LABEL: func.func @all_gather_linear_addcmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = true
// CHECK-SAME: has_bias = true
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
func.func @all_gather_linear_addcmul(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>, %bias: tensor<1x64xbf16>,
                                     %gate: tensor<32x64xbf16>, %res: tensor<32x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
  %2 = "ttir.multiply"(%gate, %1) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  %3 = "ttir.add"(%2, %res) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %3 : tensor<32x64xbf16>
}

// The generated decomposition function is emitted and marked so fusing never
// recurses into it.
// CHECK: func.func private @all_gather_minimal_matmul_async_decomp
// CHECK-SAME: attributes {tt.composite_decomposition}
