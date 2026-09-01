// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-fusing %s | FileCheck %s
// RUN: ttmlir-opt --ttir-fusing='enable-all-gather-matmul-fusion=false' %s | FileCheck %s --check-prefix=NOAGMM

// Fuses an all_gather feeding a matmul/linear (optionally with a gated-residual
// addcmul epilogue) at the TTIR level into a
//   ttcore.composite "all_gather_minimal_matmul_async"
// whose decomposition body holds the primitive form. Promotion to
// ttnn.all_gather_minimal_matmul_async (or inlining the body) happens later via
// TTNNResolveComposites.
//
// composite_attributes print in sorted key order, so the CHECK-SAME lines are
// ordered: all_gather_dim, [chunks,] cluster_axis, has_addcmul, has_bias[, scalar],
// composite_name. `chunks` is omitted when it is 1.

// matmul(all_gather(x), W) -> composite. has_bias/has_addcmul both false.
// CHECK-LABEL: func.func @all_gather_matmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = false
// CHECK-SAME: has_bias = false
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
// NOAGMM-LABEL: func.func @all_gather_matmul
// NOAGMM: "ttir.all_gather"
// NOAGMM: "ttir.matmul"
// NOAGMM-NOT: ttcore.composite
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

// transpose_b is folded by permuting W [N, K] -> [K, N] then fusing.
// CHECK-LABEL: func.func @fuse_transpose_b
// CHECK: "ttir.permute"
// CHECK: "ttcore.composite"
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
func.func @fuse_transpose_b(%x: tensor<32x128xbf16>, %w: tensor<64x512xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<64x512xbf16>) -> tensor<32x64xbf16>
  return %1 : tensor<32x64xbf16>
}

// transpose_a is still not fused (the kernel gathers A's last dim).
// CHECK-LABEL: func.func @no_fuse_transpose_a
// CHECK: "ttir.all_gather"
// CHECK: "ttir.matmul"
// CHECK-NOT: ttcore.composite
func.func @no_fuse_transpose_a(%x: tensor<32x128xbf16>, %w: tensor<32x64xbf16>)
    -> tensor<512x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = true, transpose_b = false}> : (tensor<32x512xbf16>, tensor<32x64xbf16>) -> tensor<512x64xbf16>
  return %1 : tensor<512x64xbf16>
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
// residual/gate mapped to the addcmul operands and scalar = 1.0. The fused
// kernel applies the gate per-channel, so it must be row-broadcast (`[1, N]`).
// CHECK-LABEL: func.func @all_gather_matmul_addcmul
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = true
// CHECK-SAME: scalar = 1.000000e+00 : f32
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
func.func @all_gather_matmul_addcmul(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>,
                                     %gate: tensor<1x64xbf16>, %res: tensor<32x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>) -> tensor<32x64xbf16>
  %2 = "ttir.multiply"(%1, %gate) : (tensor<32x64xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
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
                                     %gate: tensor<1x64xbf16>, %res: tensor<32x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
  %2 = "ttir.multiply"(%gate, %1) : (tensor<1x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  %3 = "ttir.add"(%2, %res) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %3 : tensor<32x64xbf16>
}

// DiT gated residual: linear -> reshape [M, N] -> [1, M, N] -> mul(gate) -> add.
// Must fuse with has_addcmul, not stop at a plain AGMM. Composite is 2D; a
// reshape restores [1, M, N] for the original add's users.
// CHECK-LABEL: func.func @all_gather_linear_addcmul_reshape
// CHECK: "ttir.permute"
// CHECK: "ttcore.composite"
// CHECK-SAME: has_addcmul = true
// CHECK-SAME: has_bias = true
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
// CHECK: "ttir.reshape"
func.func @all_gather_linear_addcmul_reshape(%x: tensor<32x128xbf16>, %w: tensor<64x512xbf16>, %bias: tensor<1x64xbf16>,
                                             %gate: tensor<1x1x64xbf16>, %res: tensor<1x32x64xbf16>)
    -> tensor<1x32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = true}> : (tensor<32x512xbf16>, tensor<64x512xbf16>, tensor<1x64xbf16>) -> tensor<32x64xbf16>
  %2 = "ttir.reshape"(%1) {shape = [1 : i32, 32 : i32, 64 : i32]} : (tensor<32x64xbf16>) -> tensor<1x32x64xbf16>
  %3 = "ttir.multiply"(%2, %gate) : (tensor<1x32x64xbf16>, tensor<1x1x64xbf16>) -> tensor<1x32x64xbf16>
  %4 = "ttir.add"(%res, %3) : (tensor<1x32x64xbf16>, tensor<1x32x64xbf16>) -> tensor<1x32x64xbf16>
  return %4 : tensor<1x32x64xbf16>
}

// A full `[M, N]` gate must NOT fuse: the fused addcmul epilogue applies the
// gate per-channel (broadcast across the M/row dim), so a full gate would be
// silently collapsed to its first row. The guard leaves the primitive
// matmul + multiply + add in place.
// CHECK-LABEL: func.func @no_fuse_full_gate
// CHECK: "ttir.matmul"
// CHECK: "ttir.multiply"
// CHECK: "ttir.add"
// CHECK-NOT: ttcore.composite
func.func @no_fuse_full_gate(%x: tensor<32x128xbf16>, %w: tensor<512x64xbf16>,
                             %gate: tensor<32x64xbf16>, %res: tensor<32x64xbf16>)
    -> tensor<32x64xbf16> {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x64xbf16>) -> tensor<32x64xbf16>
  %2 = "ttir.multiply"(%1, %gate) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  %3 = "ttir.add"(%res, %2) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
  return %3 : tensor<32x64xbf16>
}

// Wan fused QKV: one projection to 3*head_dim, then three equal last-dim
// slices. Fuses as chunks=3 and replaces the slices with composite results.
// CHECK-LABEL: func.func @all_gather_linear_qkv_chunks
// CHECK: "ttcore.composite"
// CHECK-SAME: chunks = 3 : si32
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
// CHECK-NOT: "ttir.slice_static"
func.func @all_gather_linear_qkv_chunks(%x: tensor<32x128xbf16>, %w: tensor<512x192xbf16>, %bias: tensor<1x192xbf16>)
    -> (tensor<32x64xbf16>, tensor<32x64xbf16>, tensor<32x64xbf16>) {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.linear"(%0, %w, %bias) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x192xbf16>, tensor<1x192xbf16>) -> tensor<32x192xbf16>
  %q = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x192xbf16>) -> tensor<32x64xbf16>
  %k = "ttir.slice_static"(%1) <{begins = [0 : i32, 64 : i32], ends = [32 : i32, 128 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x192xbf16>) -> tensor<32x64xbf16>
  %v = "ttir.slice_static"(%1) <{begins = [0 : i32, 128 : i32], ends = [32 : i32, 192 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x192xbf16>) -> tensor<32x64xbf16>
  return %q, %k, %v : tensor<32x64xbf16>, tensor<32x64xbf16>, tensor<32x64xbf16>
}

// Same QKV split after a leading-unit reshape [M, N] -> [1, M, N].
// CHECK-LABEL: func.func @all_gather_linear_qkv_chunks_reshape
// CHECK: "ttcore.composite"
// CHECK-SAME: chunks = 3 : si32
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
// CHECK: "ttir.reshape"
// CHECK: "ttir.reshape"
// CHECK: "ttir.reshape"
// CHECK-NOT: "ttir.slice_static"
func.func @all_gather_linear_qkv_chunks_reshape(%x: tensor<32x128xbf16>, %w: tensor<512x192xbf16>)
    -> (tensor<1x32x64xbf16>, tensor<1x32x64xbf16>, tensor<1x32x64xbf16>) {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x192xbf16>) -> tensor<32x192xbf16>
  %2 = "ttir.reshape"(%1) {shape = [1 : i32, 32 : i32, 192 : i32]} : (tensor<32x192xbf16>) -> tensor<1x32x192xbf16>
  %q = "ttir.slice_static"(%2) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x192xbf16>) -> tensor<1x32x64xbf16>
  %k = "ttir.slice_static"(%2) <{begins = [0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 32 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x192xbf16>) -> tensor<1x32x64xbf16>
  %v = "ttir.slice_static"(%2) <{begins = [0 : i32, 0 : i32, 128 : i32], ends = [1 : i32, 32 : i32, 192 : i32], step = [1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x32x192xbf16>) -> tensor<1x32x64xbf16>
  return %q, %k, %v : tensor<1x32x64xbf16>, tensor<1x32x64xbf16>, tensor<1x32x64xbf16>
}

// Unequal last-dim slices must not set chunks (would be a wrong split).
// The gather+linear still fuses; the slices remain.
// CHECK-LABEL: func.func @no_qkv_chunks_unequal
// CHECK: "ttcore.composite"
// CHECK-SAME: {all_gather_dim = 1 : si32, cluster_axis = 1 : ui32, has_addcmul = false, has_bias = false}
// CHECK-SAME: composite_name = "all_gather_minimal_matmul_async"
// CHECK: "ttir.slice_static"
// CHECK: "ttir.slice_static"
func.func @no_qkv_chunks_unequal(%x: tensor<32x128xbf16>, %w: tensor<512x192xbf16>)
    -> (tensor<32x64xbf16>, tensor<32x128xbf16>) {
  %0 = "ttir.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16>) -> tensor<32x512xbf16>
  %1 = "ttir.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16>, tensor<512x192xbf16>) -> tensor<32x192xbf16>
  %q = "ttir.slice_static"(%1) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x192xbf16>) -> tensor<32x64xbf16>
  %kv = "ttir.slice_static"(%1) <{begins = [0 : i32, 64 : i32], ends = [32 : i32, 192 : i32], step = [1 : i32, 1 : i32]}> : (tensor<32x192xbf16>) -> tensor<32x128xbf16>
  return %q, %kv : tensor<32x64xbf16>, tensor<32x128xbf16>
}

// The generated decomposition function is emitted and marked so fusing never
// recurses into it.
// CHECK: func.func private @all_gather_minimal_matmul_async_decomp
// CHECK-SAME: attributes {tt.composite_decomposition}
