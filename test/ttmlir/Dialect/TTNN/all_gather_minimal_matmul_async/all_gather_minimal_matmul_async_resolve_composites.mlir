// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=inline" --split-input-file %s | FileCheck %s --check-prefix=INLINE
// RUN: ttmlir-opt --ttcore-register-device --ttnn-resolve-composites="composite-resolution=force-promote" --split-input-file %s | FileCheck %s --check-prefix=PROMOTE

// Resolution of the `all_gather_minimal_matmul_async` composite emitted by the
// TTIR AllGatherMatmul fusing patterns:
//   - inline        -> the decomposition body (ttnn.all_gather + ttnn.matmul)
//                      is spliced in and the composite is removed.
//   - force-promote -> the composite becomes the typed
//                      ttnn.all_gather_minimal_matmul_async op (the build
//                      callback synthesizes the device and leaves semaphores
//                      unbound for TTNNAllocateDistributedOpSemaphores).

#dram = #ttnn.buffer_type<dram>
// Sharded activation A[M=32, K_local=128] (gathered to K=512 across 4 devices).
#a  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Gathered activation A[M=32, K=512].
#ag = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight B[K=512, N=64].
#w  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Output [M=32, N=64].
#o  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Plain matmul composite.
// INLINE-LABEL: func.func @resolve_matmul
// INLINE: "ttnn.all_gather"
// INLINE: "ttnn.matmul"
// INLINE-NOT: ttcore.composite
// INLINE-NOT: all_gather_minimal_matmul_async
//
// PROMOTE-LABEL: func.func @resolve_matmul
// PROMOTE: "ttnn.all_gather_minimal_matmul_async"
// PROMOTE-NOT: ttcore.composite
func.func @resolve_matmul(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>)
    -> tensor<32x64xbf16, #o> {
  %0 = "ttcore.composite"(%x, %w) <{
      composite_attributes = {all_gather_dim = 1 : si32, cluster_axis = 1 : ui32, has_addcmul = false, has_bias = false},
      composite_name = "all_gather_minimal_matmul_async",
      decomposition = @agmm_matmul_decomp}> : (tensor<32x128xbf16, #a>, tensor<512x64xbf16, #w>) -> tensor<32x64xbf16, #o>
  return %0 : tensor<32x64xbf16, #o>
}
func.func private @agmm_matmul_decomp(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>)
    -> tensor<32x64xbf16, #o> attributes {tt.composite_decomposition} {
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>) -> tensor<32x64xbf16, #o>
  return %1 : tensor<32x64xbf16, #o>
}

// -----

#dram = #ttnn.buffer_type<dram>
#a  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ag = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#w  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#o  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Gated-residual (addcmul) composite: has_addcmul + scalar carried through.
// INLINE-LABEL: func.func @resolve_addcmul
// INLINE: "ttnn.all_gather"
// INLINE: "ttnn.matmul"
// INLINE: "ttnn.multiply"
// INLINE: "ttnn.add"
// INLINE-NOT: all_gather_minimal_matmul_async
//
// PROMOTE-LABEL: func.func @resolve_addcmul
// PROMOTE: "ttnn.all_gather_minimal_matmul_async"
// PROMOTE-SAME: scalar = 1.000000e+00 : f32
// PROMOTE-NOT: ttcore.composite
func.func @resolve_addcmul(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>,
                           %res: tensor<32x64xbf16, #o>, %gate: tensor<32x64xbf16, #o>)
    -> tensor<32x64xbf16, #o> {
  %0 = "ttcore.composite"(%x, %w, %res, %gate) <{
      composite_attributes = {all_gather_dim = 1 : si32, cluster_axis = 1 : ui32, has_addcmul = true, has_bias = false, scalar = 1.000000e+00 : f32},
      composite_name = "all_gather_minimal_matmul_async",
      decomposition = @agmm_addcmul_decomp}> : (tensor<32x128xbf16, #a>, tensor<512x64xbf16, #w>, tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  return %0 : tensor<32x64xbf16, #o>
}
func.func private @agmm_addcmul_decomp(%x: tensor<32x128xbf16, #a>, %w: tensor<512x64xbf16, #w>,
                                       %res: tensor<32x64xbf16, #o>, %gate: tensor<32x64xbf16, #o>)
    -> tensor<32x64xbf16, #o> attributes {tt.composite_decomposition} {
  %0 = "ttnn.all_gather"(%x) <{all_gather_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x128xbf16, #a>) -> tensor<32x512xbf16, #ag>
  %1 = "ttnn.matmul"(%0, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x512xbf16, #ag>, tensor<512x64xbf16, #w>) -> tensor<32x64xbf16, #o>
  %2 = "ttnn.multiply"(%gate, %1) : (tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  %3 = "ttnn.add"(%res, %2) : (tensor<32x64xbf16, #o>, tensor<32x64xbf16, #o>) -> tensor<32x64xbf16, #o>
  return %3 : tensor<32x64xbf16, #o>
}
