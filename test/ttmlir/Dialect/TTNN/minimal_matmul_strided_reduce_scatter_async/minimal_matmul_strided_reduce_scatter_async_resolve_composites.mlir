// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=inline" --split-input-file %s | FileCheck %s --check-prefix=INLINE
// RUN: ttmlir-opt --ttcore-register-device --ttnn-resolve-composites="composite-resolution=force-promote" --split-input-file %s | FileCheck %s --check-prefix=PROMOTE

// Resolution of the `minimal_matmul_strided_reduce_scatter_async` composite
// emitted by the TTIR MatmulReduceScatter fusing patterns:
//   - inline        -> the decomposition body (ttnn.matmul + ttnn.reduce_scatter)
//                      is spliced in and the composite is removed.
//   - force-promote -> the composite becomes the typed
//                      ttnn.minimal_matmul_strided_reduce_scatter_async op (the
//                      build callback synthesizes the device and leaves
//                      semaphores unbound for
//                      TTNNAllocateDistributedOpSemaphores).

#dram = #ttnn.buffer_type<dram>
// Activation A[M=32, K=128].
#a  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Weight B[K=128, N=64].
#w  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Matmul output [M=32, N=64] (pre-scatter partial).
#mm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// Reduce-scatter output [M=32, N=32] (scattered across 2 devices).
#o  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Plain matmul + reduce_scatter composite.
// INLINE-LABEL: func.func @resolve_matmul
// INLINE: "ttnn.matmul"
// INLINE: "ttnn.reduce_scatter"
// INLINE-NOT: ttcore.composite
// INLINE-NOT: minimal_matmul_strided_reduce_scatter_async
//
// PROMOTE-LABEL: func.func @resolve_matmul
// PROMOTE: "ttnn.minimal_matmul_strided_reduce_scatter_async"
// PROMOTE-NOT: ttcore.composite
func.func @resolve_matmul(%x: tensor<32x128xbf16, #a>, %w: tensor<128x64xbf16, #w>)
    -> tensor<32x32xbf16, #o> {
  %0 = "ttcore.composite"(%x, %w) <{
      composite_attributes = {cluster_axis = 1 : ui32, has_addcmul = false, has_bias = false, scatter_dim = 1 : si32},
      composite_name = "minimal_matmul_strided_reduce_scatter_async",
      decomposition = @mmrs_matmul_decomp}> : (tensor<32x128xbf16, #a>, tensor<128x64xbf16, #w>) -> tensor<32x32xbf16, #o>
  return %0 : tensor<32x32xbf16, #o>
}
func.func private @mmrs_matmul_decomp(%x: tensor<32x128xbf16, #a>, %w: tensor<128x64xbf16, #w>)
    -> tensor<32x32xbf16, #o> attributes {tt.composite_decomposition} {
  %0 = "ttnn.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x64xbf16, #w>) -> tensor<32x64xbf16, #mm>
  %1 = "ttnn.reduce_scatter"(%0) <{reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x64xbf16, #mm>) -> tensor<32x32xbf16, #o>
  return %1 : tensor<32x32xbf16, #o>
}

// -----

#dram = #ttnn.buffer_type<dram>
#a  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#w  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#mm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#o  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Gated-residual (addcmul) composite: has_addcmul + scalar carried through.
// INLINE-LABEL: func.func @resolve_addcmul
// INLINE: "ttnn.matmul"
// INLINE: "ttnn.reduce_scatter"
// INLINE: "ttnn.multiply"
// INLINE: "ttnn.add"
// INLINE-NOT: minimal_matmul_strided_reduce_scatter_async
//
// PROMOTE-LABEL: func.func @resolve_addcmul
// PROMOTE: "ttnn.minimal_matmul_strided_reduce_scatter_async"
// PROMOTE-SAME: scalar = 1.000000e+00 : f32
// PROMOTE-NOT: ttcore.composite
func.func @resolve_addcmul(%x: tensor<32x128xbf16, #a>, %w: tensor<128x64xbf16, #w>,
                           %res: tensor<32x32xbf16, #o>, %gate: tensor<32x32xbf16, #o>)
    -> tensor<32x32xbf16, #o> {
  %0 = "ttcore.composite"(%x, %w, %res, %gate) <{
      composite_attributes = {cluster_axis = 1 : ui32, has_addcmul = true, has_bias = false, scalar = 1.000000e+00 : f32, scatter_dim = 1 : si32},
      composite_name = "minimal_matmul_strided_reduce_scatter_async",
      decomposition = @mmrs_addcmul_decomp}> : (tensor<32x128xbf16, #a>, tensor<128x64xbf16, #w>, tensor<32x32xbf16, #o>, tensor<32x32xbf16, #o>) -> tensor<32x32xbf16, #o>
  return %0 : tensor<32x32xbf16, #o>
}
func.func private @mmrs_addcmul_decomp(%x: tensor<32x128xbf16, #a>, %w: tensor<128x64xbf16, #w>,
                                       %res: tensor<32x32xbf16, #o>, %gate: tensor<32x32xbf16, #o>)
    -> tensor<32x32xbf16, #o> attributes {tt.composite_decomposition} {
  %0 = "ttnn.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16, #a>, tensor<128x64xbf16, #w>) -> tensor<32x64xbf16, #mm>
  %1 = "ttnn.reduce_scatter"(%0) <{reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32, cluster_axis = 1 : ui32}> : (tensor<32x64xbf16, #mm>) -> tensor<32x32xbf16, #o>
  %2 = "ttnn.multiply"(%gate, %1) : (tensor<32x32xbf16, #o>, tensor<32x32xbf16, #o>) -> tensor<32x32xbf16, #o>
  %3 = "ttnn.add"(%res, %2) : (tensor<32x32xbf16, #o>, tensor<32x32xbf16, #o>) -> tensor<32x32xbf16, #o>
  return %3 : tensor<32x32xbf16, #o>
}
