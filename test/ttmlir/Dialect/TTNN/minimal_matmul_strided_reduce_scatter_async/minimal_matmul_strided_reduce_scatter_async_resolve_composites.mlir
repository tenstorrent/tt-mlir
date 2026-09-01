// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=inline" --split-input-file %s | FileCheck %s --check-prefix=INLINE
// RUN: ttmlir-opt --ttcore-register-device --ttnn-resolve-composites="composite-resolution=force-promote" --split-input-file %s | FileCheck %s --check-prefix=PROMOTE

// Resolution of the `minimal_matmul_strided_reduce_scatter_async` composite.
// A 2D `[M, K]` activation must be unsqueezed to rank 4 before the typed
// ttnn op (tt-metal indexes scatter dim 3) and the result squeezed back.

// INLINE-LABEL: func.func @rank2_matmul_reduce_scatter
// INLINE-NOT: ttcore.composite
// INLINE-NOT: minimal_matmul_strided_reduce_scatter_async
// INLINE: "ttir.matmul"
// INLINE: "ttir.reduce_scatter"

// PROMOTE-LABEL: func.func @rank2_matmul_reduce_scatter
// PROMOTE: "ttnn.reshape"
// PROMOTE-SAME: shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]
// PROMOTE: "ttnn.minimal_matmul_strided_reduce_scatter_async"
// PROMOTE-SAME: dim = 3
// PROMOTE: "ttnn.reshape"
// PROMOTE-SAME: shape = [32 : i32, 32 : i32]
func.func @rank2_matmul_reduce_scatter(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>)
    -> tensor<32x32xbf16> {
  %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %0 = "ttcore.composite"(%x, %w) <{
      composite_name = "minimal_matmul_strided_reduce_scatter_async",
      decomposition = @rank2_decomp,
      composite_attributes = {
        cluster_axis = 1 : ui32,
        has_addcmul = false,
        has_bias = false,
        scatter_dim = 1 : si32
      }
    }> : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x32xbf16>
  return %0 : tensor<32x32xbf16>
}

func.func private @rank2_decomp(%x: tensor<32x128xbf16>, %w: tensor<128x64xbf16>)
    -> tensor<32x32xbf16> attributes {tt.composite_decomposition} {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 1 : si32}> : (tensor<32x64xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// -----

// Already rank-4: no leading reshape around the fused op.

// PROMOTE-LABEL: func.func @rank4_matmul_reduce_scatter
// PROMOTE-NOT: "ttnn.reshape"
// PROMOTE: "ttnn.minimal_matmul_strided_reduce_scatter_async"
// PROMOTE-SAME: dim = 3
func.func @rank4_matmul_reduce_scatter(%x: tensor<1x1x32x128xbf16>, %w: tensor<128x64xbf16>)
    -> tensor<1x1x32x32xbf16> {
  %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %0 = "ttcore.composite"(%x, %w) <{
      composite_name = "minimal_matmul_strided_reduce_scatter_async",
      decomposition = @rank4_decomp,
      composite_attributes = {
        cluster_axis = 1 : ui32,
        has_addcmul = false,
        has_bias = false,
        scatter_dim = 3 : si32
      }
    }> : (tensor<1x1x32x128xbf16>, tensor<128x64xbf16>) -> tensor<1x1x32x32xbf16>
  return %0 : tensor<1x1x32x32xbf16>
}

func.func private @rank4_decomp(%x: tensor<1x1x32x128xbf16>, %w: tensor<128x64xbf16>)
    -> tensor<1x1x32x32xbf16> attributes {tt.composite_decomposition} {
  %0 = "ttir.matmul"(%x, %w) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x32x128xbf16>, tensor<128x64xbf16>) -> tensor<1x1x32x64xbf16>
  %1 = "ttir.reduce_scatter"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x32x64xbf16>) -> tensor<1x1x32x32xbf16>
  return %1 : tensor<1x1x32x32xbf16>
}
