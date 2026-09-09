// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// When the typed ttnn.flash_mla_prefill op cannot be promoted (here:
// TTNNResolveComposites runs without a system_desc, so OpModel validation is
// unavailable), the composite must fall back to inlining its decomposition
// function -- the full primitive attention lowering. This pathway is what a
// no-OpModel build relies on.

// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=inline" %s | FileCheck %s

// CHECK-LABEL: func.func @flash_mla_prefill_decompose
// The composite is replaced by its primitive decomposition body...
// CHECK-DAG: "ttir.slice_static"
// CHECK-DAG: "ttir.permute"
// CHECK-DAG: "ttir.matmul"
// CHECK-DAG: "ttir.softmax"
// CHECK-DAG: "ttir.where"
// ...and neither the composite nor the typed op survive.
// CHECK-NOT: "ttcore.composite"
// CHECK-NOT: "ttnn.flash_mla_prefill"
module {
  func.func @flash_mla_prefill_decompose(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    %0 = "ttcore.composite"(%q, %k) <{composite_name = "flash_mla_prefill", decomposition = @flash_mla_prefill_decomp, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = false, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // The full primitive decomposition synthesized by the StableHLO conversion:
  // V = K[..., :head_dim_v] (latent), QK^T with GQA head-folding reshape, scale,
  // causal mask (arange/ge/where), softmax, and the probs @ V matmul.
  func.func private @flash_mla_prefill_decomp(%arg0: tensor<1x16x32x128xbf16>, %arg1: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    %0 = "ttir.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x32x64xbf16>
    %1 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 512 : i32, 128 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x1x512x128xbf16>
    %2 = "ttir.permute"(%arg1) <{permutation = array<i64: 0, 1, 3, 2>}> : (tensor<1x1x32x128xbf16>) -> tensor<1x1x128x32xbf16>
    %3 = "ttir.matmul"(%1, %2) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x512x128xbf16>, tensor<1x1x128x32xbf16>) -> tensor<1x1x512x32xbf16>
    %4 = "ttir.reshape"(%3) <{shape = [1 : i32, 16 : i32, 32 : i32, 32 : i32]}> : (tensor<1x1x512x32xbf16>) -> tensor<1x16x32x32xbf16>
    %5 = "ttir.full"() <{fill_value = 0.0883883461 : f32, shape = array<i32: 1, 16, 32, 32>}> : () -> tensor<1x16x32x32xbf16>
    %6 = "ttir.multiply"(%4, %5) : (tensor<1x16x32x32xbf16>, tensor<1x16x32x32xbf16>) -> tensor<1x16x32x32xbf16>
    %7 = "ttir.arange"() <{arange_dimension = 2 : i64, end = 32 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1x1x32x32xbf16>
    %8 = "ttir.arange"() <{arange_dimension = 3 : i64, end = 32 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<1x1x32x32xbf16>
    %9 = "ttir.ge"(%7, %8) : (tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16>
    %10 = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32: 1, 1, 32, 32>}> : () -> tensor<1x1x32x32xbf16>
    %11 = "ttir.full"() <{fill_value = 0xFF800000 : f32, shape = array<i32: 1, 1, 32, 32>}> : () -> tensor<1x1x32x32xbf16>
    %12 = "ttir.where"(%9, %10, %11) : (tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x32xbf16>
    %13 = "ttir.add"(%6, %12) : (tensor<1x16x32x32xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x32xbf16>
    %14 = "ttir.softmax"(%13) <{dimension = 3 : si32, numericStable = true}> : (tensor<1x16x32x32xbf16>) -> tensor<1x16x32x32xbf16>
    %15 = "ttir.reshape"(%14) <{shape = [1 : i32, 1 : i32, 512 : i32, 32 : i32]}> : (tensor<1x16x32x32xbf16>) -> tensor<1x1x512x32xbf16>
    %16 = "ttir.matmul"(%15, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x512x32xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x512x64xbf16>
    %17 = "ttir.reshape"(%16) <{shape = [1 : i32, 16 : i32, 32 : i32, 64 : i32]}> : (tensor<1x1x512x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %17 : tensor<1x16x32x64xbf16>
  }
}
