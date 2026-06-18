// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that TTNNResolveComposites inlines RoPE composite decomposition
// functions. This mirrors the decomposition produced by the TTIR RoPE fusing
// patterns: the composite body already contains the lowered TTNN ops.

// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=inline" --split-input-file %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#half = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Complex rotation form: decomposition body slices cos/sin to half-D,
// uses four half-D multiplies + subtract + add + concat.
// After inlining, no composite or rotary_embedding op should remain.
module {
  func.func @rope_resolve_complex_rotation(
      %x: tensor<1x1x4x128xbf16, #full>,
      %cos: tensor<1x1x4x128xbf16, #full>,
      %sin: tensor<1x1x4x128xbf16, #full>)
      -> tensor<1x1x4x128xbf16, #full> {
    // CHECK-LABEL: func.func @rope_resolve_complex_rotation
    // CHECK-NOT: "ttcore.composite"
    // CHECK-NOT: "ttnn.rotary_embedding"
    // CHECK-NOT: "ttnn.neg"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.concat"
    // CHECK-SAME: dim = 3
    // CHECK-NOT: "ttnn.neg"
    %r = "ttcore.composite"(%x, %cos, %sin)
        <{composite_name = "rotary_embedding",
          decomposition = @rope_decomp_complex}>
        : (tensor<1x1x4x128xbf16, #full>,
           tensor<1x1x4x128xbf16, #full>,
           tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    return %r : tensor<1x1x4x128xbf16, #full>
  }

  // Complex rotation decomposition function body (TTNN ops).
  func.func private @rope_decomp_complex(%arg0: tensor<1x1x4x128xbf16, #full>,
                                          %arg1: tensor<1x1x4x128xbf16, #full>,
                                          %arg2: tensor<1x1x4x128xbf16, #full>)
      -> tensor<1x1x4x128xbf16, #full>
      attributes {tt.composite_decomposition} {
    // x_lo = x[:, :, :, :64], x_hi = x[:, :, :, 64:]
    %x_lo = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x64xbf16, #half>
    %x_hi = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x64xbf16, #half>
    // cos_h = cos[:, :, :, :64], sin_h = sin[:, :, :, :64]
    %cos_h = "ttnn.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x64xbf16, #half>
    %sin_h = "ttnn.slice_static"(%arg2) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x64xbf16, #half>
    // first = x_lo*cos_h - x_hi*sin_h
    %lo_cos = "ttnn.multiply"(%x_lo, %cos_h) : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    %hi_sin = "ttnn.multiply"(%x_hi, %sin_h) : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    %first = "ttnn.subtract"(%lo_cos, %hi_sin) : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    // second = x_hi*cos_h + x_lo*sin_h
    %hi_cos = "ttnn.multiply"(%x_hi, %cos_h) : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    %lo_sin = "ttnn.multiply"(%x_lo, %sin_h) : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    %second = "ttnn.add"(%hi_cos, %lo_sin) : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    // result = concat(first, second, dim=3)
    %result = "ttnn.concat"(%first, %second) <{dim = 3 : si32}> : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x128xbf16, #full>
    return %result : tensor<1x1x4x128xbf16, #full>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#full = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#half = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// Rotate_half form: decomposition body uses neg + concat for rotate_half,
// then full-D multiply + add.
module {
  func.func @rope_resolve_rotate_half(
      %x: tensor<1x1x4x128xbf16, #full>,
      %cos: tensor<1x1x4x128xbf16, #full>,
      %sin: tensor<1x1x4x128xbf16, #full>)
      -> tensor<1x1x4x128xbf16, #full> {
    // CHECK-LABEL: func.func @rope_resolve_rotate_half
    // CHECK-NOT: "ttcore.composite"
    // CHECK-NOT: "ttnn.rotary_embedding"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.slice_static"
    // CHECK: "ttnn.neg"
    // CHECK: "ttnn.concat"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.multiply"
    // CHECK: "ttnn.add"
    %r = "ttcore.composite"(%x, %cos, %sin)
        <{composite_name = "rotary_embedding",
          decomposition = @rope_decomp_rotate_half}>
        : (tensor<1x1x4x128xbf16, #full>,
           tensor<1x1x4x128xbf16, #full>,
           tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    return %r : tensor<1x1x4x128xbf16, #full>
  }

  // Rotate_half decomposition function body (TTNN ops).
  func.func private @rope_decomp_rotate_half(%arg0: tensor<1x1x4x128xbf16, #full>,
                                              %arg1: tensor<1x1x4x128xbf16, #full>,
                                              %arg2: tensor<1x1x4x128xbf16, #full>)
      -> tensor<1x1x4x128xbf16, #full>
      attributes {tt.composite_decomposition} {
    // x_lo = x[:, :, :, :64], x_hi = x[:, :, :, 64:]
    %x_lo = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x64xbf16, #half>
    %x_hi = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32, 64 : i32], ends = [1 : i32, 1 : i32, 4 : i32, 128 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x64xbf16, #half>
    // rotate_half(x) = concat(neg(x_hi), x_lo)
    %neg_hi = "ttnn.neg"(%x_hi) : (tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x64xbf16, #half>
    %rotated = "ttnn.concat"(%neg_hi, %x_lo) <{dim = 3 : si32}> : (tensor<1x1x4x64xbf16, #half>, tensor<1x1x4x64xbf16, #half>) -> tensor<1x1x4x128xbf16, #full>
    // result = x * cos + rotate_half(x) * sin
    %x_cos = "ttnn.multiply"(%arg0, %arg1) : (tensor<1x1x4x128xbf16, #full>, tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    %rot_sin = "ttnn.multiply"(%rotated, %arg2) : (tensor<1x1x4x128xbf16, #full>, tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    %result = "ttnn.add"(%x_cos, %rot_sin) : (tensor<1x1x4x128xbf16, #full>, tensor<1x1x4x128xbf16, #full>) -> tensor<1x1x4x128xbf16, #full>
    return %result : tensor<1x1x4x128xbf16, #full>
  }
}
