// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// When the typed ttnn.chunked_scaled_dot_product_attention op cannot be promoted
// (here: TTNNResolveComposites runs in inline mode), the composite falls back to
// inlining its decomposition function.
//
// Unlike flash_mla_prefill, the chunked-prefill decomposition is intentionally
// *lean* (verify-only): chunked SDPA attends over a paged K/V cache addressed at
// runtime by page_table and offsets the causal mask by the runtime device tensor
// chunk_start_idx, neither of which lowers cleanly to static TTIR primitives.
// Real numeric correctness comes from promotion to the typed ttnn op; this
// fallback only needs to be structurally valid IR that type-checks, inlines, and
// verifies. It is modelled as an identity over the query.

// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=inline" %s | FileCheck %s

// CHECK-LABEL: func.func @chunked_sdpa_decompose
// The composite is replaced by its (lean identity) decomposition body, and
// neither the composite nor the typed op survive.
// CHECK-NOT: "ttcore.composite"
// CHECK-NOT: "ttnn.chunked_scaled_dot_product_attention"
// The result of the inlined identity body is the query argument itself.
// CHECK: return %arg0
module {
  func.func @chunked_sdpa_decompose(%query: tensor<1x12x64x64xf32>, %key: tensor<128x12x32x64xf32>, %value: tensor<128x12x32x64xf32>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xf32> {
    %0 = "ttcore.composite"(%query, %key, %value, %page_table, %chunk_start_idx) <{composite_name = "chunked_scaled_dot_product_attention", decomposition = @chunked_scaled_dot_product_attention_decomp, composite_attributes = {scale = 1.250000e-01 : f32}}> : (tensor<1x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x12x64x64xf32>
    return %0 : tensor<1x12x64x64xf32>
  }

  // The lean fallback decomposition synthesized by the StableHLO conversion: an
  // identity over the query (the output mirrors the query shape).
  func.func private @chunked_scaled_dot_product_attention_decomp(%query: tensor<1x12x64x64xf32>, %key: tensor<128x12x32x64xf32>, %value: tensor<128x12x32x64xf32>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xf32> {
    return %query : tensor<1x12x64x64xf32>
  }
}
