// RUN: ttmlir-opt --split-input-file %s | FileCheck %s
// Positive tests for rotary_embedding operation.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#l1_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

// Verify that cos and sin tensors with the same dtype and shape but different
// memory layouts (DRAM vs L1) verify successfully. This tests the relaxed
// RotaryEmbeddingOp verifier that compares dtype and shape separately instead
// of checking full type equality.
module {
  func.func @different_cos_sin_layouts(%input: tensor<1x1x32x32xbf16, #dram_interleaved>, %cos: tensor<1x1x32x32xbf16, #dram_interleaved>, %sin: tensor<1x1x32x32xbf16, #l1_interleaved>) -> tensor<1x1x32x32xbf16, #dram_interleaved> {
    // CHECK: ttnn.rotary_embedding
    %0 = "ttnn.rotary_embedding"(%input, %cos, %sin) : (tensor<1x1x32x32xbf16, #dram_interleaved>, tensor<1x1x32x32xbf16, #dram_interleaved>, tensor<1x1x32x32xbf16, #l1_interleaved>) -> tensor<1x1x32x32xbf16, #dram_interleaved>
    return %0 : tensor<1x1x32x32xbf16, #dram_interleaved>
  }
}
