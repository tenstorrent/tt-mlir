// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for rotary_embedding operation.

// Verify that the parsing fails if cos and sin have different data types.
#dram = #ttnn.buffer_type<dram>
#bf16_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#f32_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @dtype_mismatch(%input: tensor<1x1x32x32xbf16, #bf16_encoding>, %cos: tensor<1x1x32x32xbf16, #bf16_encoding>, %sin: tensor<1x1x32x32xf32, #f32_encoding>) -> tensor<1x1x32x32xbf16, #bf16_encoding> {
    // CHECK: error: 'ttnn.rotary_embedding' op cos and sin tensor dtypes must match.
    %0 = "ttnn.rotary_embedding"(%input, %cos, %sin) : (tensor<1x1x32x32xbf16, #bf16_encoding>, tensor<1x1x32x32xbf16, #bf16_encoding>, tensor<1x1x32x32xf32, #f32_encoding>) -> tensor<1x1x32x32xbf16, #bf16_encoding>
    return %0 : tensor<1x1x32x32xbf16, #bf16_encoding>
  }
}

// -----

// Verify that the parsing fails if cos and sin have different shapes.
#dram = #ttnn.buffer_type<dram>
#encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#cos_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @shape_mismatch(%input: tensor<1x1x32x32xbf16, #encoding>, %cos: tensor<1x1x32x64xbf16, #cos_encoding>, %sin: tensor<1x1x32x32xbf16, #encoding>) -> tensor<1x1x32x32xbf16, #encoding> {
    // CHECK: error: 'ttnn.rotary_embedding' op cos and sin tensor shapes must match.
    %0 = "ttnn.rotary_embedding"(%input, %cos, %sin) : (tensor<1x1x32x32xbf16, #encoding>, tensor<1x1x32x64xbf16, #cos_encoding>, tensor<1x1x32x32xbf16, #encoding>) -> tensor<1x1x32x32xbf16, #encoding>
    return %0 : tensor<1x1x32x32xbf16, #encoding>
  }
}
