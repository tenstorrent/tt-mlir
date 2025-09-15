// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#input_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 4 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#output_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @nlp_concatenate_heads(%input: tensor<2x4x4x32xf32, #input_encoding>) -> tensor<1x1x32x32xf32, #output_encoding> {
    // CHECK: error: 'ttnn.nlp_concatenate_heads' op output tensor dim 3 must be
    %0 = "ttnn.nlp_concatenate_heads"(%input) : (tensor<2x4x4x32xf32, #input_encoding>) -> tensor<1x1x32x32xf32, #output_encoding>
    return %0 : tensor<1x1x32x32xf32, #output_encoding>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#encoding = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @nlp_concatenate_heads(%input: tensor<8xf32, #encoding>) -> tensor<8xf32, #encoding> {
    // CHECK: error: 'ttnn.nlp_concatenate_heads' op output tensor must be a 4D tensor
    %0 = "ttnn.nlp_concatenate_heads"(%input) : (tensor<8xf32, #encoding>) -> tensor<8xf32, #encoding>
    return %0 : tensor<8xf32, #encoding>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#input_encoding = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#output_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @nlp_concatenate_heads(%input: tensor<8xf32, #input_encoding>) -> tensor<1x1x1x1xf32, #output_encoding> {
    // CHECK: error: 'ttnn.nlp_concatenate_heads' op input tensor must be a 4D tensor
    %0 = "ttnn.nlp_concatenate_heads"(%input) : (tensor<8xf32, #input_encoding>) -> tensor<1x1x1x1xf32, #output_encoding>
    return %0 : tensor<1x1x1x1xf32, #output_encoding>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#input_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 4 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#output_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 4 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @nlp_concatenate_heads(%input: tensor<2x4x4x32xf32, #input_encoding>) -> tensor<2x2x4x128xf32, #output_encoding> {
    // CHECK: error: 'ttnn.nlp_concatenate_heads' op output tensor dim 1 must be 1
    %0 = "ttnn.nlp_concatenate_heads"(%input) : (tensor<2x4x4x32xf32, #input_encoding>) -> tensor<2x2x4x128xf32, #output_encoding>
    return %0 : tensor<2x2x4x128xf32, #output_encoding>
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#input_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 4 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#output_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8 + d1 * 8 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @nlp_concatenate_heads(%input: tensor<2x4x4x32xf32, #input_encoding>) -> tensor<2x1x8x128xf32, #output_encoding> {
    // CHECK: error: 'ttnn.nlp_concatenate_heads' op output tensor dim 2 must be the same as input tensor dim 1
    %0 = "ttnn.nlp_concatenate_heads"(%input) : (tensor<2x4x4x32xf32, #input_encoding>) -> tensor<2x1x8x128xf32, #output_encoding>
    return %0 : tensor<2x1x8x128xf32, #output_encoding>
  }
}
