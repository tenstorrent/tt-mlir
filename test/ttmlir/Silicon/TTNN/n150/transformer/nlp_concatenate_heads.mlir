// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#dram = #ttnn.buffer_type<dram>
#input_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 4 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#output_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4 + d1 * 4 + d2,d3), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @nlp_concatenate_heads(%input: tensor<2x4x4x32xf32, #input_encoding>) -> tensor<2x1x4x128xf32, #output_encoding> {
    %0 = "ttnn.nlp_concatenate_heads"(%input) : (tensor<2x4x4x32xf32, #input_encoding>) -> tensor<2x1x4x128xf32, #output_encoding>
    return %0 : tensor<2x1x4x128xf32, #output_encoding>
  }
}
