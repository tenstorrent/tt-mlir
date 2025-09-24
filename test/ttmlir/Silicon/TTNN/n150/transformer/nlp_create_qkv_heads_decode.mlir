// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>

#dram_interleaved_encoding_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#dram_interleaved_encoding_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @forward(%input: tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>, tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>, tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>) {
    %0, %1, %2 = "ttnn.nlp_create_qkv_heads_decode"(%input) <{ num_heads = 32 : ui32, num_kv_heads = 32 : ui32, overlap_qk_coregrid = true }> : (tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>, tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>, tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>)
    return %0, %1, %2 : tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>, tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>, tensor<1x32x32x32xbf16, #dram_interleaved_encoding_out>
  }
}
