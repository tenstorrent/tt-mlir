// RUN: ttmlir-opt --convert-ttnn-to-emitpy -o %t.mlir %s
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_interleaved_encoding_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#l1_height_sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

module {
  func.func @forward(%input: tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>) {
    %0, %1, %2 = "ttnn.nlp_create_qkv_heads_decode"(%input) <{ num_heads = 32 : ui32, num_kv_heads = 32 : ui32, overlap_qk_coregrid = true }> : (tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>)
    return %0, %1, %2 : tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>
  }
}
