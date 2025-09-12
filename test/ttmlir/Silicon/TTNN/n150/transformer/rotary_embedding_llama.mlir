// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#system_memory = #ttnn.buffer_type<system_memory>

#dram_interleaved_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#height_sharded_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <8x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

#height_sharded_mem_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0, 0), (7, 0)>]>, <32x32>, <row_major>, <physical>>>

module {
  func.func @prefill(%input: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %cos: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %sin: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %trans_mat: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #dram_interleaved_encoding> {
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = false }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #dram_interleaved_encoding>
    return %0 : tensor<1x1x32x32xbf16, #dram_interleaved_encoding>
  }

  func.func @decode(%input: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %cos: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %sin: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %trans_mat: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding> {
    %0 = "ttnn.to_memory_config"(%input) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
    %1 = "ttnn.to_memory_config"(%cos) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
    %2 = "ttnn.to_memory_config"(%sin) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
    %3 = "ttnn.to_memory_config"(%trans_mat) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
    %4 = "ttnn.rotary_embedding_llama"(%0, %1, %2, %3) <{ is_decode_mode = true }> : (tensor<1x1x32x32xbf16, #height_sharded_encoding>, tensor<1x1x32x32xbf16, #height_sharded_encoding>, tensor<1x1x32x32xbf16, #height_sharded_encoding>, tensor<1x1x32x32xbf16, #height_sharded_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
    return %4 : tensor<1x1x32x32xbf16, #height_sharded_encoding>
  }
}
