// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#dram_interleaved_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  func.func @dtype(%input: tensor<1x1x32x32xf32, #dram_interleaved_encoding>, %cos: tensor<1x1x32x32xf32, #dram_interleaved_encoding>, %sin: tensor<1x1x32x32xf32, #dram_interleaved_encoding>, %trans_mat: tensor<1x1x32x32xf32, #dram_interleaved_encoding>) -> tensor<1x1x32x32xf32, #dram_interleaved_encoding> {
    // CHECK: error: 'ttnn.rotary_embedding_llama' op all input tensors must be bfloat16 type.
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = false }> : (tensor<1x1x32x32xf32, #dram_interleaved_encoding>, tensor<1x1x32x32xf32, #dram_interleaved_encoding>, tensor<1x1x32x32xf32, #dram_interleaved_encoding>, tensor<1x1x32x32xf32, #dram_interleaved_encoding>) -> tensor<1x1x32x32xf32, #dram_interleaved_encoding>
    return %0 : tensor<1x1x32x32xf32, #dram_interleaved_encoding>
  }
}


// -----
#system_memory = #ttnn.buffer_type<system_memory>
#system_memory_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #system_memory>>
module {
  func.func @row_major(%input: tensor<1x1x32x32xbf16, #system_memory_encoding>, %cos: tensor<1x1x32x32xbf16, #system_memory_encoding>, %sin: tensor<1x1x32x32xbf16, #system_memory_encoding>, %trans_mat: tensor<1x1x32x32xbf16, #system_memory_encoding>) -> tensor<1x1x32x32xbf16, #system_memory_encoding> {
    // CHECK: error: 'ttnn.rotary_embedding_llama' op all input tensors must be on device.
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = false }> : (tensor<1x1x32x32xbf16, #system_memory_encoding>, tensor<1x1x32x32xbf16, #system_memory_encoding>, tensor<1x1x32x32xbf16, #system_memory_encoding>, tensor<1x1x32x32xbf16, #system_memory_encoding>) -> tensor<1x1x32x32xbf16, #system_memory_encoding>
    return %0 : tensor<1x1x32x32xbf16, #system_memory_encoding>
  }
}

// -----
#dram = #ttnn.buffer_type<dram>
#row_major_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
module {
  func.func @row_major(%input: tensor<1x1x32x32xbf16, #row_major_encoding>, %cos: tensor<1x1x32x32xbf16, #row_major_encoding>, %sin: tensor<1x1x32x32xbf16, #row_major_encoding>, %trans_mat: tensor<1x1x32x32xbf16, #row_major_encoding>) -> tensor<1x1x32x32xbf16, #row_major_encoding> {
    // CHECK: error: 'ttnn.rotary_embedding_llama' op all input tensors must have tiled layout.
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = false }> : (tensor<1x1x32x32xbf16, #row_major_encoding>, tensor<1x1x32x32xbf16, #row_major_encoding>, tensor<1x1x32x32xbf16, #row_major_encoding>, tensor<1x1x32x32xbf16, #row_major_encoding>) -> tensor<1x1x32x32xbf16, #row_major_encoding>
    return %0 : tensor<1x1x32x32xbf16, #row_major_encoding>
  }
}

// -----
#dram = #ttnn.buffer_type<dram>
#dram_interleaved_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#cos_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 30 + d1 * 30 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @row_major(%input: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %cos: tensor<1x1x30x32xbf16, #cos_encoding>, %sin: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %trans_mat: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #dram_interleaved_encoding> {
    // CHECK: error: 'ttnn.rotary_embedding_llama' op cos and sin tensors must have the same shape.
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = false }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x30x32xbf16, #cos_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #dram_interleaved_encoding>
    return %0 : tensor<1x1x32x32xbf16, #dram_interleaved_encoding>
  }
}

// -----
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#dram_interleaved_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#height_sharded_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <8x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#result_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 30 + d1 * 30 + d2, d3), <8x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#height_sharded_mem_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0, 0), (7, 0)>]>, <32x32>, <row_major>>>

func.func @decode(%input: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %cos: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %sin: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %trans_mat: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x30x32xbf16, #result_encoding> {
  // CHECK: error: 'ttnn.rotary_embedding_llama' op output shape must match input shape.
  %0 = "ttnn.to_memory_config"(%input) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
  %1 = "ttnn.to_memory_config"(%cos) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
  %2 = "ttnn.to_memory_config"(%sin) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
  %3 = "ttnn.to_memory_config"(%trans_mat) <{memory_config = #height_sharded_mem_config }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>) -> tensor<1x1x32x32xbf16, #height_sharded_encoding>
  %4 = "ttnn.rotary_embedding_llama"(%0, %1, %2, %3) <{ is_decode_mode = true }> : (tensor<1x1x32x32xbf16, #height_sharded_encoding>, tensor<1x1x32x32xbf16, #height_sharded_encoding>, tensor<1x1x32x32xbf16, #height_sharded_encoding>, tensor<1x1x32x32xbf16, #height_sharded_encoding>) -> tensor<1x1x30x32xbf16, #result_encoding>
  return %4 : tensor<1x1x30x32xbf16, #result_encoding>
}

// -----
#dram = #ttnn.buffer_type<dram>
#dram_interleaved_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#trans_mat_encoding = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 30 + d1 * 30 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  func.func @dtype(%input: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %cos: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %sin: tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, %trans_mat: tensor<1x1x30x32xbf16, #trans_mat_encoding>) -> tensor<1x1x32x32xbf16, #dram_interleaved_encoding> {
    // CHECK: error: 'ttnn.rotary_embedding_llama' op transformation matrix must have shape
    %0 = "ttnn.rotary_embedding_llama"(%input, %cos, %sin, %trans_mat) <{ is_decode_mode = false }> : (tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x32x32xbf16, #dram_interleaved_encoding>, tensor<1x1x30x32xbf16, #trans_mat_encoding>) -> tensor<1x1x32x32xbf16, #dram_interleaved_encoding>
    return %0 : tensor<1x1x32x32xbf16, #dram_interleaved_encoding>
  }
}
