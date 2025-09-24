// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<24x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#ttnn_layout_8heads_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_8heads_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x8>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#ttnn_layout_16heads_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <16x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_16heads_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x16>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#ttnn_layout_32heads_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_32heads_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 4096 + d2, d3), <1x32>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#ttnn_layout_12heads_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <12x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_12heads_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 288 + d1 * 288 + d2, d3), <1x12>, memref<1x2x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#ttnn_layout_4heads_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 32 + d2, d3), <4x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_4heads_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x4>, memref<1x8x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#ttnn_layout_single_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_single_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x8>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

#ttnn_layout_24heads_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 32 + d2, d3), <24x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#ttnn_layout_24heads_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2304 + d1 * 2304 + d2, d3), <1x24>, memref<1x4x!ttcore.tile<32x32, bf16>, #l1>, <width_sharded>>

module {
  func.func @nlp_concat_heads_decode_8_heads(%arg0: tensor<1x8x32x128xbf16, #ttnn_layout>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_8heads_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>]>, <32x128>, <row_major>, <physical>>>}> : (tensor<1x8x32x128xbf16, #ttnn_layout>) -> tensor<1x8x32x128xbf16, #ttnn_layout_8heads_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 8 : ui32}> : (tensor<1x8x32x128xbf16, #ttnn_layout_8heads_in>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_8heads_out>
    return %0 : tensor<1x1x32x1024xbf16, #ttnn_layout_8heads_out>
  }

  func.func @nlp_concat_heads_decode_16_heads(%arg0: tensor<1x16x32x64xbf16, #ttnn_layout1>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_16heads_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,1)>]>, <32x64>, <row_major>, <physical>>>}> : (tensor<1x16x32x64xbf16, #ttnn_layout1>) -> tensor<1x16x32x64xbf16, #ttnn_layout_16heads_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 16 : ui32}> : (tensor<1x16x32x64xbf16, #ttnn_layout_16heads_in>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_16heads_out>
    return %0 : tensor<1x1x32x1024xbf16, #ttnn_layout_16heads_out>
  }

  func.func @nlp_concat_heads_decode_32_heads(%arg0: tensor<1x32x32x128xbf16, #ttnn_layout2>) -> tensor<1x1x32x4096xbf16, #ttnn_layout_32heads_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,3)>]>, <32x128>, <row_major>, <physical>>>}> : (tensor<1x32x32x128xbf16, #ttnn_layout2>) -> tensor<1x32x32x128xbf16, #ttnn_layout_32heads_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 32 : ui32}> : (tensor<1x32x32x128xbf16, #ttnn_layout_32heads_in>) -> tensor<1x1x32x4096xbf16, #ttnn_layout_32heads_out>
    return %0 : tensor<1x1x32x4096xbf16, #ttnn_layout_32heads_out>
  }

  func.func @nlp_concat_heads_decode_12_heads(%arg0: tensor<1x12x32x64xbf16, #ttnn_layout3>) -> tensor<1x1x32x768xbf16, #ttnn_layout_12heads_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,0)>, #ttnn.core_range<(0,1), (3,1)>]>, <32x64>, <row_major>, <physical>>>}> : (tensor<1x12x32x64xbf16, #ttnn_layout3>) -> tensor<1x12x32x64xbf16, #ttnn_layout_12heads_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 12 : ui32}> : (tensor<1x12x32x64xbf16, #ttnn_layout_12heads_in>) -> tensor<1x1x32x768xbf16, #ttnn_layout_12heads_out>
    return %0 : tensor<1x1x32x768xbf16, #ttnn_layout_12heads_out>
  }

  func.func @nlp_concat_heads_decode_4_heads(%arg0: tensor<1x4x32x256xbf16, #ttnn_layout4>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_4heads_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (3,0)>]>, <32x256>, <row_major>, <physical>>>}> : (tensor<1x4x32x256xbf16, #ttnn_layout4>) -> tensor<1x4x32x256xbf16, #ttnn_layout_4heads_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 4 : ui32}> : (tensor<1x4x32x256xbf16, #ttnn_layout_4heads_in>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_4heads_out>
    return %0 : tensor<1x1x32x1024xbf16, #ttnn_layout_4heads_out>
  }

  func.func @nlp_concat_heads_decode_single_user(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout5>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_single_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,0)>]>, <32x128>, <row_major>, <physical>>>}> : (tensor<1x1x32x128xbf16, #ttnn_layout5>) -> tensor<1x1x32x128xbf16, #ttnn_layout_single_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 8 : ui32}> : (tensor<1x1x32x128xbf16, #ttnn_layout_single_in>) -> tensor<1x1x32x1024xbf16, #ttnn_layout_single_out>
    return %0 : tensor<1x1x32x1024xbf16, #ttnn_layout_single_out>
  }

  func.func @nlp_concat_heads_decode_24_heads(%arg0: tensor<1x24x32x128xbf16, #ttnn_layout6>) -> tensor<1x1x32x3072xbf16, #ttnn_layout_24heads_out> {
    %input = "ttnn.to_memory_config"(%arg0) <{memory_config = #ttnn.memory_config<#l1, <height_sharded>, #ttnn.shard_spec<#ttnn.core_range_set<[#ttnn.core_range<(0,0), (7,2)>]>, <32x128>, <row_major>, <physical>>>}> : (tensor<1x24x32x128xbf16, #ttnn_layout6>) -> tensor<1x24x32x128xbf16, #ttnn_layout_24heads_in>
    %0 = "ttnn.nlp_concat_heads_decode"(%input) <{num_heads = 24 : ui32}> : (tensor<1x24x32x128xbf16, #ttnn_layout_24heads_in>) -> tensor<1x1x32x3072xbf16, #ttnn_layout_24heads_out>
    return %0 : tensor<1x1x32x3072xbf16, #ttnn_layout_24heads_out>
  }
}
