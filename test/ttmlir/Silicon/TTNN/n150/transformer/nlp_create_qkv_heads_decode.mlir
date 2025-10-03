// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#dram_interleaved_encoding_in = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#dram_interleaved_encoding_in_unary_tensor = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>

#l1_height_sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <32x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>
#l1_height_sharded_batch_8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <8x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>>

module {
  func.func @forward_no_batch_slice(%input: tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>) {
    %0, %1, %2 = "ttnn.nlp_create_qkv_heads_decode"(%input) <{ num_heads = 32 : ui32, num_kv_heads = 32 : ui32, overlap_qk_coregrid = true }> : (tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>)
    return %0, %1, %2 : tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>, tensor<1x32x32x32xbf16, #l1_height_sharded>
  }

  func.func @forward_with_batch_slice(%input: tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>) -> (tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>, tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>, tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>) {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.constant"(%0) <{dtype = #ttcore.supportedDataTypes<u32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, value = dense<[8]> : tensor<1xui32>}> : (!ttnn.device) -> tensor<1xui32, #dram_interleaved_encoding_in_unary_tensor>
    %2, %3, %4 = "ttnn.nlp_create_qkv_heads_decode"(%input, %1) <{ num_heads = 32 : ui32, num_kv_heads = 32 : ui32, overlap_qk_coregrid = true, slice_size = 8 : ui32 }> : (tensor<1x1x32x3072xbf16, #dram_interleaved_encoding_in>, tensor<1xui32, #dram_interleaved_encoding_in_unary_tensor>) -> (tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>, tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>, tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>)
    return %2, %3, %4 : tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>, tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>, tensor<1x8x32x32xbf16, #l1_height_sharded_batch_8>
  }
}
