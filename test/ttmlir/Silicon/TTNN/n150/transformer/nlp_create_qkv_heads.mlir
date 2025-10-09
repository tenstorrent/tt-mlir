// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#dram = #ttnn.buffer_type<dram>
#ttnn_layout0 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x144x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 32 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1536 + d1 * 64 + d2, d3), <1x1>, memref<48x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x48x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 64 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x384x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 128 + d2, d3), <1x1>, memref<128x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout9 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout10 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout11 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout12 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 64 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout13 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x96x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout14 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x64x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout15 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 32 + d2, d3), <1x1>, memref<24x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout17 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout18 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x72x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout19 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6144 + d1 * 512 + d2, d3), <1x1>, memref<192x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout20 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @nlp_create_qkv_heads_single_basic_transposed(%arg0: tensor<1x1x32x4608xbf16, #ttnn_layout0>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x64x32xbf16, #ttnn_layout2>, tensor<1x24x32x64xbf16, #ttnn_layout1>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 24 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x4608xbf16, #ttnn_layout0>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x64x32xbf16, #ttnn_layout2>, tensor<1x24x32x64xbf16, #ttnn_layout1>)
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x4608xbf16, #ttnn_layout0>) -> ()
    return %query, %key, %value : tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x64x32xbf16, #ttnn_layout2>, tensor<1x24x32x64xbf16, #ttnn_layout1>
  }

  func.func @nlp_create_qkv_heads_single_basic_not_transposed(%arg0: tensor<1x1x32x4608xbf16, #ttnn_layout0>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x32x64xbf16, #ttnn_layout1>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 24 : ui32, transpose_k_heads = false}> : (tensor<1x1x32x4608xbf16, #ttnn_layout0>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x32x64xbf16, #ttnn_layout1>)
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x4608xbf16, #ttnn_layout0>) -> ()
    return %query, %key, %value : tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x24x32x64xbf16, #ttnn_layout1>
  }

  func.func @nlp_create_qkv_heads_single_8_heads(%arg0: tensor<1x1x32x1536xbf16, #ttnn_layout3>) -> (tensor<1x8x32x64xbf16, #ttnn_layout4>, tensor<1x8x64x32xbf16, #ttnn_layout5>, tensor<1x8x32x64xbf16, #ttnn_layout4>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 8 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>) -> (tensor<1x8x32x64xbf16, #ttnn_layout4>, tensor<1x8x64x32xbf16, #ttnn_layout5>, tensor<1x8x32x64xbf16, #ttnn_layout4>)
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>) -> ()
    return %query, %key, %value : tensor<1x8x32x64xbf16, #ttnn_layout4>, tensor<1x8x64x32xbf16, #ttnn_layout5>, tensor<1x8x32x64xbf16, #ttnn_layout4>
  }

  func.func @nlp_create_qkv_heads_single_head_128(%arg0: tensor<1x1x32x12288xbf16, #ttnn_layout6>) -> (tensor<1x32x32x128xbf16, #ttnn_layout7>, tensor<1x32x128x32xbf16, #ttnn_layout8>, tensor<1x32x32x128xbf16, #ttnn_layout7>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 32 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x12288xbf16, #ttnn_layout6>) -> (tensor<1x32x32x128xbf16, #ttnn_layout7>, tensor<1x32x128x32xbf16, #ttnn_layout8>, tensor<1x32x32x128xbf16, #ttnn_layout7>)
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x12288xbf16, #ttnn_layout6>) -> ()
    return %query, %key, %value : tensor<1x32x32x128xbf16, #ttnn_layout7>, tensor<1x32x128x32xbf16, #ttnn_layout8>, tensor<1x32x32x128xbf16, #ttnn_layout7>
  }

  func.func @nlp_create_qkv_heads_dual_basic_gqa(%arg0: tensor<1x1x32x1536xbf16, #ttnn_layout3>, %arg1: tensor<1x1x32x1024xbf16, #ttnn_layout9>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x8x64x32xbf16, #ttnn_layout5>, tensor<1x8x32x64xbf16, #ttnn_layout4>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{num_kv_heads = 8 : ui32, num_q_heads = 24 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>, tensor<1x1x32x1024xbf16, #ttnn_layout9>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x8x64x32xbf16, #ttnn_layout5>, tensor<1x8x32x64xbf16, #ttnn_layout4>)
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x1x32x1024xbf16, #ttnn_layout9>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>) -> ()
    return %query, %key, %value : tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x8x64x32xbf16, #ttnn_layout5>, tensor<1x8x32x64xbf16, #ttnn_layout4>
  }

  func.func @nlp_create_qkv_heads_dual_not_transposed(%arg0: tensor<1x1x32x1536xbf16, #ttnn_layout3>, %arg1: tensor<1x1x32x1024xbf16, #ttnn_layout9>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x8x32x64xbf16, #ttnn_layout4>, tensor<1x8x32x64xbf16, #ttnn_layout4>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{num_kv_heads = 8 : ui32, num_q_heads = 24 : ui32, transpose_k_heads = false}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>, tensor<1x1x32x1024xbf16, #ttnn_layout9>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x8x32x64xbf16, #ttnn_layout4>, tensor<1x8x32x64xbf16, #ttnn_layout4>)
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x1x32x1024xbf16, #ttnn_layout9>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>) -> ()
    return %query, %key, %value : tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x8x32x64xbf16, #ttnn_layout4>, tensor<1x8x32x64xbf16, #ttnn_layout4>
  }

  func.func @nlp_create_qkv_heads_dual_mqa(%arg0: tensor<1x1x32x1536xbf16, #ttnn_layout3>, %arg1: tensor<1x1x32x128xbf16, #ttnn_layout10>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x1x64x32xbf16, #ttnn_layout12>, tensor<1x1x32x64xbf16, #ttnn_layout11>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{num_kv_heads = 1 : ui32, num_q_heads = 24 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>, tensor<1x1x32x128xbf16, #ttnn_layout10>) -> (tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x1x64x32xbf16, #ttnn_layout12>, tensor<1x1x32x64xbf16, #ttnn_layout11>)
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x1x32x128xbf16, #ttnn_layout10>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x1536xbf16, #ttnn_layout3>) -> ()
    return %query, %key, %value : tensor<1x24x32x64xbf16, #ttnn_layout1>, tensor<1x1x64x32xbf16, #ttnn_layout12>, tensor<1x1x32x64xbf16, #ttnn_layout11>
  }

  func.func @nlp_create_qkv_heads_dual_head_128(%arg0: tensor<1x1x32x3072xbf16, #ttnn_layout13>, %arg1: tensor<1x1x32x2048xbf16, #ttnn_layout14>) -> (tensor<1x24x32x128xbf16, #ttnn_layout15>, tensor<1x8x128x32xbf16, #ttnn_layout16>, tensor<1x8x32x128xbf16, #ttnn_layout17>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{num_kv_heads = 8 : ui32, num_q_heads = 24 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x3072xbf16, #ttnn_layout13>, tensor<1x1x32x2048xbf16, #ttnn_layout14>) -> (tensor<1x24x32x128xbf16, #ttnn_layout15>, tensor<1x8x128x32xbf16, #ttnn_layout16>, tensor<1x8x32x128xbf16, #ttnn_layout17>)
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x1x32x2048xbf16, #ttnn_layout14>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x3072xbf16, #ttnn_layout13>) -> ()
    return %query, %key, %value : tensor<1x24x32x128xbf16, #ttnn_layout15>, tensor<1x8x128x32xbf16, #ttnn_layout16>, tensor<1x8x32x128xbf16, #ttnn_layout17>
  }

  func.func @nlp_create_qkv_heads_llama_style(%arg0: tensor<1x1x32x4096xbf16, #ttnn_layout6>, %arg1: tensor<1x1x32x2048xbf16, #ttnn_layout14>) -> (tensor<1x32x32x128xbf16, #ttnn_layout7>, tensor<1x8x128x32xbf16, #ttnn_layout16>, tensor<1x8x32x128xbf16, #ttnn_layout17>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0, %arg1) <{num_kv_heads = 8 : ui32, num_q_heads = 32 : ui32, transpose_k_heads = true}> : (tensor<1x1x32x4096xbf16, #ttnn_layout6>, tensor<1x1x32x2048xbf16, #ttnn_layout14>) -> (tensor<1x32x32x128xbf16, #ttnn_layout7>, tensor<1x8x128x32xbf16, #ttnn_layout16>, tensor<1x8x32x128xbf16, #ttnn_layout17>)
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<1x1x32x2048xbf16, #ttnn_layout14>) -> ()
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x32x4096xbf16, #ttnn_layout6>) -> ()
    return %query, %key, %value : tensor<1x32x32x128xbf16, #ttnn_layout7>, tensor<1x8x128x32xbf16, #ttnn_layout16>, tensor<1x8x32x128xbf16, #ttnn_layout17>
  }

  func.func @nlp_create_qkv_heads_gpt_style(%arg0: tensor<1x1x512x2304xbf16, #ttnn_layout18>) -> (tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x64x512xbf16, #ttnn_layout20>, tensor<1x12x512x64xbf16, #ttnn_layout19>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 12 : ui32, transpose_k_heads = true}> : (tensor<1x1x512x2304xbf16, #ttnn_layout18>) -> (tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x64x512xbf16, #ttnn_layout20>, tensor<1x12x512x64xbf16, #ttnn_layout19>)
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x512x2304xbf16, #ttnn_layout18>) -> ()
    return %query, %key, %value : tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x64x512xbf16, #ttnn_layout20>, tensor<1x12x512x64xbf16, #ttnn_layout19>
  }

  func.func @nlp_create_qkv_heads_bert_style(%arg0: tensor<1x1x512x2304xbf16, #ttnn_layout18>) -> (tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x512x64xbf16, #ttnn_layout19>) {
    %query, %key, %value = "ttnn.nlp_create_qkv_heads"(%arg0) <{num_q_heads = 12 : ui32, transpose_k_heads = false}> : (tensor<1x1x512x2304xbf16, #ttnn_layout18>) -> (tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x512x64xbf16, #ttnn_layout19>)
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x512x2304xbf16, #ttnn_layout18>) -> ()
    return %query, %key, %value : tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x512x64xbf16, #ttnn_layout19>, tensor<1x12x512x64xbf16, #ttnn_layout19>
  }
}
