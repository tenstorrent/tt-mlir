// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for paged flash multi-latent attention decode operation.

// Verify that the op fails if the key cache has more than one KV head (nkv > 1).
#dram = #ttnn.buffer_type<dram>
#query_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#key_nkv2_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <1x1>, memref<256x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#page_table_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module {
  func.func @mla_decode_key_nkv_too_large(%query: tensor<1x8x16x64xbf16, #query_layout>, %key: tensor<128x2x32x64xbf16, #key_nkv2_layout>, %page_table: tensor<8x4xui32, #page_table_layout>) -> tensor<1x8x16x64xbf16, #query_layout> {
    // CHECK: error: 'ttnn.paged_flash_multi_latent_attention_decode' op Key num KV heads (nkv) must be 1, got 2.
    %0 = "ttnn.paged_flash_multi_latent_attention_decode"(%query, %key, %page_table) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 0, 0>}> : (tensor<1x8x16x64xbf16, #query_layout>, tensor<128x2x32x64xbf16, #key_nkv2_layout>, tensor<8x4xui32, #page_table_layout>) -> tensor<1x8x16x64xbf16, #query_layout>
    return %0 : tensor<1x8x16x64xbf16, #query_layout>
  }
}

// -----

// Verify that the op fails if a provided value cache has more than one KV head
// (nkv > 1), even when the key cache is valid (nkv == 1).
#dram = #ttnn.buffer_type<dram>
#query_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#key_nkv1_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<128x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#value_nkv2_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 32 + d2, d3), <1x1>, memref<256x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#page_table_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module {
  func.func @mla_decode_value_nkv_too_large(%query: tensor<1x8x16x64xbf16, #query_layout>, %key: tensor<128x1x32x64xbf16, #key_nkv1_layout>, %value: tensor<128x2x32x64xbf16, #value_nkv2_layout>, %page_table: tensor<8x4xui32, #page_table_layout>) -> tensor<1x8x16x64xbf16, #query_layout> {
    // CHECK: error: 'ttnn.paged_flash_multi_latent_attention_decode' op Value num KV heads (nkv) must be 1, got 2.
    %0 = "ttnn.paged_flash_multi_latent_attention_decode"(%query, %key, %value, %page_table) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>}> : (tensor<1x8x16x64xbf16, #query_layout>, tensor<128x1x32x64xbf16, #key_nkv1_layout>, tensor<128x2x32x64xbf16, #value_nkv2_layout>, tensor<8x4xui32, #page_table_layout>) -> tensor<1x8x16x64xbf16, #query_layout>
    return %0 : tensor<1x8x16x64xbf16, #query_layout>
  }
}
