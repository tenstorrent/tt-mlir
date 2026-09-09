// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for paged flash multi-latent attention decode operation.

// Verify that the op fails if the key cache has more than one KV head (nkv > 1).
module {
  func.func @mla_decode_key_nkv_too_large(%query: tensor<1x8x16x64xbf16>, %key: tensor<128x2x32x64xbf16>, %page_table: tensor<8x4xui32>, %cur_pos: tensor<8xsi32>) -> tensor<1x8x16x64xbf16> {
    // CHECK: error: 'ttir.paged_flash_multi_latent_attention_decode' op Key num KV heads (nkv) must be 1, got 2.
    %0 = "ttir.paged_flash_multi_latent_attention_decode"(%query, %key, %page_table, %cur_pos) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 1, 0>}> : (tensor<1x8x16x64xbf16>, tensor<128x2x32x64xbf16>, tensor<8x4xui32>, tensor<8xsi32>) -> tensor<1x8x16x64xbf16>
    return %0 : tensor<1x8x16x64xbf16>
  }
}

// -----

// Verify that the op fails if a provided value cache has more than one KV head
// (nkv > 1), even when the key cache is valid (nkv == 1).
module {
  func.func @mla_decode_value_nkv_too_large(%query: tensor<1x8x16x64xbf16>, %key: tensor<128x1x32x64xbf16>, %value: tensor<128x2x32x64xbf16>, %page_table: tensor<8x4xui32>) -> tensor<1x8x16x64xbf16> {
    // CHECK: error: 'ttir.paged_flash_multi_latent_attention_decode' op Value num KV heads (nkv) must be 1, got 2.
    %0 = "ttir.paged_flash_multi_latent_attention_decode"(%query, %key, %value, %page_table) <{head_dim_v = 64 : ui32, is_causal = true, operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>}> : (tensor<1x8x16x64xbf16>, tensor<128x1x32x64xbf16>, tensor<128x2x32x64xbf16>, tensor<8x4xui32>) -> tensor<1x8x16x64xbf16>
    return %0 : tensor<1x8x16x64xbf16>
  }
}
