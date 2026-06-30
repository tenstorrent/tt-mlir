// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @paged_flash_mla_decode attributes {} {
  // Causal MLA-from-latent decode: value is absent (V is taken from the leading
  // head_dim_v features of the latent K cache) and cur_pos drives the causal
  // mask. nkv == 1 (single compressed latent KV head).
  func.func public @mla_decode_causal_latent(%query: tensor<1x8x16x64xbf16>, %key: tensor<128x1x32x64xbf16>, %page_table: tensor<8x4xi32>, %cur_pos: tensor<8xi32>) -> tensor<1x8x16x64xbf16> {
    // CHECK-LABEL: @mla_decode_causal_latent
    // CHECK: "ttir.paged_flash_multi_latent_attention_decode"
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 1, 0, 1, 0>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %page_table, %cur_pos) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", has_attention_mask = "False", has_attention_sink = "False", has_cur_pos_tensor = "True", has_value = "False", is_causal = "True"}} : (tensor<1x8x16x64xbf16>, tensor<128x1x32x64xbf16>, tensor<8x4xi32>, tensor<8xi32>) -> tensor<1x8x16x64xbf16>
    return %0 : tensor<1x8x16x64xbf16>
  }

  // Non-causal decode with an explicit additive attention mask and a separate,
  // provided value cache whose head dim (head_dim_v = 32) differs from the
  // Q/K head dim (dh_qk = 64).
  func.func public @mla_decode_value_mask(%query: tensor<1x8x16x64xbf16>, %key: tensor<128x1x32x64xbf16>, %value: tensor<128x1x32x32xbf16>, %page_table: tensor<8x4xi32>, %attn_mask: tensor<8x16x1x128xbf16>) -> tensor<1x8x16x32xbf16> {
    // CHECK-LABEL: @mla_decode_value_mask
    // CHECK: "ttir.paged_flash_multi_latent_attention_decode"
    // CHECK-DAG: head_dim_v = 32 : ui32
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 0, 0>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %value, %page_table, %attn_mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "32", has_attention_mask = "True", has_attention_sink = "False", has_cur_pos_tensor = "False", has_value = "True", is_causal = "False"}} : (tensor<1x8x16x64xbf16>, tensor<128x1x32x64xbf16>, tensor<128x1x32x32xbf16>, tensor<8x4xi32>, tensor<8x16x1x128xbf16>) -> tensor<1x8x16x32xbf16>
    return %0 : tensor<1x8x16x32xbf16>
  }

  // Causal decode with a provided value cache, cur_pos, a per-query-head
  // attention sink, and an explicit softmax scale.
  func.func public @mla_decode_sink_scale(%query: tensor<1x8x16x64xbf16>, %key: tensor<128x1x32x64xbf16>, %value: tensor<128x1x32x64xbf16>, %page_table: tensor<8x4xi32>, %cur_pos: tensor<8xi32>, %sink: tensor<16xbf16>) -> tensor<1x8x16x64xbf16> {
    // CHECK-LABEL: @mla_decode_sink_scale
    // CHECK: "ttir.paged_flash_multi_latent_attention_decode"
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: scale = {{.+}} : f32
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 1, 1>
    %0 = stablehlo.custom_call @tt.paged_flash_mla_decode(%query, %key, %value, %page_table, %cur_pos, %sink) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", has_attention_mask = "False", has_attention_sink = "True", has_cur_pos_tensor = "True", has_value = "True", is_causal = "True", scale = "0.125"}} : (tensor<1x8x16x64xbf16>, tensor<128x1x32x64xbf16>, tensor<128x1x32x64xbf16>, tensor<8x4xi32>, tensor<8xi32>, tensor<16xbf16>) -> tensor<1x8x16x64xbf16>
    return %0 : tensor<1x8x16x64xbf16>
  }
}
