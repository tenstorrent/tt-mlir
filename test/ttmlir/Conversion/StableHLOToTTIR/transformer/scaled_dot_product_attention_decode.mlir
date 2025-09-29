
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @sdpa attributes {} {
  func.func public @sdpa_with_attn_mask(%query: tensor<1x1x12x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>, %cur_pos: tensor<1xi32>, %attn_mask: tensor<1x1x12x32xbf16>) -> tensor<1x1x12x64xbf16> {
    // CHECK: "ttir.scaled_dot_product_attention_decode"
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention_decode(%query, %key, %value, %cur_pos, %attn_mask) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "True", has_attention_sink = "False", is_causal = "False"}} : (tensor<1x1x12x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1xi32>, tensor<1x1x12x32xbf16>) -> tensor<1x1x12x64xbf16>
    return %0 : tensor<1x1x12x64xbf16>
  }

  func.func public @sdpa_causal(%query: tensor<1x1x12x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>, %cur_pos: tensor<1xi32>) -> tensor<1x1x12x64xbf16> {
    // CHECK: "ttir.scaled_dot_product_attention_decode"
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention_decode(%query, %key, %value, %cur_pos) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "False", has_attention_sink = "False", is_causal = "True"}} : (tensor<1x1x12x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1xi32>) -> tensor<1x1x12x64xbf16>
    return %0 : tensor<1x1x12x64xbf16>
  }
}
