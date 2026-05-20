// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @flash_mla_prefill attributes {} {
  // Causal, no value (MLA-from-latent), no mask.
  func.func public @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_no_value
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: (%arg0, %arg1)
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 0>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Causal, with explicit value tensor.
  func.func public @flash_mla_prefill_causal_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_with_value
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: (%arg0, %arg1, %arg2)
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal with attention mask.
  func.func public @flash_mla_prefill_with_mask(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_with_mask
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: (%arg0, %arg1, %arg2)
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 1>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", is_causal = "False", has_value = "False", has_attention_mask = "True"}} : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // With value, mask, and explicit scale.
  func.func public @flash_mla_prefill_value_mask_scale(%query: tensor<2x8x64x128xbf16>, %key: tensor<2x1x64x128xbf16>, %value: tensor<2x1x64x96xbf16>, %mask: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_value_mask_scale
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: (%arg0, %arg1, %arg2, %arg3)
    // CHECK-DAG: head_dim_v = 96 : ui32
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: scale = 1.250000e-01
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1>
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "96", is_causal = "False", has_value = "True", has_attention_mask = "True", scale = "0.125"}} : (tensor<2x8x64x128xbf16>, tensor<2x1x64x128xbf16>, tensor<2x1x64x96xbf16>, tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16>
    return %0 : tensor<2x8x64x96xbf16>
  }
}
