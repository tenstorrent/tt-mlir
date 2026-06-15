// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @flash_mla_prefill attributes {} {
  // Causal, no value (MLA-from-latent), no mask.
  func.func public @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_no_value
    // CHECK: "ttcore.composite"(%arg0, %arg1)
    // CHECK-SAME: has_attention_mask = false
    // CHECK-SAME: has_value = false
    // CHECK-SAME: head_dim_v = 64 : ui32
    // CHECK-SAME: is_causal = true
    // CHECK-SAME: composite_name = "flash_mla_prefill"
    // CHECK-SAME: decomposition = @flash_mla_prefill_decomp
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", is_causal = "True", has_value = "False", has_attention_mask = "False"}} : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Causal, with explicit value tensor.
  func.func public @flash_mla_prefill_causal_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_with_value
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: has_attention_mask = false
    // CHECK-SAME: has_value = true
    // CHECK-SAME: head_dim_v = 64 : ui32
    // CHECK-SAME: is_causal = true
    // CHECK-SAME: composite_name = "flash_mla_prefill"
    // CHECK-SAME: decomposition = @flash_mla_prefill_decomp_0
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", is_causal = "True", has_value = "True", has_attention_mask = "False"}} : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal with attention mask.
  func.func public @flash_mla_prefill_with_mask(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_with_mask
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2)
    // CHECK-SAME: has_attention_mask = true
    // CHECK-SAME: has_value = false
    // CHECK-SAME: head_dim_v = 64 : ui32
    // CHECK-SAME: is_causal = false
    // CHECK-SAME: composite_name = "flash_mla_prefill"
    // CHECK-SAME: decomposition = @flash_mla_prefill_decomp_1
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "64", is_causal = "False", has_value = "False", has_attention_mask = "True"}} : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // With value, mask, and explicit scale.
  func.func public @flash_mla_prefill_value_mask_scale(%query: tensor<2x8x64x128xbf16>, %key: tensor<2x1x64x128xbf16>, %value: tensor<2x1x64x96xbf16>, %mask: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_value_mask_scale
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2, %arg3)
    // CHECK-SAME: has_attention_mask = true
    // CHECK-SAME: has_value = true
    // CHECK-SAME: head_dim_v = 96 : ui32
    // CHECK-SAME: is_causal = false
    // CHECK-SAME: scale = 1.250000e-01 : f32
    // CHECK-SAME: composite_name = "flash_mla_prefill"
    // CHECK-SAME: decomposition = @flash_mla_prefill_decomp_2
    %0 = stablehlo.custom_call @tt.flash_mla_prefill(%query, %key, %value, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {head_dim_v = "96", is_causal = "False", has_value = "True", has_attention_mask = "True", scale = "0.125"}} : (tensor<2x8x64x128xbf16>, tensor<2x1x64x128xbf16>, tensor<2x1x64x96xbf16>, tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16>
    return %0 : tensor<2x8x64x96xbf16>
  }

  // The synthesized decomposition holds the full primitive attention lowering:
  // QK^T (with GQA head-folding reshape), scale, causal/additive mask, softmax,
  // and the probs @ V matmul. The latent (no-value) form derives V from the
  // first head_dim_v features of K via slice_static.
  // CHECK: func.func private @flash_mla_prefill_decomp
  // CHECK: "ttir.slice_static"
  // CHECK: "ttir.matmul"
  // CHECK: "ttir.softmax"
  // CHECK: "ttir.matmul"
}
