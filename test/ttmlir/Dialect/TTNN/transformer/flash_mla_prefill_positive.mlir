// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// Positive tests that lower ttir.flash_mla_prefill through the TTNN backend
// pipeline and verify the resulting ttnn.flash_mla_prefill op carries the
// expected attributes and operand-segment layout.

module {
  // Causal, MLA-from-latent (no value, no mask).
  func.func @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_no_value
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 0>
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Causal, with explicit value tensor.
  func.func @flash_mla_prefill_causal_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_with_value
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0>
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal with attention mask.
  func.func @flash_mla_prefill_with_mask(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_with_mask
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 1>
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal, value + mask + explicit scale.
  func.func @flash_mla_prefill_value_mask_scale(%query: tensor<2x8x64x128xbf16>, %key: tensor<2x1x64x128xbf16>, %value: tensor<2x1x64x96xbf16>, %mask: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_value_mask_scale
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: head_dim_v = 96 : ui32
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: scale = 1.250000e-01
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1>
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value, %mask) <{operandSegmentSizes = array<i32: 1, 1, 1, 1>, head_dim_v = 96 : ui32, is_causal = false, scale = 0.125 : f32}> : (tensor<2x8x64x128xbf16>, tensor<2x1x64x128xbf16>, tensor<2x1x64x96xbf16>, tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16>
    return %0 : tensor<2x8x64x96xbf16>
  }

  // Hq == Hkv (vanilla MHA layout).
  func.func @flash_mla_prefill_mha(%query: tensor<1x8x32x128xbf16>, %key: tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_mha
    // CHECK: "ttnn.flash_mla_prefill"
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x8x32x128xbf16>, tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }
}
