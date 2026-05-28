// RUN: ttmlir-opt %s | FileCheck %s

// Positive tests covering the legal operand-segment configurations of ttir.flash_mla_prefill.

module {
  // Causal, MLA-from-latent (no value, no mask).
  func.func @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_no_value
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 0>
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Causal, with explicit value tensor.
  func.func @flash_mla_prefill_causal_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_with_value
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0>
    %0 = "ttir.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Causal, no value, with explicit scale.
  func.func @flash_mla_prefill_causal_scale(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_scale
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: scale = 1.250000e-01
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true, scale = 0.125 : f32}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal, mask shape [1 x 1 x Sq x Sq] (full broadcast).
  func.func @flash_mla_prefill_mask_full_broadcast(%query: tensor<2x16x32x128xbf16>, %key: tensor<2x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<2x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_mask_full_broadcast
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 1>
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<2x16x32x128xbf16>, tensor<2x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<2x16x32x64xbf16>
    return %0 : tensor<2x16x32x64xbf16>
  }

  // Non-causal, mask shape [B x 1 x Sq x Sq] (per-batch mask).
  func.func @flash_mla_prefill_mask_batch(%query: tensor<2x16x32x128xbf16>, %key: tensor<2x1x32x128xbf16>, %mask: tensor<2x1x32x32xbf16>) -> tensor<2x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_mask_batch
    // CHECK: "ttir.flash_mla_prefill"
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<2x16x32x128xbf16>, tensor<2x1x32x128xbf16>, tensor<2x1x32x32xbf16>) -> tensor<2x16x32x64xbf16>
    return %0 : tensor<2x16x32x64xbf16>
  }

  // Non-causal, mask shape [1 x Hq x Sq x Sq] (per-head mask).
  func.func @flash_mla_prefill_mask_per_head(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x16x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_mask_per_head
    // CHECK: "ttir.flash_mla_prefill"
    %0 = "ttir.flash_mla_prefill"(%query, %key, %mask) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, head_dim_v = 64 : ui32, is_causal = false}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x16x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal, value + mask + explicit scale.
  func.func @flash_mla_prefill_value_mask_scale(%query: tensor<2x8x64x128xbf16>, %key: tensor<2x1x64x128xbf16>, %value: tensor<2x1x64x96xbf16>, %mask: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_value_mask_scale
    // CHECK: "ttir.flash_mla_prefill"
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
    // CHECK: "ttir.flash_mla_prefill"
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x8x32x128xbf16>, tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }

  // head_dim_v == qkHeadSize boundary (value absent).
  func.func @flash_mla_prefill_head_dim_equal_qk(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x128xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_head_dim_equal_qk
    // CHECK: "ttir.flash_mla_prefill"
    // CHECK-DAG: head_dim_v = 128 : ui32
    %0 = "ttir.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 128 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x128xbf16>
    return %0 : tensor<1x16x32x128xbf16>
  }
}
