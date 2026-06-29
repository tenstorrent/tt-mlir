// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="composite-resolution=force-promote" %s | FileCheck %s

// Positive tests that resolve a ttcore.composite "flash_mla_prefill" through the
// TTNN backend pipeline (TTNNResolveComposites) and verify it is promoted to the
// typed ttnn.flash_mla_prefill op carrying the expected attributes and
// operand-segment layout. The synthesized decomposition functions are the
// fallback bodies and are deleted once the typed promotion succeeds.

module {
  // Causal, MLA-from-latent (no value, no mask).
  func.func @flash_mla_prefill_causal_no_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_no_value
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: head_dim_v = 64 : ui32
    // CHECK-DAG: is_causal = true
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 0>
    // CHECK-NOT: "ttcore.composite"
    %0 = "ttcore.composite"(%query, %key) <{composite_name = "flash_mla_prefill", decomposition = @decomp_no_value, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = false, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
  func.func private @decomp_no_value(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Causal, with explicit value tensor.
  func.func @flash_mla_prefill_causal_with_value(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_causal_with_value
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0>
    %0 = "ttcore.composite"(%query, %key, %value) <{composite_name = "flash_mla_prefill", decomposition = @decomp_with_value, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = true, has_attention_mask = false}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
  func.func private @decomp_with_value(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %v: tensor<1x1x32x64xbf16>) -> tensor<1x16x32x64xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }

  // Non-causal with attention mask.
  func.func @flash_mla_prefill_with_mask(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_with_mask
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-DAG: is_causal = false
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 0, 1>
    %0 = "ttcore.composite"(%query, %key, %mask) <{composite_name = "flash_mla_prefill", decomposition = @decomp_with_mask, composite_attributes = {head_dim_v = 64 : ui32, is_causal = false, has_value = false, has_attention_mask = true}}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
  func.func private @decomp_with_mask(%q: tensor<1x16x32x128xbf16>, %k: tensor<1x1x32x128xbf16>, %m: tensor<1x1x32x32xbf16>) -> tensor<1x16x32x64xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xbf16>) -> tensor<1x16x32x64xbf16>
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
    %0 = "ttcore.composite"(%query, %key, %value, %mask) <{composite_name = "flash_mla_prefill", decomposition = @decomp_value_mask_scale, composite_attributes = {head_dim_v = 96 : ui32, is_causal = false, scale = 1.250000e-01 : f32, has_value = true, has_attention_mask = true}}> : (tensor<2x8x64x128xbf16>, tensor<2x1x64x128xbf16>, tensor<2x1x64x96xbf16>, tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16>
    return %0 : tensor<2x8x64x96xbf16>
  }
  func.func private @decomp_value_mask_scale(%q: tensor<2x8x64x128xbf16>, %k: tensor<2x1x64x128xbf16>, %v: tensor<2x1x64x96xbf16>, %m: tensor<2x1x64x64xbf16>) -> tensor<2x8x64x96xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 8 : i32, 64 : i32, 96 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<2x8x64x128xbf16>) -> tensor<2x8x64x96xbf16>
    return %0 : tensor<2x8x64x96xbf16>
  }

  // Hq == Hkv (vanilla MHA layout).
  func.func @flash_mla_prefill_mha(%query: tensor<1x8x32x128xbf16>, %key: tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16> {
    // CHECK-LABEL: @flash_mla_prefill_mha
    // CHECK: "ttnn.flash_mla_prefill"
    %0 = "ttcore.composite"(%query, %key) <{composite_name = "flash_mla_prefill", decomposition = @decomp_mha, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = false, has_attention_mask = false}}> : (tensor<1x8x32x128xbf16>, tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }
  func.func private @decomp_mha(%q: tensor<1x8x32x128xbf16>, %k: tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 8 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8x32x128xbf16>) -> tensor<1x8x32x64xbf16>
    return %0 : tensor<1x8x32x64xbf16>
  }

  // f32 query + key + value.
  func.func @flash_mla_prefill_f32_with_value(%query: tensor<1x16x32x128xf32>, %key: tensor<1x1x32x128xf32>, %value: tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xf32> {
    // CHECK-LABEL: @flash_mla_prefill_f32_with_value
    // CHECK: "ttnn.flash_mla_prefill"
    // CHECK-SAME: bf16
    // CHECK-NOT: "ttcore.composite"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttcore.composite"(%query, %key, %value) <{composite_name = "flash_mla_prefill", decomposition = @decomp_f32, composite_attributes = {head_dim_v = 64 : ui32, is_causal = true, has_value = true, has_attention_mask = false}}> : (tensor<1x16x32x128xf32>, tensor<1x1x32x128xf32>, tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }
  func.func private @decomp_f32(%q: tensor<1x16x32x128xf32>, %k: tensor<1x1x32x128xf32>, %v: tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xf32> {
    %0 = "ttir.slice_static"(%q) <{begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32], ends = [1 : i32, 16 : i32, 32 : i32, 64 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x16x32x128xf32>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }
}
