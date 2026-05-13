
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @sdpa attributes {} {
  func.func public @sdpa_with_attn_mask(%query: tensor<1x12x32x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x12x32x64xbf16> {
    // CHECK: "ttir.scaled_dot_product_attention"
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%query, %key, %value, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "False"}} : (tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x12x32x64xbf16>
    return %0 : tensor<1x12x32x64xbf16>
  }

  func.func public @sdpa_causal(%query: tensor<1x12x32x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>) -> tensor<1x12x32x64xbf16> {
    // CHECK-LABEL: @sdpa_causal
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-DAG: (%arg0, %arg1, %arg2)
    // CHECK-DAG: is_causal = true,
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "True"}} : (tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>) -> tensor<1x12x32x64xbf16>
    return %0 : tensor<1x12x32x64xbf16>
  }

  func.func public @sdpa_with_attn_mask_frontend_attr(%query: tensor<1x12x32x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>, %mask: tensor<1x1x32x32xbf16>) -> tensor<1x12x32x64xbf16> {
    // CHECK-LABEL: @sdpa_with_attn_mask_frontend_attr
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-DAG: (%arg0, %arg1, %arg2, %arg3)
    // CHECK-DAG: is_causal = false,
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1, 0>}
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%query, %key, %value, %mask) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "False", has_attention_mask = "True"}} : (tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x12x32x64xbf16>
    return %0 : tensor<1x12x32x64xbf16>
  }

  func.func public @sdpa_with_attn_sink(%query: tensor<1x12x32x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>, %attention_sink: tensor<12x32xbf16>) -> tensor<1x12x32x64xbf16> {
    // CHECK-LABEL: @sdpa_with_attn_sink
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-DAG: (%arg0, %arg1, %arg2, %arg3)
    // CHECK-DAG: is_causal = true,
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0, 1>}
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%query, %key, %value, %attention_sink) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_sink = "True", is_causal = "True"}} : (tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<12x32xbf16>) -> tensor<1x12x32x64xbf16>
    return %0 : tensor<1x12x32x64xbf16>
  }

  func.func public @sdpa_with_attn_mask_and_sink(%query: tensor<1x12x32x64xbf16>, %key: tensor<1x12x32x64xbf16>, %value: tensor<1x12x32x64xbf16>, %mask: tensor<1x1x32x32xbf16>, %attention_sink: tensor<12x32xbf16>) -> tensor<1x12x32x64xbf16> {
    // CHECK-LABEL: @sdpa_with_attn_mask_and_sink
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-DAG: (%arg0, %arg1, %arg2, %arg3, %arg4)
    // CHECK-DAG: is_causal = false,
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1>}
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%query, %key, %value, %mask, %attention_sink) {api_version = 0 : i32, mhlo.frontend_attributes = {has_attention_mask = "True", has_attention_sink = "True", is_causal = "False"}} : (tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x12x32x64xbf16>, tensor<1x1x32x32xbf16>, tensor<12x32xbf16>) -> tensor<1x12x32x64xbf16>
    return %0 : tensor<1x12x32x64xbf16>
  }

  func.func public @sdpa_with_sliding_window(%query: tensor<1x12x64x64xbf16>, %key: tensor<1x12x64x64xbf16>, %value: tensor<1x12x64x64xbf16>) -> tensor<1x12x64x64xbf16> {
    // CHECK-LABEL: @sdpa_with_sliding_window
    // CHECK: "ttir.scaled_dot_product_attention"
    // CHECK-DAG: (%arg0, %arg1, %arg2)
    // CHECK-DAG: is_causal = true,
    // CHECK-DAG: sliding_window_size = 16 : ui32
    // CHECK-DAG: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>
    %0 = stablehlo.custom_call @tt.scaled_dot_product_attention(%query, %key, %value) {api_version = 0 : i32, mhlo.frontend_attributes = {is_causal = "True", sliding_window_size = "16"}} : (tensor<1x12x64x64xbf16>, tensor<1x12x64x64xbf16>, tensor<1x12x64x64xbf16>) -> tensor<1x12x64x64xbf16>
    return %0 : tensor<1x12x64x64xbf16>
  }
}
