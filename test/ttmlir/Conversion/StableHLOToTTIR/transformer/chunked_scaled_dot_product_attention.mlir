// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// The chunked-prefill custom_call converts to a
// ttcore.composite "chunked_scaled_dot_product_attention" (promoted to the typed
// ttnn op later by TTNNResolveComposites). Inputs are query, key, value,
// page_table, chunk_start_idx; the only carried attribute is the optional scale.
module @chunked_sdpa attributes {} {
  // query: [num_users, num_heads, chunk_len, head_size]; key/value: paged cache
  // [num_blocks, num_kv_heads, block_size, head_size].
  func.func public @chunked_sdpa(%query: tensor<1x12x64x64xbf16>, %key: tensor<128x12x32x64xbf16>, %value: tensor<128x12x32x64xbf16>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
    // CHECK-LABEL: @chunked_sdpa
    // Attributes print alphabetically: composite_attributes (carrying scale),
    // then composite_name, then decomposition.
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2, %arg3, %arg4)
    // CHECK-SAME: composite_attributes = {scale = 1.250000e-01 : f32}
    // CHECK-SAME: composite_name = "chunked_scaled_dot_product_attention"
    // CHECK-SAME: decomposition = @chunked_scaled_dot_product_attention_decomp
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%query, %key, %value, %page_table, %chunk_start_idx) {api_version = 0 : i32, mhlo.frontend_attributes = {scale = "0.125"}} : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x12x64x64xbf16>
    return %0 : tensor<1x12x64x64xbf16>
  }

  // No scale frontend attribute.
  func.func public @chunked_sdpa_no_scale(%query: tensor<1x12x64x64xbf16>, %key: tensor<128x12x32x64xbf16>, %value: tensor<128x12x32x64xbf16>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
    // CHECK-LABEL: @chunked_sdpa_no_scale
    // With no scale frontend attribute the composite carries empty attributes.
    // CHECK: "ttcore.composite"(%arg0, %arg1, %arg2, %arg3, %arg4)
    // CHECK-SAME: composite_attributes = {}
    // CHECK-SAME: composite_name = "chunked_scaled_dot_product_attention"
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%query, %key, %value, %page_table, %chunk_start_idx) {api_version = 0 : i32} : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x12x64x64xbf16>
    return %0 : tensor<1x12x64x64xbf16>
  }

  // The synthesized lean fallback decomposition is an identity over the query;
  // numeric correctness comes from promotion to the typed ttnn op.
  // CHECK: func.func private @chunked_scaled_dot_product_attention_decomp
}
