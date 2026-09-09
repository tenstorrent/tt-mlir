// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @chunked_sdpa attributes {} {
  // query: [num_users, num_heads, chunk_len, head_size]; key/value: paged cache
  // [num_blocks, num_kv_heads, block_size, head_size].
  func.func public @chunked_sdpa(%query: tensor<1x12x64x64xbf16>, %key: tensor<128x12x32x64xbf16>, %value: tensor<128x12x32x64xbf16>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
    // CHECK-LABEL: @chunked_sdpa
    // CHECK: "ttir.chunked_scaled_dot_product_attention"
    // CHECK-SAME: <{scale = 1.250000e-01 : f32}>
    // CHECK-SAME: (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x4xi32>, tensor<1xi32>, tensor<1x12x64x64xbf16>) -> tensor<1x12x64x64xbf16>
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%query, %key, %value, %page_table, %chunk_start_idx) {api_version = 0 : i32, mhlo.frontend_attributes = {scale = "0.125"}} : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x12x64x64xbf16>
    return %0 : tensor<1x12x64x64xbf16>
  }

  // No scale frontend attribute.
  func.func public @chunked_sdpa_no_scale(%query: tensor<1x12x64x64xbf16>, %key: tensor<128x12x32x64xbf16>, %value: tensor<128x12x32x64xbf16>, %page_table: tensor<1x4xi32>, %chunk_start_idx: tensor<1xi32>) -> tensor<1x12x64x64xbf16> {
    // CHECK-LABEL: @chunked_sdpa_no_scale
    // CHECK: "ttir.chunked_scaled_dot_product_attention"
    // CHECK-NOT: scale
    %0 = stablehlo.custom_call @tt.chunked_scaled_dot_product_attention(%query, %key, %value, %page_table, %chunk_start_idx) {api_version = 0 : i32} : (tensor<1x12x64x64xbf16>, tensor<128x12x32x64xbf16>, tensor<128x12x32x64xbf16>, tensor<1x4xi32>, tensor<1xi32>) -> tensor<1x12x64x64xbf16>
    return %0 : tensor<1x12x64x64xbf16>
  }
}
