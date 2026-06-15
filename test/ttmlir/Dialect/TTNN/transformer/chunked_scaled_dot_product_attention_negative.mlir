// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

#dram = #ttnn.buffer_type<dram>

// Query must be a 4D tensor: [num_users, num_heads, chunk_len, head_size].
#q3d_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<1536x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xi32, #dram>, <interleaved>>
#csi_layout = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1xi32, #dram>, <interleaved>>
func.func @query_not_4d(%q: tensor<12x64x64xbf16, #q3d_layout>, %k: tensor<128x12x32x64xbf16, #kv_layout>, %v: tensor<128x12x32x64xbf16, #kv_layout>, %pt: tensor<1x4xi32, #pt_layout>, %csi: tensor<1xi32, #csi_layout>) -> tensor<12x64x64xbf16, #q3d_layout> {
  // expected-error @+1 {{Query must be a 4D tensor.}}
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi) <{scale = 1.250000e-01 : f32}> : (tensor<12x64x64xbf16, #q3d_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<1x4xi32, #pt_layout>, tensor<1xi32, #csi_layout>) -> tensor<12x64x64xbf16, #q3d_layout>
  return %0 : tensor<12x64x64xbf16, #q3d_layout>
}

// -----

#dram = #ttnn.buffer_type<dram>

// chunk_start_idx must be a 1D tensor.
#q_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<1536x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xi32, #dram>, <interleaved>>
#csi2d_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xi32, #dram>, <interleaved>>
func.func @chunk_start_idx_not_1d(%q: tensor<1x12x64x64xbf16, #q_layout>, %k: tensor<128x12x32x64xbf16, #kv_layout>, %v: tensor<128x12x32x64xbf16, #kv_layout>, %pt: tensor<1x4xi32, #pt_layout>, %csi: tensor<1x1xi32, #csi2d_layout>) -> tensor<1x12x64x64xbf16, #q_layout> {
  // expected-error @+1 {{Chunk start index must be a 1D tensor.}}
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16, #q_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<1x4xi32, #pt_layout>, tensor<1x1xi32, #csi2d_layout>) -> tensor<1x12x64x64xbf16, #q_layout>
  return %0 : tensor<1x12x64x64xbf16, #q_layout>
}

// -----

#dram = #ttnn.buffer_type<dram>

// Chunk start index must have shape [1] (a single shared offset), not [N>1].
#q_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<1536x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xi32, #dram>, <interleaved>>
#csi4_layout = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<4xi32, #dram>, <interleaved>>
func.func @chunk_start_idx_not_len1(%q: tensor<1x12x64x64xbf16, #q_layout>, %k: tensor<128x12x32x64xbf16, #kv_layout>, %v: tensor<128x12x32x64xbf16, #kv_layout>, %pt: tensor<1x4xi32, #pt_layout>, %csi: tensor<4xi32, #csi4_layout>) -> tensor<1x12x64x64xbf16, #q_layout> {
  // expected-error @+1 {{Chunk start index must have shape [1]}}
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16, #q_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<1x4xi32, #pt_layout>, tensor<4xi32, #csi4_layout>) -> tensor<1x12x64x64xbf16, #q_layout>
  return %0 : tensor<1x12x64x64xbf16, #q_layout>
}

// -----

#dram = #ttnn.buffer_type<dram>

// Query head size (last dim) must match key/value head size.
#q_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv128_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<1536x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xi32, #dram>, <interleaved>>
#csi_layout = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1xi32, #dram>, <interleaved>>
func.func @head_size_mismatch(%q: tensor<1x12x64x64xbf16, #q_layout>, %k: tensor<128x12x32x128xbf16, #kv128_layout>, %v: tensor<128x12x32x128xbf16, #kv128_layout>, %pt: tensor<1x4xi32, #pt_layout>, %csi: tensor<1xi32, #csi_layout>) -> tensor<1x12x64x64xbf16, #q_layout> {
  // expected-error @+1 {{Query head size must match key/value head size.}}
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16, #q_layout>, tensor<128x12x32x128xbf16, #kv128_layout>, tensor<128x12x32x128xbf16, #kv128_layout>, tensor<1x4xi32, #pt_layout>, tensor<1xi32, #csi_layout>) -> tensor<1x12x64x64xbf16, #q_layout>
  return %0 : tensor<1x12x64x64xbf16, #q_layout>
}

// -----

#dram = #ttnn.buffer_type<dram>

// Query num heads must be divisible by key/value num heads (12 % 8 != 0).
#q_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<24x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv8_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<1024x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xi32, #dram>, <interleaved>>
#csi_layout = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1xi32, #dram>, <interleaved>>
func.func @heads_not_divisible(%q: tensor<1x12x64x64xbf16, #q_layout>, %k: tensor<128x8x32x64xbf16, #kv8_layout>, %v: tensor<128x8x32x64xbf16, #kv8_layout>, %pt: tensor<1x4xi32, #pt_layout>, %csi: tensor<1xi32, #csi_layout>) -> tensor<1x12x64x64xbf16, #q_layout> {
  // expected-error @+1 {{Query num heads must be divisible by key/value num heads.}}
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi) <{scale = 1.250000e-01 : f32}> : (tensor<1x12x64x64xbf16, #q_layout>, tensor<128x8x32x64xbf16, #kv8_layout>, tensor<128x8x32x64xbf16, #kv8_layout>, tensor<1x4xi32, #pt_layout>, tensor<1xi32, #csi_layout>) -> tensor<1x12x64x64xbf16, #q_layout>
  return %0 : tensor<1x12x64x64xbf16, #q_layout>
}

// -----

#dram = #ttnn.buffer_type<dram>

// Page table number of users (dim 0) must match query number of users.
#q2_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 64 + d2, d3), <1x1>, memref<48x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#kv_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<1536x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#pt_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4xi32, #dram>, <interleaved>>
#csi_layout = #ttnn.ttnn_layout<(d0) -> (d0), <1x1>, memref<1xi32, #dram>, <interleaved>>
func.func @page_table_users_mismatch(%q: tensor<2x12x64x64xbf16, #q2_layout>, %k: tensor<128x12x32x64xbf16, #kv_layout>, %v: tensor<128x12x32x64xbf16, #kv_layout>, %pt: tensor<1x4xi32, #pt_layout>, %csi: tensor<1xi32, #csi_layout>) -> tensor<2x12x64x64xbf16, #q2_layout> {
  // expected-error @+1 {{Page table number of users must match query number of users.}}
  %0 = "ttnn.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi) <{scale = 1.250000e-01 : f32}> : (tensor<2x12x64x64xbf16, #q2_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<128x12x32x64xbf16, #kv_layout>, tensor<1x4xi32, #pt_layout>, tensor<1xi32, #csi_layout>) -> tensor<2x12x64x64xbf16, #q2_layout>
  return %0 : tensor<2x12x64x64xbf16, #q2_layout>
}
