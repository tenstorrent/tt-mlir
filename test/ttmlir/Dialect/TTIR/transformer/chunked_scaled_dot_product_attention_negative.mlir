// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Query must be a 4D tensor: [num_users, num_heads, chunk_len, head_size].
func.func @query_not_4d(%q: tensor<12x64x64xf32>, %k: tensor<128x12x32x64xf32>, %v: tensor<128x12x32x64xf32>, %pt: tensor<1x4xi32>, %csi: tensor<1xi32>, %o: tensor<12x64x64xf32>) -> tensor<12x64x64xf32> {
  // expected-error @+1 {{Query must be a 4D tensor.}}
  %1 = "ttir.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi, %o) : (tensor<12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>, tensor<12x64x64xf32>) -> tensor<12x64x64xf32>
  return %1 : tensor<12x64x64xf32>
}

// -----

// chunk_start_idx must be a 1D tensor.
func.func @chunk_start_idx_not_1d(%q: tensor<1x12x64x64xf32>, %k: tensor<128x12x32x64xf32>, %v: tensor<128x12x32x64xf32>, %pt: tensor<1x4xi32>, %csi: tensor<1x1xi32>, %o: tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32> {
  // expected-error @+1 {{Chunk start index must be a 1D tensor.}}
  %1 = "ttir.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi, %o) : (tensor<1x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1x1xi32>, tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32>
  return %1 : tensor<1x12x64x64xf32>
}

// -----

// Chunk start index must have shape [1] (a single shared offset), not [N>1].
func.func @chunk_start_idx_not_len1(%q: tensor<1x12x64x64xf32>, %k: tensor<128x12x32x64xf32>, %v: tensor<128x12x32x64xf32>, %pt: tensor<1x4xi32>, %csi: tensor<4xi32>, %o: tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32> {
  // expected-error @+1 {{Chunk start index must have shape [1]}}
  %1 = "ttir.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi, %o) : (tensor<1x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<4xi32>, tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32>
  return %1 : tensor<1x12x64x64xf32>
}

// -----

// Query head size (last dim) must match key/value head size.
func.func @head_size_mismatch(%q: tensor<1x12x64x64xf32>, %k: tensor<128x12x32x128xf32>, %v: tensor<128x12x32x128xf32>, %pt: tensor<1x4xi32>, %csi: tensor<1xi32>, %o: tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32> {
  // expected-error @+1 {{Query head size must match key/value head size.}}
  %1 = "ttir.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi, %o) : (tensor<1x12x64x64xf32>, tensor<128x12x32x128xf32>, tensor<128x12x32x128xf32>, tensor<1x4xi32>, tensor<1xi32>, tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32>
  return %1 : tensor<1x12x64x64xf32>
}

// -----

// Query num heads must be divisible by key/value num heads (12 % 8 != 0).
func.func @heads_not_divisible(%q: tensor<1x12x64x64xf32>, %k: tensor<128x8x32x64xf32>, %v: tensor<128x8x32x64xf32>, %pt: tensor<1x4xi32>, %csi: tensor<1xi32>, %o: tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32> {
  // expected-error @+1 {{Query num heads must be divisible by key/value num heads.}}
  %1 = "ttir.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi, %o) : (tensor<1x12x64x64xf32>, tensor<128x8x32x64xf32>, tensor<128x8x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>, tensor<1x12x64x64xf32>) -> tensor<1x12x64x64xf32>
  return %1 : tensor<1x12x64x64xf32>
}

// -----

// Page table number of users (dim 0) must match query number of users.
func.func @page_table_users_mismatch(%q: tensor<2x12x64x64xf32>, %k: tensor<128x12x32x64xf32>, %v: tensor<128x12x32x64xf32>, %pt: tensor<1x4xi32>, %csi: tensor<1xi32>, %o: tensor<2x12x64x64xf32>) -> tensor<2x12x64x64xf32> {
  // expected-error @+1 {{Page table number of users must match query number of users.}}
  %1 = "ttir.chunked_scaled_dot_product_attention"(%q, %k, %v, %pt, %csi, %o) : (tensor<2x12x64x64xf32>, tensor<128x12x32x64xf32>, tensor<128x12x32x64xf32>, tensor<1x4xi32>, tensor<1xi32>, tensor<2x12x64x64xf32>) -> tensor<2x12x64x64xf32>
  return %1 : tensor<2x12x64x64xf32>
}
