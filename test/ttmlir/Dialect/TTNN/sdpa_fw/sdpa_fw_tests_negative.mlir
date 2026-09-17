// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// The query sequence dimension must be tile-height aligned.
module {
  func.func @sdpa_fw_query_sequence_unaligned(
      %query: tensor<1x8x31x?xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x31x64xbf16> {
    // expected-error @+1 {{'ttnn.sdpa_fw' op query sequence dimension (dim 2) must be a multiple of TILE_HEIGHT (32), but got 31}}
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x31x?xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>) -> tensor<1x8x31x64xbf16>
    return %0 : tensor<1x8x31x64xbf16>
  }
}

// -----

// The query head dimension must be tile-width aligned.
module {
  func.func @sdpa_fw_query_head_unaligned(
      %query: tensor<1x8x64x33xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // expected-error @+1 {{'ttnn.sdpa_fw' op query head dimension (dim 3) must be a multiple of TILE_WIDTH (32), but got 33}}
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x33xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}

// -----

// The key sequence dimension must be tile-height aligned.
module {
  func.func @sdpa_fw_key_sequence_unaligned(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x33x64xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // expected-error @+1 {{'ttnn.sdpa_fw' op key sequence dimension (dim 2) must be a multiple of TILE_HEIGHT (32), but got 33}}
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x33x64xbf16>,
           tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}

// -----

// The key head dimension must be tile-width aligned.
module {
  func.func @sdpa_fw_key_head_unaligned(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x31xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // expected-error @+1 {{'ttnn.sdpa_fw' op key head dimension (dim 3) must be a multiple of TILE_WIDTH (32), but got 31}}
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x31xbf16>,
           tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}

// -----

// V has the same sequence alignment requirement as Q and K.
module {
  func.func @sdpa_fw_value_sequence_unaligned(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x31x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // expected-error @+1 {{'ttnn.sdpa_fw' op value sequence dimension (dim 2) must be a multiple of TILE_HEIGHT (32), but got 31}}
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x31x64xbf16>) -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}

// -----

// A dynamic sequence dimension does not hide a statically unaligned query
// head dimension.
module {
  func.func @sdpa_fw_dynamic_query_sequence_unaligned_head(
      %query: tensor<1x8x?x33xbf16>, %key: tensor<1x8x?x64xbf16>,
      %value: tensor<1x8x?x64xbf16>) -> tensor<1x8x?x64xbf16> {
    // expected-error @+1 {{'ttnn.sdpa_fw' op query head dimension (dim 3) must be a multiple of TILE_WIDTH (32), but got 33}}
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x?x33xbf16>, tensor<1x8x?x64xbf16>,
           tensor<1x8x?x64xbf16>) -> tensor<1x8x?x64xbf16>
    return %0 : tensor<1x8x?x64xbf16>
  }
}

// -----

// An unaligned V head dimension is supported.
module {
  func.func @sdpa_fw_value_head_unaligned_supported(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x33xbf16>) -> tensor<1x8x64x33xbf16> {
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x33xbf16>) -> tensor<1x8x64x33xbf16>
    return %0 : tensor<1x8x64x33xbf16>
  }
}

// -----

// Dynamic dimensions are deferred to runtime.
module {
  func.func @sdpa_fw_dynamic_dimensions(
      %query: tensor<1x8x?x?xbf16>, %key: tensor<1x8x?x?xbf16>,
      %value: tensor<1x8x?x?xbf16>) -> tensor<1x8x?x?xbf16> {
    %0 = "ttnn.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>,
           tensor<1x8x?x?xbf16>) -> tensor<1x8x?x?xbf16>
    return %0 : tensor<1x8x?x?xbf16>
  }
}
