// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s
// Negative tests for the ttnn.sdpa_bw operation.

// Dropout is not implemented in the backward pass, so a nonzero
// dropout_probability must be rejected rather than silently ignored.
module {
  func.func @sdpa_bw_nonzero_dropout(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // expected-error @+1 {{dropout_probability must be 0.0 because dropout is not implemented in the backward pass}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 3.000000e-01 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}

// -----

// The query sequence dimension must be tile-height aligned.
module {
  func.func @sdpa_bw_query_sequence_unaligned(
      %grad_output: tensor<1x8x31x64xbf16>, %attn_output: tensor<1x8x31x64xbf16>,
      %query: tensor<1x8x31x?xbf16>, %key: tensor<1x8x31x64xbf16>,
      %value: tensor<1x8x31x64xbf16>, %intermediates: tensor<1x8x31x32xf32>)
      -> (tensor<1x8x31x?xbf16>, tensor<1x8x31x64xbf16>, tensor<1x8x31x64xbf16>) {
    // expected-error @+1 {{'ttnn.sdpa_bw' op query sequence dimension (dim 2) must be a multiple of TILE_HEIGHT (32), but got 31}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x31x64xbf16>, tensor<1x8x31x64xbf16>, tensor<1x8x31x?xbf16>,
           tensor<1x8x31x64xbf16>, tensor<1x8x31x64xbf16>, tensor<1x8x31x32xf32>)
          -> (tensor<1x8x31x?xbf16>, tensor<1x8x31x64xbf16>, tensor<1x8x31x64xbf16>)
    return %0, %1, %2 : tensor<1x8x31x?xbf16>, tensor<1x8x31x64xbf16>, tensor<1x8x31x64xbf16>
  }
}

// -----

// The query head dimension must be tile-width aligned.
module {
  func.func @sdpa_bw_query_head_unaligned(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x33xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x33xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // expected-error @+1 {{'ttnn.sdpa_bw' op query head dimension (dim 3) must be a multiple of TILE_WIDTH (32), but got 33}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x33xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x33xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x33xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}

// -----

// The key sequence dimension must be tile-height aligned.
module {
  func.func @sdpa_bw_key_sequence_unaligned(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x33x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x33x64xbf16>, tensor<1x8x64x64xbf16>) {
    // expected-error @+1 {{'ttnn.sdpa_bw' op key sequence dimension (dim 2) must be a multiple of TILE_HEIGHT (32), but got 33}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x33x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x33x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x33x64xbf16>, tensor<1x8x64x64xbf16>
  }
}

// -----

// The key head dimension must be tile-width aligned.
module {
  func.func @sdpa_bw_key_head_unaligned(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x31xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x31xbf16>, tensor<1x8x64x64xbf16>) {
    // expected-error @+1 {{'ttnn.sdpa_bw' op key head dimension (dim 3) must be a multiple of TILE_WIDTH (32), but got 31}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x31xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x31xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x31xbf16>, tensor<1x8x64x64xbf16>
  }
}

// -----

// V has the same sequence alignment requirement as Q and K.
module {
  func.func @sdpa_bw_value_sequence_unaligned(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x31x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x31x64xbf16>) {
    // expected-error @+1 {{'ttnn.sdpa_bw' op value sequence dimension (dim 2) must be a multiple of TILE_HEIGHT (32), but got 31}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x31x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x31x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x31x64xbf16>
  }
}

// -----

// A dynamic sequence dimension does not hide a statically unaligned query
// head dimension.
module {
  func.func @sdpa_bw_dynamic_query_sequence_unaligned_head(
      %grad_output: tensor<1x8x?x64xbf16>, %attn_output: tensor<1x8x?x64xbf16>,
      %query: tensor<1x8x?x33xbf16>, %key: tensor<1x8x?x64xbf16>,
      %value: tensor<1x8x?x64xbf16>, %intermediates: tensor<1x8x?x32xf32>)
      -> (tensor<1x8x?x33xbf16>, tensor<1x8x?x64xbf16>, tensor<1x8x?x64xbf16>) {
    // expected-error @+1 {{'ttnn.sdpa_bw' op query head dimension (dim 3) must be a multiple of TILE_WIDTH (32), but got 33}}
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x?x64xbf16>, tensor<1x8x?x64xbf16>, tensor<1x8x?x33xbf16>,
           tensor<1x8x?x64xbf16>, tensor<1x8x?x64xbf16>, tensor<1x8x?x32xf32>)
          -> (tensor<1x8x?x33xbf16>, tensor<1x8x?x64xbf16>, tensor<1x8x?x64xbf16>)
    return %0, %1, %2 : tensor<1x8x?x33xbf16>, tensor<1x8x?x64xbf16>, tensor<1x8x?x64xbf16>
  }
}

// -----

// DiffVDim supports an unaligned V head dimension.
module {
  func.func @sdpa_bw_value_head_unaligned_supported(
      %grad_output: tensor<1x8x64x33xbf16>, %attn_output: tensor<1x8x64x33xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x33xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x33xbf16>) {
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x33xbf16>, tensor<1x8x64x33xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x33xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x33xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x33xbf16>
  }
}

// -----

// Dynamic dimensions are deferred to runtime.
module {
  func.func @sdpa_bw_dynamic_dimensions(
      %grad_output: tensor<1x8x?x?xbf16>, %attn_output: tensor<1x8x?x?xbf16>,
      %query: tensor<1x8x?x?xbf16>, %key: tensor<1x8x?x?xbf16>,
      %value: tensor<1x8x?x?xbf16>, %intermediates: tensor<1x8x?x32xf32>)
      -> (tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>) {
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>,
           tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>, tensor<1x8x?x32xf32>)
          -> (tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>)
    return %0, %1, %2 : tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>, tensor<1x8x?x?xbf16>
  }
}
