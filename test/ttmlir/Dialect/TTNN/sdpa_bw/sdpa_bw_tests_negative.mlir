// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for the ttnn.sdpa_bw operation.

// Dropout is not implemented in the backward pass, so a nonzero
// dropout_probability must be rejected rather than silently ignored.
module {
  func.func @sdpa_bw_nonzero_dropout(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: error: 'ttnn.sdpa_bw' op dropout_probability must be 0.0 because dropout is not implemented in the backward pass
    %0, %1, %2 = "ttnn.sdpa_bw"(%grad_output, %attn_output, %query, %key, %value, %intermediates)
        <{mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 3.000000e-01 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
