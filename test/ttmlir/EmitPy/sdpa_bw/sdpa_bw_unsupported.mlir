// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.sdpa_bw is deliberately unsupported. ttml does not
// expose the metal::sdpa_bw primitive through its Python bindings.

module {
  func.func @sdpa_bw(
      %grad_output: tensor<1x8x64x64xbf16>, %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>) {
    // CHECK: failed to legalize operation 'ttnn.sdpa_bw'
    %0, %1, %2 = "ttcore.composite"(%grad_output, %attn_output, %query, %key, %value, %intermediates) <{
        composite_name = "sdpa_bw",
        decomposition = @sdpa_bw_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
    return %0, %1, %2 : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_bw_decomposition(
      %grad_output: tensor<1x8x64x64xbf16>,
      %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>,
      %intermediates: tensor<1x8x64x32xf32>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
          tensor<1x8x64x64xbf16>) {
    return %query, %key, %value : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }
}
