// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.sdpa_fw is deliberately unsupported. ttml does not
// expose the metal::sdpa_fw primitive through its Python bindings.

module {
  func.func @sdpa_fw(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                     %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK: failed to legalize operation 'ttnn.sdpa_fw'
    %0 = "ttcore.composite"(%query, %key, %value) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = false}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_fw_decomposition(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    return %query : tensor<1x8x64x64xbf16>
  }
}
