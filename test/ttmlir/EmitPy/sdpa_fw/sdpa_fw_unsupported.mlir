// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path%" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.sdpa_fw is deliberately unsupported. ttml does not
// expose the metal::sdpa_fw primitive through its Python bindings.

module {
  func.func @sdpa_fw(%query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
                     %value: tensor<1x8x64x64xbf16>) -> tensor<1x8x64x64xbf16> {
    // CHECK: failed to legalize operation 'ttnn.sdpa_fw'
    %0 = "ttir.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>)
          -> tensor<1x8x64x64xbf16>
    return %0 : tensor<1x8x64x64xbf16>
  }
}
