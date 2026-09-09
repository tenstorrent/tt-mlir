// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.layernorm_fw is deliberately unsupported. ttml does
// not expose the metal::layernorm_fw primitive through its Python bindings.

module {
  func.func @layernorm_fw(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                          %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: failed to legalize operation 'ttnn.layernorm_fw'
    %0 = "ttcore.composite"(%input, %weight, %bias) <{
        composite_name = "layernorm_fw",
        decomposition = @layernorm_fw_decomp,
        composite_attributes = {
          epsilon = 1.000000e-05 : f32,
          return_mean_rstd = false}}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }

  func.func private @layernorm_fw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %weight: tensor<1x1x1x256xbf16>,
      %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %input : tensor<1x1x128x256xbf16>
  }
}
