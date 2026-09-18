// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.rmsnorm_fw is deliberately unsupported. TTML does
// not expose the metal::rmsnorm_fw primitive through its Python bindings.

module {
  func.func @rmsnorm_fw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: failed to legalize operation 'ttnn.rmsnorm_fw'
    %output = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_decomp,
        composite_attributes = {
          return_intermediates = false,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %output : tensor<1x1x128x256xbf16>
  }

  func.func private @rmsnorm_fw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %input : tensor<1x1x128x256xbf16>
  }
}
