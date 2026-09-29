// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.rmsnorm_bw is deliberately unsupported. TTML does
// not expose the metal::rmsnorm_bw primitive through its Python bindings.

module {
  func.func @rmsnorm_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<1x1x128x1xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK: failed to legalize operation 'ttnn.rmsnorm_bw'
    %grad_input, %grad_gamma = "ttcore.composite"(%input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>,
           tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %grad_input, %grad_gamma
        : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }

  func.func private @rmsnorm_bw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<1x1x128x1xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    return %input, %gamma
        : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}
