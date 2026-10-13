// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.silu_bw is deliberately unsupported. TTML does not
// expose the metal::silu_bw primitive through its Python bindings.

module {
  func.func @silu_bw(
      %input: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    // CHECK: failed to legalize operation 'ttnn.silu_bw'
    %grad_input = "ttcore.composite"(%input, %grad_output) <{
        composite_name = "silu_bw",
        decomposition = @silu_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %grad_input : tensor<1x1x128x256xbf16>
  }

  func.func private @silu_bw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    return %input : tensor<1x1x128x256xbf16>
  }
}
