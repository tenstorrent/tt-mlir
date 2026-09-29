// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

// EmitPy lowering of ttnn.swiglu_elemwise_bw is deliberately unsupported. TTML
// does not expose the metal::swiglu_elemwise_bw primitive through its Python
// bindings.

module {
  func.func @swiglu_elemwise_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gate: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    // CHECK: failed to legalize operation 'ttnn.swiglu_elemwise_bw'
    %grad_input, %grad_gate = "ttcore.composite"(%input, %gate, %grad_output) <{
        composite_name = "swiglu_elemwise_bw",
        decomposition = @swiglu_elemwise_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>)
    return %grad_input, %grad_gate
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }

  func.func private @swiglu_elemwise_bw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gate: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    return %input, %gate
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>
  }
}
