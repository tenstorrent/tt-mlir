// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // CHECK-LABEL: swiglu_elemwise_bw
  func.func @swiglu_elemwise_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gate: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    // CHECK: ttml::metal::swiglu_elemwise_bw(
    // CHECK: .dL_dlinear1
    // CHECK: .dL_dgate
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
