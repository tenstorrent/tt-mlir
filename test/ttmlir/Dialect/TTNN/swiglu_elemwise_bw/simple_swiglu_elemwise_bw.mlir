// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @swiglu_elemwise_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gate: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x256xbf16>) {
    // CHECK: "ttnn.swiglu_elemwise_bw"
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
