// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @silu_bw(
      %input: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    // CHECK: "ttnn.silu_bw"
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
