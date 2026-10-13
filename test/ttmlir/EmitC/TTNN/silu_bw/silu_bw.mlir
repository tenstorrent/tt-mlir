// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // CHECK-LABEL: silu_bw
  func.func @silu_bw(
      %input: tensor<1x1x128x256xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> tensor<1x1x128x256xbf16> {
    // CHECK: ttml::metal::silu_bw(
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
