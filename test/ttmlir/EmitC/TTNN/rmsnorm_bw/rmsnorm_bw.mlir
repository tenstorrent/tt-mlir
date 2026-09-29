// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // CHECK-LABEL: rmsnorm_bw
  func.func @rmsnorm_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<1x1x128x1xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK: ttml::metal::rmsnorm_bw(
    // CHECK-COUNT-2: util_get_optional_value(
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
