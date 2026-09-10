// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // Output only.
  // CHECK-LABEL: rmsnorm_fw
  func.func @rmsnorm_fw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: ttml::metal::rmsnorm_fw(
    // CHECK-COUNT-1: util_get_optional_value(
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

  // Output and RMS intermediate.
  // CHECK-LABEL: rmsnorm_fw_intermediates
  func.func @rmsnorm_fw_intermediates(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>) {
    // CHECK: ttml::metal::rmsnorm_fw(
    // CHECK-COUNT-2: util_get_optional_value(
    %output, %rms = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_intermediates_decomp,
        composite_attributes = {
          return_intermediates = true,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>)
    return %output, %rms
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>
  }

  func.func private @rmsnorm_fw_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    return %input : tensor<1x1x128x256xbf16>
  }

  func.func private @rmsnorm_fw_intermediates_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>) {
    %rms = "ttir.zeros"() <{shape = array<i32: 1, 1, 128, 1>}>
        : () -> tensor<1x1x128x1xbf16>
    return %input, %rms
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>
  }
}
