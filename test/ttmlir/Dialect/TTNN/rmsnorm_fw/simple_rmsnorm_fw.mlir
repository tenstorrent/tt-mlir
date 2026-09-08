// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // RMS norm forward without the intermediate RMS tensor.
  func.func @rmsnorm_fw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: "ttnn.rmsnorm_fw"
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

  // RMS norm forward returning the RMS tensor used by the backward pass.
  func.func @rmsnorm_fw_intermediates(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>) {
    // CHECK: "ttnn.rmsnorm_fw"
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
