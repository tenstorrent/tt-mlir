// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @rmsnorm_bw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<1x1x128x1xbf16>,
      %grad_output: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK: "ttnn.rmsnorm_bw"
    %grad_input, %grad_gamma = "ttcore.composite"(%input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_decomp}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>,
           tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %grad_input, %grad_gamma
        : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }

  // A batched case, which exercises the trailing ttnn::sum over dims (0, 1, 2).
  func.func @rmsnorm_bw_batched(
      %input: tensor<2x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<2x1x128x1xbf16>,
      %grad_output: tensor<2x1x128x256xbf16>)
      -> (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK: "ttnn.rmsnorm_bw"
    %grad_input, %grad_gamma = "ttcore.composite"(%input, %gamma, %rms, %grad_output) <{
        composite_name = "rmsnorm_bw",
        decomposition = @rmsnorm_bw_batched_decomp}>
        : (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>,
           tensor<2x1x128x1xbf16>, tensor<2x1x128x256xbf16>)
          -> (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
    return %grad_input, %grad_gamma
        : tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>
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

  func.func private @rmsnorm_bw_batched_decomp(
      %input: tensor<2x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>,
      %rms: tensor<2x1x128x1xbf16>,
      %grad_output: tensor<2x1x128x256xbf16>)
      -> (tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>) {
    return %input, %gamma
        : tensor<2x1x128x256xbf16>, tensor<1x1x1x256xbf16>
  }
}
