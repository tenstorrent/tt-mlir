// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Layer norm forward without the backward statistics.
  func.func @layernorm_fw(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                          %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: "ttnn.layernorm_fw"
    %0 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = false}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }

  // Layer norm forward returning the mean and rstd used by the backward pass.
  func.func @layernorm_fw_mean_rstd(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                                    %bias: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>) {
    // CHECK: "ttnn.layernorm_fw"
    %0, %1, %2 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = true}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>)
    return %0, %1, %2 : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>
  }
}
