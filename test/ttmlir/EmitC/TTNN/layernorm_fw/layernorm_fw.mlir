// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // Single (output) result.
  // CHECK-LABEL: layernorm_fw
  func.func @layernorm_fw(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                          %bias: tensor<1x1x1x256xbf16>) -> tensor<1x1x128x256xbf16> {
    // CHECK: ttml::metal::layernorm_fw(
    // CHECK: util_get_optional_value(
    %0 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = false}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> tensor<1x1x128x256xbf16>
    return %0 : tensor<1x1x128x256xbf16>
  }

  // output + mean + rstd: all three results are unpacked.
  // CHECK-LABEL: layernorm_fw_mean_rstd
  func.func @layernorm_fw_mean_rstd(%input: tensor<1x1x128x256xbf16>, %weight: tensor<1x1x1x256xbf16>,
                                    %bias: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>) {
    // CHECK: ttml::metal::layernorm_fw(
    // CHECK-COUNT-3: util_get_optional_value(
    %0, %1, %2 = "ttir.layernorm_fw"(%input, %weight, %bias) <{
        epsilon = 1.000000e-05 : f32,
        return_mean_rstd = true}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>)
    return %0, %1, %2 : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>
  }
}
