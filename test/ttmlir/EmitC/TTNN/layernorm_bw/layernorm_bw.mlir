// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir | FileCheck %s

module {
  // CHECK-LABEL: layernorm_bw
  // CHECK: ttml::metal::layernorm_bw(
  // CHECK-COUNT-3: util_get_optional_value(
  func.func @layernorm_bw(
      %input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>,
      %mean: tensor<1x1x128x1xbf16>, %rstd: tensor<1x1x128x1xbf16>,
      %grad: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) {
    %0:3 = "ttcore.composite"(%input, %gamma, %mean, %rstd, %grad) <{
      composite_name = "layernorm_bw",
      decomposition = @layernorm_bw_decomp,
      composite_attributes = {}
    }> : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>
  }

  func.func private @layernorm_bw_decomp(
      %input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>,
      %mean: tensor<1x1x128x1xbf16>, %rstd: tensor<1x1x128x1xbf16>,
      %grad: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) {
    return %grad, %gamma, %gamma : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>
  }
}
