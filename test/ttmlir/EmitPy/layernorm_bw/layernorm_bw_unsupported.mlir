// RUN: not ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% composite-resolution=force-promote" %s 2>&1 | FileCheck %s

module {
  func.func @layernorm_bw(
      %input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>,
      %mean: tensor<1x1x128x1xbf16>, %rstd: tensor<1x1x128x1xbf16>,
      %grad: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK: failed to legalize operation 'ttnn.layernorm_bw'
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
