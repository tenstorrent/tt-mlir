// REQUIRES: stablehlo
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir %s | FileCheck %s

module {
  func.func @layernorm_bw(
      %input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>,
      %mean: tensor<1x1x128x1xbf16>, %rstd: tensor<1x1x128x1xbf16>,
      %grad: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) {
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_attributes = {}
    // CHECK-SAME: composite_name = "layernorm_bw"
    // CHECK-SAME: decomposition = @tenstorrent.layernorm_bw.impl
    // CHECK-NOT: stablehlo.composite
    %0:3 = stablehlo.composite "tenstorrent.layernorm_bw" %input, %gamma, %mean, %rstd, %grad {
      decomposition = @tenstorrent.layernorm_bw.impl
    } : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x1xbf16>, tensor<1x1x128x256xbf16>) -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>)
    return %0#0, %0#1, %0#2 : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>
  }

  func.func private @tenstorrent.layernorm_bw.impl(
      %input: tensor<1x1x128x256xbf16>, %gamma: tensor<1x1x1x256xbf16>,
      %mean: tensor<1x1x128x1xbf16>, %rstd: tensor<1x1x128x1xbf16>,
      %grad: tensor<1x1x128x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>) {
    return %grad, %gamma, %gamma : tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>, tensor<1x1x1x256xbf16>
  }
}
