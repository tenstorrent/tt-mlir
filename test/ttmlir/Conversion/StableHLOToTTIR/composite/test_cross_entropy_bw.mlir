// REQUIRES: stablehlo
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @cross_entropy_bw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    // CHECK: "ttir.cross_entropy_bw"
    // CHECK-SAME: scaler = 3.125000e-02 : f32
    // CHECK-SAME: (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    // CHECK-NOT: stablehlo.composite
    %0 = stablehlo.composite "tenstorrent.cross_entropy_bw" %input, %target, %grad {
      composite_attributes = {scaler = 3.125e-02 : f32},
      decomposition = @tenstorrent.cross_entropy_bw.impl
    } : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
  func.func private @tenstorrent.cross_entropy_bw.impl(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16> {
    return %input : tensor<4x1x32x64xbf16>
  }
}
