// REQUIRES: stablehlo
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @cross_entropy_fw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xbf16> {
    // CHECK: "ttcore.composite"
    // CHECK-SAME: composite_name = "cross_entropy_fw"
    // CHECK-SAME: decomposition = @tenstorrent.cross_entropy_fw.impl
    // CHECK-SAME: (tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x1xbf16>
    // CHECK-NOT: stablehlo.composite
    %0 = stablehlo.composite "tenstorrent.cross_entropy_fw" %input, %target {
      decomposition = @tenstorrent.cross_entropy_fw.impl
    } : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
  func.func private @tenstorrent.cross_entropy_fw.impl(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>) -> tensor<4x1x32x1xbf16> {
    %0 = stablehlo.slice %input [0:4, 0:1, 0:32, 0:1] : (tensor<4x1x32x64xbf16>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}
