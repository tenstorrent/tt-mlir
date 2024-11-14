// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x32x64x128xf32>) -> tensor<1x128x32x64xf32> {
    // CHECK: %[[C:.*]] = "ttir.transpose"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.transpose"[[C:.*]]
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x32x64x128xf32>) -> tensor<1x128x32x64xf32>
    return %0 : tensor<1x128x32x64xf32>
  }
}