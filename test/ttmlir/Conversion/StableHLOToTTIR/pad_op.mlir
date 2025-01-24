// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x128x128x384xf32>) -> tensor<1x132x132x384xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xf64>
    %0 = stablehlo.convert %cst : (tensor<1xf64>) -> tensor<1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1xf32>) -> tensor<f32>
    %2 = stablehlo.pad %arg0, %1, low = [0, 0, 0, 0], high = [0, 4, 4, 0], interior = [0, 0, 0, 0] : (tensor<1x128x128x384xf32>, tensor<f32>) -> tensor<1x132x132x384xf32>
    return %2 : tensor<1x132x132x384xf32>
  }
}
