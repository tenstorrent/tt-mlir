// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module attributes {} {
  func.func @main(%arg0: tensor<32x32x3xf32>) -> tensor<32x32x3xf32> {
    %0 = stablehlo.round_nearest_even %arg0 : tensor<32x32x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.round_nearest_even"[[C:.*]]
    return %0 : tensor<32x32x3xf32>
  }
}
