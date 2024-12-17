// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_dnamic_iota attributes {} {
  func.func public @test_dynamic_iota() -> tensor<1x32x128x128xf32> {
    // CHECK: %[[C:.*]] = "ttir.arange"[[C:.*]]
    %output_shape = stablehlo.constant dense<[1, 32, 128, 128]> : tensor<4xi64>
    %0 = "stablehlo.dynamic_iota"(%output_shape) {iota_dimension = 1: i64} : (tensor<4xi64>) -> tensor<1x32x128x128xf32>
    return %0 : tensor<1x32x128x128xf32>
  }
}
