// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_eltwise_atan2 attributes {} {
  func.func public @test_atan2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<32x32xf32>
    // CHECK: %[[C:.*]] = ttir.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.atan2"[[C:.*]]
    return %0 : tensor<32x32xf32>
  }
}
