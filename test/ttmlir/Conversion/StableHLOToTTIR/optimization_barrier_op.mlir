// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_eltwise_optimization_barrier attributes {} {
  func.func public @test_optimization_barrier(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0:2 = stablehlo.optimization_barrier %arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>
    // CHECK: %0 = "ttir.optimization_barrier"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %1 = "ttir.optimization_barrier"(%arg1) : (tensor<64x128xf32>) -> tensor<64x128xf32>
    %1 = stablehlo.add %0#0, %0#1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %3 = "ttir.add"(%0, %1, %2) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
