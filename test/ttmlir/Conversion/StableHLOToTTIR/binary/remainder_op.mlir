// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_remainder attributes {} {
  func.func public @test_remainder(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.remainder %arg0, %arg1 : tensor<32x32xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<32x32xf32>
    // CHECK: %[[REM:[0-9]+]] = "ttir.remainder"(%arg0, %arg1, %[[EMPTY]]){{.*}} -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
    // CHECK: return %[[REM]] : tensor<32x32xf32>
  }
}
