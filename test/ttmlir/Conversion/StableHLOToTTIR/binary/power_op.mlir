// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_power attributes {} {
  func.func public @test_power(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.power %arg0, %arg1 : tensor<32x32xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<32x32xf32>
    // CHECK: %[[REM:[0-9]+]] = "ttir.pow"(%arg0, %arg1, %[[EMPTY]]){{.*}} -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
    // CHECK: return %[[REM]] : tensor<32x32xf32>
  }
}
