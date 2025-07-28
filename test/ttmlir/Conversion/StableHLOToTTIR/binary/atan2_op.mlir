// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_eltwise_atan2 attributes {} {
  func.func public @test_atan2(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.atan2 %arg0, %arg1 : tensor<32x32xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = ttir.empty() : tensor<32x32xf32>
    // CHECK: %[[ATAN2:[0-9]+]] = "ttir.atan2"(%arg0, %arg1, %[[EMPTY]]){{.*}} -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
    // CHECK: return %[[ATAN2]] : tensor<32x32xf32>
  }
}
