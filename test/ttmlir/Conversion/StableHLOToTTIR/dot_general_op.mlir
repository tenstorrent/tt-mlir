// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_subtract attributes {} {
  func.func public @test_dot_general(%arg0 : tensor<32x32xf32>, %arg1 : tensor<32x32xf32>) -> tensor<32x32xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [0] x [0] : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.matmul"[[C:.*]]
    return %0 : tensor<32x32xf32>
  }
}
