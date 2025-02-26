// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_exp attributes {} {
  func.func public @test_exp(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.exponential %arg0 : tensor<13x21x3xf32>
    // CHECK: = tensor.empty
    // CHECK: = "ttir.exp"
    return %0 : tensor<13x21x3xf32>
  }
}
