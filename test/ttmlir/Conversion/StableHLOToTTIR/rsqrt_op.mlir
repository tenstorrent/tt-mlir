// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_rsqrt attributes {} {
  func.func public @test_rsqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.rsqrt %arg0 : tensor<13x21x3xf32>
    // CHECK: = tensor.empty
    // CHECK: = "ttir.rsqrt"
    return %0 : tensor<13x21x3xf32>
  }
}
