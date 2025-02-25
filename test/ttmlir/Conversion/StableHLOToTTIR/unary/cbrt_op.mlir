// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_rsqrt attributes {} {
  func.func public @test_cbrt(%arg0: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.cbrt %arg0 : tensor<4xf64>
    // CHECK: = tensor.empty
    // CHECK: = "ttir.cbrt"
    return %0 : tensor<4xf64>
  }
}
