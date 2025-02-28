// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_divice attributes {} {
  func.func public @test_divide(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.divide %arg0, %arg1 : tensor<13x21x3xf32>
    // CHECK: = tensor.empty
    // CHECK: = "ttir.div"
    return %0 : tensor<13x21x3xf32>
  }
}
