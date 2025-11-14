// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_ceil attributes {} {
  func.func public @test_ceil(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.ceil %arg0 : tensor<13x21x3xf32>
    // CHECK: "ttir.ceil"(%arg0)
    return %0 : tensor<13x21x3xf32>
  }
}
