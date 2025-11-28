// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_cosine attributes {} {
  func.func public @test_cosine(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.cosine %arg0 : tensor<13x21x3xf32>
    // CHECK: "ttir.cos"(%arg0)
    return %0 : tensor<13x21x3xf32>
  }
}
