// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_sqrt attributes {} {
  func.func public @test_sqrt(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.sqrt %arg0 : tensor<13x21x3xf32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.sqrt"
    return %0 : tensor<13x21x3xf32>
  }
}
