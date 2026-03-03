// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_tan attributes {} {
  func.func public @test_tan(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.tan %arg0 : tensor<13x21x3xf32>
    // CHECK: "ttir.tan"(%arg0)
    return %0 : tensor<13x21x3xf32>
  }
}
