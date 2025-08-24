// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_abs attributes {} {
  func.func public @test_abs(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.abs %arg0 : tensor<13x21x3xf32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.abs"
    return %0 : tensor<13x21x3xf32>
  }
}
