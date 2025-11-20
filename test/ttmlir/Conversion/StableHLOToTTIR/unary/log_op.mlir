// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_log attributes {} {
  func.func public @test_log(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.log %arg0 : tensor<13x21x3xf32>
    // CHECK: "ttir.log"(%arg0)
    return %0 : tensor<13x21x3xf32>
  }
}
