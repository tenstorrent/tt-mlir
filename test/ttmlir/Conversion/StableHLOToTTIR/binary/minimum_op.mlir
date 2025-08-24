// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_minimum attributes {} {
  func.func public @test_minimum(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    // CHECK: = ttir.empty()
    // CHECK-SAME: [[TENSOR:tensor<13x21x3xf32>]]
    // CHECK: = "ttir.minimum"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }
}
