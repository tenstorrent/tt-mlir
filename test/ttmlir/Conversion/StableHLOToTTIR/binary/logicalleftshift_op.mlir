// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_shift_left_logical_op attributes {} {
  func.func public @test_sll(%arg0: tensor<5xui32>, %arg1: tensor<5xui32>) -> tensor<5xui32> {
    %0 = stablehlo.shift_left %arg0, %arg1 : tensor<5xui32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.logical_left_shift"
    return %0 : tensor<5xui32>
  }
}