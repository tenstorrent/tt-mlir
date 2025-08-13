// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_shift_right_logical_op attributes {} {
  func.func public @test_srl(%arg0: tensor<5xi32>, %arg1: tensor<5xi32>) -> tensor<5xi32> {
    %0 = stablehlo.shift_right_logical %arg0, %arg1 : tensor<5xi32>
    // CHECK: = ttir.empty
    // CHECK: = "ttir.shift_right_logical"
    return %0 : tensor<5xi32>
  }
}
