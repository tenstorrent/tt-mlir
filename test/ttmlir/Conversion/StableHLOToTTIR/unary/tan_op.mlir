// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_tan attributes {} {
  func.func public @test_tan(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.tan %arg0 : tensor<13x21x3xf32>
    // CHECK: [[VAL0:%[0-9]+]] = tensor.empty() : [[TENSOR_SIZE:tensor<[0-9]+x[0-9]+x[0-9]+xf[0-9]+>]]
    // CHECK: [[VAL1:%[0-9]+]] = "ttir.tan"(%arg0, [[VAL0]]) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : ([[TENSOR_SIZE]], [[TENSOR_SIZE]]) -> [[TENSOR_SIZE]]
    return %0 : tensor<13x21x3xf32>
  }
}
