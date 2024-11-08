// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_get_dimension_size attributes {} {
  func.func public @test_get_dimension_size(%arg0: tensor<13x21x3xf32>) -> tensor<i32> {
    %0 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<13x21x3xf32>) -> tensor<i32>
    // CHECK: [[VAL:%[0-9]+]] = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<{{[0-9]+}}x{{[0-9]+}}x{{[0-9]+}}xf32>) -> tensor<1xi32>
    return %0 : tensor<i32>
    // CHECK: return [[VAL]] : tensor<1xi32>
  }
}
