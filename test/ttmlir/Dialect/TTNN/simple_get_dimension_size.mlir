// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<13x21x3xf32>) -> tensor<1xi32> {
    %0 = "ttir.get_dimension_size"(%arg0) <{dimension = 1 : i32}> : (tensor<13x21x3xf32>) -> tensor<1xi32>
    // CHECK: [[VAL:%[0-9]+]] = "ttnn.full"(%{{[0-9]+}}) <{fillValue = 2.100000e+01 : f32}> : (!tt.device<#device>) -> tensor<1xi32, #layout1>
    return %0 : tensor<1xi32>
    // CHECK: return [[VAL]] : tensor<1xi32, #layout1>
  }
}
