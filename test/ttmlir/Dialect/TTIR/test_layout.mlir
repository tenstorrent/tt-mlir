// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-layout %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<8x64x128xf32>, %arg1: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
    // CHECK: %[[C:.*]] = tensor.empty() : tensor<8x64x128xf32, #layout>
    %0 = tensor.empty() : tensor<8x64x128xf32>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<8x64x128xf32>, tensor<8x64x128xf32>, tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    return %1 : tensor<8x64x128xf32>
  }

  func.func @test_unused_argument(%arg0: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
    // CHECK: %[[C:.*]] = tensor.empty() : tensor<8x64x128xf32, #layout>
    %0 = tensor.empty() : tensor<8x64x128xf32>
    return %0 : tensor<8x64x128xf32>
  }
}
