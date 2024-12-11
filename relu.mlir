// RUN: ttmlir-opt --ttir-load-system-desc --ttir-implicit-device --ttir-layout %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<8x64x128xf32>) -> tensor<8x64x128xf32> {
    // CHECK: %[[C:.*]] = tensor.empty() : tensor<8x64x128xf32, #layout>
    %0 = "ttir.constant"() {value = dense<0.0>: tensor<8x64x128xf32> } : () -> tensor<8x64x128xf32>
    %1 = tensor.empty() : tensor<8x64x128xf32>
    %2 = "ttir.maximum"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<8x64x128xf32>, tensor<8x64x128xf32>, tensor<8x64x128xf32>) -> tensor<8x64x128xf32>
    return %2 : tensor<8x64x128xf32>
  }

}
