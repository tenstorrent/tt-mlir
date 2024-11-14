// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @clamp(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = tensor.empty() : tensor<64x128xbf16>
    // CHECK: %[[DEVICE:.*]] = "ttnn.to_device"(%arg0,
    // CHECK: %[[LAYOUT:.*]] = "ttnn.to_layout"(%[[DEVICE]])
    // CHECK: = "ttnn.clamp"(%[[LAYOUT]])
    // CHECK-SAME: {max = 3.000000e+00 : f32, min = 2.000000e+00 : f32}
    // CHECK-SAME: [[TENSOR:tensor<64x128xbf16]], #layout{{[0-9]+}}>) -> [[TENSOR]]
    %1 = "ttir.clamp"(%arg0, %0) <{max = 3.000000e+00 : f32, min = 2.000000e+00 : f32, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %1 : tensor<64x128xbf16>
  }
}