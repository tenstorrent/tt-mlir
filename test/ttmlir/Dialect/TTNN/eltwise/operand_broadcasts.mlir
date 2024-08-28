// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @bcast_one_dim(%arg0: tensor<2x256x512xf32>, %arg1: tensor<256x512xf32>) -> tensor<2x256x512xf32> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<2x256x512xf32>
    // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<2x256x512xf32>, tensor<256x512xf32>, tensor<2x256x512xf32>) -> tensor<2x256x512xf32>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<2x256x512xf32>
  }

  func.func @bcast_multi_dim(%arg0: tensor<256x128x64x32xf32>, %arg1: tensor<64x32xf32>) -> tensor<256x128x64x32xf32> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<256x128x64x32xf32>
    // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]]
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<256x128x64x32xf32>, tensor<64x32xf32>, tensor<256x128x64x32xf32>) -> tensor<256x128x64x32xf32>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<256x128x64x32xf32>
  }

}
