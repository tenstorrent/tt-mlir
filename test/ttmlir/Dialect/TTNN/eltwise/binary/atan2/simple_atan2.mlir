// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<3xf32>) -> tensor<3xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<3xf32>
    // CHECK: %[[C:.*]] = "ttnn.atan2"[[C:.*]]
    %1 = "ttir.atan2"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<3xf32>, tensor<3xf32>) -> tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
