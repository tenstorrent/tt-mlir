// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<32xf32>, %arg1: tensor<512x128xf32>) -> tensor<32x128xf32> {
    %0 = tensor.empty() : tensor<32x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.embedding"[[C:.*]]
    %1 = "ttir.embedding"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<32xf32>, tensor<512x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
}
