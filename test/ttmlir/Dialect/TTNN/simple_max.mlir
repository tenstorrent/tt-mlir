// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<512x32xbf16>) -> tensor<512xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<512xbf16>
    // CHECK: %[[C:.*]] = "ttnn.max"[[C:.*]]
    %1 = "ttir.max"(%arg0, %0) <{dim_arg = [1: i32], keep_dim = false, operand_constraints = [#any_device, #any_device]}> : (tensor<512x32xbf16>, tensor<512xbf16>) -> tensor<512xbf16>
    return %1 : tensor<512xbf16>
  }
}
