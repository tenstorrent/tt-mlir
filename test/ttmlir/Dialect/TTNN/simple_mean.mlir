// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x32xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<512x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.mean"[[C:.*]]
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x32xbf16>) -> tensor<512x32xbf16>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<512x32xbf16>
  }
}
