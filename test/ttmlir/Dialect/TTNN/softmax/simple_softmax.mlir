// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    %0 = tensor.empty() : tensor<512x1024xbf16>
    // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
    // Check for positive dimension attribute
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    %2 = tensor.empty() : tensor<512x1024xbf16>
    // CHECK: %[[C:.*]] = "ttnn.softmax"[[C:.*]]
    // Check for negative dimension attribute
    %3 = "ttir.softmax"(%1, %2) <{dimension = -1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %3 : tensor<512x1024xbf16>
  }
}
