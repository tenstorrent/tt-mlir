// RUN: not ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s 2>&1 | FileCheck %s
// CHECK: error: 'ttir.softmax' op Dimension attribute must be within the bounds of the input tensor
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = tensor.empty() : tensor<512x1024xbf16>
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = -3 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }
}
