// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
    %0 = tensor.empty() : tensor<8x8xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %1 = "ttir.transpose"(%arg0, %0) <{dimension1 = 0 : si32, dimension2 = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>
    return %1 : tensor<8x8xbf16>
  }
}
