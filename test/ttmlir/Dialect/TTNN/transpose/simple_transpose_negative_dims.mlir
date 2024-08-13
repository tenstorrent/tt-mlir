// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<8x8xbf16>) -> tensor<8x8xbf16> {
    %0 = tensor.empty() : tensor<8x8xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %1 = "ttir.transpose"(%arg0, %0) <{dimension1 = -1 : si32, dimension2 = -2 : si32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<8x8xbf16>, tensor<8x8xbf16>) -> tensor<8x8xbf16>
    return %1 : tensor<8x8xbf16>
  }
}
