// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>) -> tensor<128x64xbf16> {
    %0 = tensor.empty() : tensor<128x64xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %1 = "ttir.transpose"(%arg0, %0) <{dimension1 = 0 : si32, dimension2 = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    return %1 : tensor<128x64xbf16>
  }
}
