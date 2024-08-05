// RUN: ttmlir-opt --ttir-load-system-desc --ttir-layout --ttnn-open-device --convert-ttir-to-ttnn %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %[[C:.*]] = "ttnn.matmul"[[C:.*]]
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    return %1 : tensor<64x96xbf16>
  }
}
