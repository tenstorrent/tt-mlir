// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    %0 = tensor.empty() : tensor<128x64xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
    %2 = tensor.empty() : tensor<64x128xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose
    %3 = "ttir.transpose"(%0, %2) <{dim0 = 1 : si32, dim1 = 0 : si32, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<128x64xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %3 : tensor<64x128xbf16>
  }
}
