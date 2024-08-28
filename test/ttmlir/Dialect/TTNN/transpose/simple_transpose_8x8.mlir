// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = tensor.empty() : tensor<32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
    %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32, operand_constraints = [#any_device, #any_device]}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
