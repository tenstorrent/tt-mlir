// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true memory-layout-analysis-enabled=true memory-layout-analysis-policy=L1Interleaved" %s | FileCheck %s
// UNSUPPORTED: true
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16> {
    %0 = tensor.empty() : tensor<5120x5120xbf16>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<5120x5120xbf16>, tensor<5120x5120xbf16>) -> tensor<5120x5120xbf16>
    return %1 : tensor<5120x5120xbf16>
  }
}
