// RUN: --ttnn-optimizer="memory-layout-analysis-enabled=true" %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>, %arg2: tensor<64x96xbf16>, %arg3: tensor<96x32xbf16>, %arg4: tensor<64x32xbf16>) -> tensor<64x32xbf16> {
    %0 = tensor.empty() : tensor<64x96xbf16>
    // CHECK: %[[C:.*]] = "ttnn.matmul"[[C:.*]]
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %2 = tensor.empty() : tensor<64x96xbf16>
    %3 = "ttir.add"(%1, %arg2, %2) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %4 = tensor.empty() : tensor<64x96xbf16>
    %5 = "ttir.relu"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
    %6 = tensor.empty() : tensor<64x32xbf16>
    %7 = "ttir.matmul"(%5, %arg3, %6) <{operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x96xbf16>, tensor<96x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %8 = tensor.empty() : tensor<64x32xbf16>
    %9 = "ttir.add"(%7, %arg4, %8) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device, #any_device, #any_device]}> : (tensor<64x32xbf16>, tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    %10 = tensor.empty() : tensor<64x32xbf16>
    %11 = "ttir.relu"(%9, %10) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device, #any_device]}> : (tensor<64x32xbf16>, tensor<64x32xbf16>) -> tensor<64x32xbf16>
    return %11 : tensor<64x32xbf16>
  }
}