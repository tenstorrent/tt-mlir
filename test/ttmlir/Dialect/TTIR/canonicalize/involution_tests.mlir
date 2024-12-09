// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
module {
  func.func @involution_two_in_the_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK-NOT: "ttir.neg"
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = "ttir.neg"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = "ttir.neg"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }

  func.func @involution_three_in_the_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.neg"
    // CHECK-NOT: "ttir.neg"
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = "ttir.neg"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = "ttir.neg"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = "ttir.neg"(%3, %4) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#any_device_tile, #any_device_tile]}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %5 : tensor<64x64xf32>
  }
}
