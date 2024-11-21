// RUN: ttmlir-opt --ttir-reshape-fold %s| FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|any_device_tile>
// Tests if we fold when translating from "ttir.reshape" which is called on the two same shapes.
module @reshape_test {
  func.func @main(%arg0: tensor<1xi32>) -> (tensor<1xi32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<1xi32>
    %1 = "ttir.reshape"(%arg0, %0) <{operand_constraints = [#any_device_tile, #any_device_tile], shape = [1 : i32]}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK: return %arg0 : tensor<1xi32>
    return %1 : tensor<1xi32>
  }
}
