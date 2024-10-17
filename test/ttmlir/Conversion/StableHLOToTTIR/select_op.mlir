// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_select attributes {} {
  func.func public @test_select(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
    %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xi1>
    %1 = stablehlo.select %0, %arg0, %arg1 : (tensor<13x37xi1>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty()
    // CHECK: %[[VAL1:[0-9]+]] = "ttir.eq"
    // CHECK: %[[SELECT:[0-9]+]] = "ttir.where"(%[[VAL1:[0-9]+]], %arg0, %arg1, %[[EMPTY:[0-9]+]]) <{operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<13x37xbf16>, tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    return %1 : tensor<13x37xf32>
  }
}
