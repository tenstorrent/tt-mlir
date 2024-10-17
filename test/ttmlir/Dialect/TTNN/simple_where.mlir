// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>
module @jit_eltwise_where {
  func.func public @test_where(%arg0: tensor<13x37xf32>, %arg1: tensor<13x37xf32>) -> tensor<13x37xf32> {
    %0 = tensor.empty() : tensor<13x37xf32>
    %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>, operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
    %2 = tensor.empty() : tensor<13x37xf32>
    %3 = "ttir.where"(%1, %arg0, %arg1, %2) <{operand_constraints = [#any_device_tile, #any_device_tile, #any_device_tile, #any_device_tile]}> : (tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>, tensor<13x37xf32>) -> tensor<13x37xf32>
     // CHECK: %[[EMPTY:.*]] = "ttnn.empty"{{.*}}
     // CHECK: %[[VAL1:[0-9]+]] = "ttnn.eq"(%{{[0-9]+}}, %{{[0-9]+}}, %[[EMPTY]])
     // CHECK: %{{[0-9]+}} = "ttnn.where"(%[[VAL1]], %{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
     return %3 : tensor<13x37xf32>
  }
}
