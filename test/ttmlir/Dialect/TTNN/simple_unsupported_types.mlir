// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>
module {
  func.func @main(%arg0: tensor<1x1x12x16xi32>) -> tensor<1x192xi32> {
    %0 = tensor.empty() : tensor<1x192xi32>
    %1 = "ttir.reshape"(%arg0, %0) <{operand_constraints = [#any_device_tile, #any_device_tile], shape = [1 : i32, 192 : i32]}> : (tensor<1x1x12x16xi32>, tensor<1x192xi32>) -> tensor<1x192xi32>
    // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
    return %1 : tensor<1x192xi32>
  }
}
