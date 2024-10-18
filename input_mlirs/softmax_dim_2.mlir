module attributes {} {
  func.func @forward(%arg0: tensor<1x160x96xf32> {ttir.name = "input"}) -> (tensor<1x160x96xf32> {ttir.name = "ModelFromDramQueue.output_softmax_0"}) {
    %0 = tensor.empty() : tensor<1x160x96xf32>
    %1 = "ttir.softmax"(%arg0, %0) <{dimension = 2 : si32, operand_constraints = [#tt.operand_constraint<dram|l1|scalar|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device|any_device_tile|l1_block_sharded>, #tt.operand_constraint<dram|l1|scalar|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device|any_device_tile|l1_block_sharded>]}> : (tensor<1x160x96xf32>, tensor<1x160x96xf32>) -> tensor<1x160x96xf32>
    return %1 : tensor<1x160x96xf32>
  }
}
