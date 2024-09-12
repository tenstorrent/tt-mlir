#l1_block_sharded = #tt.operand_constraint<l1_block_sharded>

func.func @output_operand_constraint(%arg0: tensor<224x64xf32>) -> tensor<224x64xf32> {
  // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
  %0 = tensor.empty() : tensor<224x64xf32>
  // CHECK: %[[C:.*]] = "ttnn.relu"[[C:.*]]
  %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>, operand_constraints = [#l1_block_sharded, #l1_block_sharded]}> : (tensor<224x64xf32>, tensor<224x64xf32>) -> tensor<224x64xf32>
  return %1 : tensor<224x64xf32>
}
