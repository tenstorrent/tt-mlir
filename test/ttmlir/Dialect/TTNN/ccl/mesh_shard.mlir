// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#operand_constraint = #tt.operand_constraint<system|scalar|system_scalar>
module attributes {} {
  func.func @forward(%arg0: tensor<8192x784xf32>) -> tensor<4096x196xf32> {
    %0 = tensor.empty() : tensor<4096x196xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{operand_constraints = [#operand_constraint, #operand_constraint], shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<2x4>, shard_type = #tt.shard_type<devices>}> : (tensor<8192x784xf32>, tensor<4096x196xf32>) -> tensor<4096x196xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    return %1 : tensor<4096x196xf32>
  }
}
