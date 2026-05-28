// RUN: ttmlir-opt -split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module @mesh_shard_test attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @forward(%arg0: tensor<8192x784xf32>) -> tensor<8192x392xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<8192x784xf32>) -> tensor<8192x392xf32>
    return %0 : tensor<8192x392xf32>
  }
}
// CHECK: [[REG:.*]] = "ttnn.distribute_tensor"([[ARG:.*]], [[DEV:.*]]) <{
// CHECK-SAME: mapper_config
// CHECK-SAME: placements = [<shard, 0 : i32>, <shard, 1 : i32>]
// CHECK-SAME: mesh_shape_override = [1 : ui32, 2 : ui32]
