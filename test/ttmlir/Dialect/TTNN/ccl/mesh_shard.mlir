// RUN: ttmlir-opt -split-input-file --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @mesh_shard_test attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x1>]>} {
  func.func @forward(%arg0: tensor<8192x784xf32>) -> tensor<8192x392xf32> {
    %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: 0, 1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #ttcore.shard_type<devices>}> : (tensor<8192x784xf32>) -> tensor<8192x392xf32>
    return %0 : tensor<8192x392xf32>
  }
}
// CHECK: [[DEVICE:%[0-9]+]] = "ttnn.get_device"()
// CHECK-NEXT: [[REG:.*]] = "ttnn.mesh_shard"([[ARG:.*]], [[DEVICE]])
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 2>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
