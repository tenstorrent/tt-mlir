// RUN: ttmlir-opt -split-input-file --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

module @mesh_shard_test attributes {tt.meshes = #tt.meshes<[<"mesh" = 1x1>]>} {
  func.func @forward(%arg0: tensor<8192x784xf32>) -> tensor<8192x392xf32> {
    %0 = ttir.empty() : tensor<8192x392xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 0, 1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #tt.shard_type<devices>}> : (tensor<8192x784xf32>, tensor<8192x392xf32>) -> tensor<8192x392xf32>
    return %1 : tensor<8192x392xf32>
  }
}
// CHECK: [[DEVICE:%[0-9]+]] = "ttnn.get_device"()
// CHECK-NEXT: [[REG:.*]] = "ttnn.mesh_shard"([[ARG:.*]], [[DEVICE]])
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<devices>

// -----

module @mesh_shard_test attributes {tt.meshes = #tt.meshes<[<"mesh" = 1x1>]>} {
  func.func @forward(%arg0: tensor<8192x784xf32, #tt.mesh_sharding<"mesh">>) -> tensor<8192x392xf32, #tt.mesh_sharding<"mesh", [ 2(0),  1]>> {
    %0 = ttir.empty() : tensor<8192x392xf32, #tt.mesh_sharding<"mesh", [ 2(0),  1]>>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: 0, 1>, shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 2>, shard_type = #tt.shard_type<identity>}> : (tensor<8192x784xf32, #tt.mesh_sharding<"mesh">>, tensor<8192x392xf32, #tt.mesh_sharding<"mesh", [ 2(0),  1]>>) -> tensor<8192x392xf32, #tt.mesh_sharding<"mesh", [ 2(0),  1]>>
    return %1 : tensor<8192x392xf32, #tt.mesh_sharding<"mesh", [ 2(0),  1]>>
  }
}
// CHECK: [[DEVICE:%[0-9]+]] = "ttnn.get_device"()
// CHECK-NEXT: [[REG:.*]] = "ttnn.mesh_shard"([[ARG:.*]], [[DEVICE]])
// CHECK-SAME: shard_dims = array<i64: 0, 1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 2>
// CHECK-SAME: shard_type = #tt.shard_type<identity>
