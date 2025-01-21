// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=4,8" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// REQUIRES: multi-chip-x32

module @jit_fwd_six attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<2048x98xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<4x8>, shard_type = #tt.shard_type<devices>}> : (tensor<8192x784xf32>, tensor<2048x98xf32>) -> tensor<2048x98xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %2 = tensor.empty() : tensor<98x2048xf32>
    %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<8x1>, shard_type = #tt.shard_type<devices>}> : (tensor<784x2048xf32>, tensor<98x2048xf32>) -> tensor<98x2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %4 = tensor.empty() : tensor<2048xf32>
    %5 = "ttir.mesh_shard"(%arg2, %4) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %6 = call @shmap_body(%1, %3, %5) : (tensor<2048x98xf32>, tensor<98x2048xf32>, tensor<2048xf32>) -> tensor<2048x2048xf32>
    %7 = tensor.empty() : tensor<8192x2048xf32>
    %8 = "ttir.mesh_shard"(%6, %7) <{shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = #tt.grid<4x1>, shard_type = #tt.shard_type<devices>}> : (tensor<2048x2048xf32>, tensor<8192x2048xf32>) -> tensor<8192x2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    return %8 : tensor<8192x2048xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x98xf32>, %arg1: tensor<98x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<2048x2048xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<2048x98xf32>, tensor<98x2048xf32>) -> tensor<2048x2048xf32>
    %1 = tensor.empty() : tensor<2048x2048xf32>
    %2 = "ttir.all_reduce"(%0, %1) <{channel_handle = 1 : si32, dim = 1 : si32, reduce_type = #tt.reduce_type<sum>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %3 = tensor.empty() : tensor<1x2048xf32>
    %4 = "ttir.reshape"(%arg2, %3) <{shape = [1 : i32, 2048 : i32]}> : (tensor<2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %5 = tensor.empty() : tensor<1x2048xf32>
    %6 = "ttir.broadcast"(%4, %5) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %7 = tensor.empty() : tensor<2048x2048xf32>
    %8 = "ttir.broadcast"(%6, %7) <{broadcast_dimensions = array<i32: 2048, 1>}> : (tensor<1x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %9 = tensor.empty() : tensor<2048x2048xf32>
    %10 = "ttir.add"(%2, %8, %9) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %10 : tensor<2048x2048xf32>
  }
}

module @jit_fwd_eight attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<2048x98xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<4x8>, shard_type = #tt.shard_type<devices>}> : (tensor<8192x784xf32>, tensor<2048x98xf32>) -> tensor<2048x98xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %2 = tensor.empty() : tensor<98x2048xf32>
    %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<8x1>, shard_type = #tt.shard_type<devices>}> : (tensor<784x2048xf32>, tensor<98x2048xf32>) -> tensor<98x2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %4 = tensor.empty() : tensor<2048xf32>
    %5 = "ttir.mesh_shard"(%arg2, %4) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %6 = call @shmap_body(%1, %3, %5) : (tensor<2048x98xf32>, tensor<98x2048xf32>, tensor<2048xf32>) -> tensor<2048x2048xf32>
    %7 = tensor.empty() : tensor<8192x2048xf32>
    %8 = "ttir.mesh_shard"(%6, %7) <{shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = #tt.grid<4x1>, shard_type = #tt.shard_type<devices>}> : (tensor<2048x2048xf32>, tensor<8192x2048xf32>) -> tensor<8192x2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    return %8 : tensor<8192x2048xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x98xf32>, %arg1: tensor<98x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<2048x2048xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<2048x98xf32>, tensor<98x2048xf32>) -> tensor<2048x2048xf32>
    %1 = tensor.empty() : tensor<2048x2048xf32>
    %2 = "ttir.all_reduce"(%0, %1) <{channel_handle = 1 : si32, dim = 1 : si32, reduce_type = #tt.reduce_type<sum>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %3 = tensor.empty() : tensor<1x2048xf32>
    %4 = "ttir.reshape"(%arg2, %3) <{shape = [1 : i32, 2048 : i32]}> : (tensor<2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %5 = tensor.empty() : tensor<1x2048xf32>
    %6 = "ttir.broadcast"(%4, %5) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %7 = tensor.empty() : tensor<2048x2048xf32>
    %8 = "ttir.broadcast"(%6, %7) <{broadcast_dimensions = array<i32: 2048, 1>}> : (tensor<1x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    %9 = tensor.empty() : tensor<2048x2048xf32>
    %10 = "ttir.add"(%2, %8, %9) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<2048x2048xf32>, tensor<2048x2048xf32>, tensor<2048x2048xf32>) -> tensor<2048x2048xf32>
    return %10 : tensor<2048x2048xf32>
  }
}
