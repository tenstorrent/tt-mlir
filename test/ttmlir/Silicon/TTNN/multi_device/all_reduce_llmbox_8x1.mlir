// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=8,1" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// REQUIRES: multi-chip-x8

module @jit_fwd_two attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<1024x784xf32>
    %1 = "ttir.mesh_shard"(%arg0, %0) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<8x1>, shard_type = #tt.shard_type<devices>}> : (tensor<8192x784xf32>, tensor<1024x784xf32>) -> tensor<1024x784xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %2 = tensor.empty() : tensor<784x2048xf32>
    %3 = "ttir.mesh_shard"(%arg1, %2) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<784x2048xf32>, tensor<784x2048xf32>) -> tensor<784x2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %4 = tensor.empty() : tensor<2048xf32>
    %5 = "ttir.mesh_shard"(%arg2, %4) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<2048xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %6 = tensor.empty() : tensor<2048x1024xf32>
    %7 = "ttir.mesh_shard"(%arg3, %6) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<2048x1024xf32>, tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %8 = tensor.empty() : tensor<1024xf32>
    %9 = "ttir.mesh_shard"(%arg4, %8) <{shard_direction = #tt.shard_direction<full_to_shard>, shard_shape = #tt.grid<1>, shard_type = #tt.shard_type<replicate>}> : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    %10 = call @shmap_body(%1, %3, %5, %7, %9) : (tensor<1024x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<1024x1024xf32>
    %11 = tensor.empty() : tensor<8192x1024xf32>
    %12 = "ttir.mesh_shard"(%10, %11) <{shard_direction = #tt.shard_direction<shard_to_full>, shard_shape = #tt.grid<8x1>, shard_type = #tt.shard_type<devices>}> : (tensor<1024x1024xf32>, tensor<8192x1024xf32>) -> tensor<8192x1024xf32>
    // CHECK: %[[C:.*]] = "ttnn.mesh_shard"[[C:.*]]
    return %12 : tensor<8192x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<1024x1024xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x784xf32>, tensor<784x2048xf32>) -> tensor<1024x2048xf32>
    %1 = tensor.empty() : tensor<1x2048xf32>
    %2 = "ttir.reshape"(%arg2, %1) <{shape = [1 : i32, 2048 : i32]}> : (tensor<2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %3 = tensor.empty() : tensor<1x2048xf32>
    %4 = "ttir.broadcast"(%2, %3) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x2048xf32>, tensor<1x2048xf32>) -> tensor<1x2048xf32>
    %5 = tensor.empty() : tensor<1024x2048xf32>
    %6 = "ttir.broadcast"(%4, %5) <{broadcast_dimensions = array<i32: 1024, 1>}> : (tensor<1x2048xf32>, tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
    %7 = tensor.empty() : tensor<1024x2048xf32>
    %8 = "ttir.add"(%0, %6, %7) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1024x2048xf32>, tensor<1024x2048xf32>, tensor<1024x2048xf32>) -> tensor<1024x2048xf32>
    %9 = "ttir.dot_general"(%8, %arg3) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1024x2048xf32>, tensor<2048x1024xf32>) -> tensor<1024x1024xf32>
    %10 = tensor.empty() : tensor<1x1024xf32>
    %11 = "ttir.reshape"(%arg4, %10) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %12 = tensor.empty() : tensor<1x1024xf32>
    %13 = "ttir.broadcast"(%11, %12) <{broadcast_dimensions = array<i32: 1, 1>}> : (tensor<1x1024xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %14 = tensor.empty() : tensor<1024x1024xf32>
    %15 = "ttir.broadcast"(%13, %14) <{broadcast_dimensions = array<i32: 1024, 1>}> : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %16 = tensor.empty() : tensor<1024x1024xf32>
    %17 = "ttir.add"(%9, %15, %16) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    return %17 : tensor<1024x1024xf32>
  }
}
