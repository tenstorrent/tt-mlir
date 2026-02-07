// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_data_parallel_n300 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,1,1,1]<=[2]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<32x1x1024x2048xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 1, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    %4 = call @shmap_body(%1, %3) : (tensor<32x1x1024x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<32x1x1024x512xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x1x1024x512xf32>) -> tensor<32x1x1024x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[2,1,1,1]<=[2]}"} : (tensor<32x1x1024x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<32x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<32x1x1024x512xf32> {jax.result_info = "[('batch',), None, None, None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<32x1x1024x2048xf32>) -> tensor<32x1024x2048xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x2048x512xf32>) -> tensor<1x2048x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1024x2048xf32>, tensor<1x2048x512xf32>) -> tensor<32x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<32x1024x1x512xf32>) -> tensor<32x1x1024x512xf32>
    return %3 : tensor<32x1x1024x512xf32>
  }
}

// -----

module @jit_data_parallel_t3000 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<8x1x1024x2048xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    %4 = call @shmap_body(%1, %3) : (tensor<8x1x1024x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<8x1x1024x512xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8x1x1024x512xf32>) -> tensor<8x1x1024x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,1]<=[8]}"} : (tensor<8x1x1024x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<8x1x1024x512xf32> {jax.result_info = "[('batch',), None, None, None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<8x1x1024x2048xf32>) -> tensor<8x1024x2048xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x2048x512xf32>) -> tensor<1x2048x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x1024x2048xf32>, tensor<1x2048x512xf32>) -> tensor<8x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<8x1024x1x512xf32>) -> tensor<8x1x1024x512xf32>
    return %3 : tensor<8x1x1024x512xf32>
  }
}

// -----

module @jit_data_parallel_tg attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[32,1,1,1]<=[32]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<2x1x1024x2048xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 32, 1, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    %4 = call @shmap_body(%1, %3) : (tensor<2x1x1024x2048xf32>, tensor<1x1x2048x512xf32>) -> tensor<2x1x1024x512xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2x1x1024x512xf32>) -> tensor<2x1x1024x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[32,1,1,1]<=[32]}"} : (tensor<2x1x1024x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 32, 1, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<2x1x1024x512xf32> {jax.result_info = "[('batch',), None, None, None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<2x1x1024x2048xf32>) -> tensor<2x1024x2048xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x2048x512xf32>) -> tensor<1x2048x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<2x1024x2048xf32>, tensor<1x2048x512xf32>) -> tensor<2x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<2x1024x1x512xf32>) -> tensor<2x1x1024x512xf32>
    return %3 : tensor<2x1x1024x512xf32>
  }
}

// -----

module @jit_tensor_parallel_n300 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<64x1x1024x1024xf32>, tensor<1x1x1024x512xf32>) -> tensor<64x1x512x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x512x512xf32>) -> tensor<64x1x512x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<64x1x512x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<64x1x1024x1024xf32>, %arg1: tensor<1x1x1024x512xf32>) -> (tensor<64x1x512x512xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<64x1x1024x1024xf32>) -> tensor<64x1024x1024xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x1024x512xf32>) -> tensor<1x1024x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1024x1024xf32>, tensor<1x1024x512xf32>) -> tensor<64x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x1024x1x512xf32>) -> tensor<64x1x1024x512xf32>
    %4 = "stablehlo.reduce_scatter"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 2 : i64, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<64x1x1024x512xf32>) -> tensor<64x1x512x512xf32>
    return %4 : tensor<64x1x512x512xf32>
  }
}

// -----

module @jit_tensor_parallel_t3000 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x256x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<64x1x1024x256xf32>, tensor<1x1x256x512xf32>) -> tensor<64x1x128x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x128x512xf32>) -> tensor<64x1x128x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<64x1x128x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<64x1x1024x256xf32>, %arg1: tensor<1x1x256x512xf32>) -> (tensor<64x1x128x512xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<64x1x1024x256xf32>) -> tensor<64x1024x256xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x256x512xf32>) -> tensor<1x256x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1024x256xf32>, tensor<1x256x512xf32>) -> tensor<64x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x1024x1x512xf32>) -> tensor<64x1x1024x512xf32>
    %4 = "stablehlo.reduce_scatter"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 2 : i64, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<64x1x1024x512xf32>) -> tensor<64x1x128x512xf32>
    return %4 : tensor<64x1x128x512xf32>
  }
}

// -----

module @jit_tensor_parallel_tg attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x64x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<64x1x1024x64xf32>, tensor<1x1x64x512xf32>) -> tensor<64x1x32x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x32x512xf32>) -> tensor<64x1x32x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<64x1x32x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<64x1x1024x64xf32>, %arg1: tensor<1x1x64x512xf32>) -> (tensor<64x1x32x512xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<64x1x1024x64xf32>) -> tensor<64x1024x64xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x64x512xf32>) -> tensor<1x64x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1024x64xf32>, tensor<1x64x512xf32>) -> tensor<64x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<64x1024x1x512xf32>) -> tensor<64x1x1024x512xf32>
    %4 = "stablehlo.reduce_scatter"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, scatter_dimension = 2 : i64, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<64x1x1024x512xf32>) -> tensor<64x1x32x512xf32>
    return %4 : tensor<64x1x32x512xf32>
  }
}

// -----

module @jit_data_tensor_parallel_t3000 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,1,1,4]<=[8]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<32x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 1, 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x512x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<32x1x1024x512xf32>, tensor<1x1x512x512xf32>) -> tensor<32x1x256x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x1x256x512xf32>) -> tensor<32x1x256x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[2,1,4,1]<=[8]}"} : (tensor<32x1x256x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<32x1x1024x512xf32>, %arg1: tensor<1x1x512x512xf32>) -> (tensor<32x1x256x512xf32> {jax.result_info = "[('batch',), None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<32x1x1024x512xf32>) -> tensor<32x1024x512xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x512x512xf32>) -> tensor<1x512x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x1024x512xf32>, tensor<1x512x512xf32>) -> tensor<32x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<32x1024x1x512xf32>) -> tensor<32x1x1024x512xf32>
    %4 = "stablehlo.reduce_scatter"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 2 : i64, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x1x1024x512xf32>) -> tensor<32x1x256x512xf32>
    return %4 : tensor<32x1x256x512xf32>
  }
}

// -----

module @jit_data_tensor_parallel_tg attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x1x1024x2048xf32>, %arg1: tensor<1x1x2048x512xf32>) -> (tensor<64x1x1024x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1,1,4]<=[32]}"} : (tensor<64x1x1024x2048xf32>) -> tensor<64x1x1024x2048xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<64x1x1024x2048xf32>) -> tensor<8x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1, 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x2048x512xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x512xf32>) -> tensor<1x1x512x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8x1x1024x512xf32>, tensor<1x1x512x512xf32>) -> tensor<8x1x256x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8x1x256x512xf32>) -> tensor<8x1x256x512xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[8,1,4,1]<=[32]}"} : (tensor<8x1x256x512xf32>) -> tensor<64x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<64x1x1024x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8x1x1024x512xf32>, %arg1: tensor<1x1x512x512xf32>) -> (tensor<8x1x256x512xf32> {jax.result_info = "[('batch',), None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<8x1x1024x512xf32>) -> tensor<8x1024x512xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x1x512x512xf32>) -> tensor<1x512x512xf32>
    %2 = stablehlo.dot_general %0, %1, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<8x1024x512xf32>, tensor<1x512x512xf32>) -> tensor<8x1024x1x512xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1, 3] : (tensor<8x1024x1x512xf32>) -> tensor<8x1x1024x512xf32>
    %4 = "stablehlo.reduce_scatter"(%3) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 2 : i64, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<8x1x1024x512xf32>) -> tensor<8x1x256x512xf32>
    return %4 : tensor<8x1x256x512xf32>
  }
}
