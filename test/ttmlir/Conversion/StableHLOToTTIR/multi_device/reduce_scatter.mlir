// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
// UNSUPPORTED: true

module @jit_fwd_one attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<2048x392xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<392x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = call @shmap_body(%1, %3, %5) : (tensor<2048x392xf32>, tensor<392x2048xf32>, tensor<1024xf32>) -> tensor<2048x1024xf32>
    %7 = stablehlo.custom_call @Sharding(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[8]}"} : (tensor<2048x1024xf32>) -> tensor<8192x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %8 : tensor<8192x2048xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x392xf32>, %arg1: tensor<392x2048xf32>, %arg2: tensor<1024xf32>) -> (tensor<2048x1024xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x392xf32>, tensor<392x2048xf32>) -> tensor<2048x2048xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2048x2048xf32>) -> tensor<2048x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<2048x1024xf32>
    %4 = stablehlo.add %1, %3 : tensor<2048x1024xf32>
    return %4 : tensor<2048x1024xf32>
  }
}

module @jit_fwd_two attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<4096x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<196x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<512xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = call @shmap_body(%1, %3, %5) : (tensor<4096x196xf32>, tensor<196x2048xf32>, tensor<512xf32>) -> tensor<4096x512xf32>
    %7 = stablehlo.custom_call @Sharding(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4096x512xf32>) -> tensor<4096x512xf32>
    %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<4096x512xf32>) -> tensor<8192x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %8 : tensor<8192x2048xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4096x196xf32>, %arg1: tensor<196x2048xf32>, %arg2: tensor<512xf32>) -> (tensor<4096x512xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x2048xf32>) -> tensor<4096x2048xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<4096x2048xf32>) -> tensor<4096x512xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<4096x512xf32>
    %4 = stablehlo.add %1, %3 : tensor<4096x512xf32>
    return %4 : tensor<4096x512xf32>
  }
}

module @jit_fwd_three attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<196x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<512xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = call @shmap_body(%1, %3, %5) : (tensor<1024x196xf32>, tensor<196x2048xf32>, tensor<512xf32>) -> tensor<1024x512xf32>
    %7 = stablehlo.custom_call @Sharding(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x512xf32>) -> tensor<1024x512xf32>
    %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<1024x512xf32>) -> tensor<8192x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %8 : tensor<8192x2048xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x196xf32>, %arg1: tensor<196x2048xf32>, %arg2: tensor<512xf32>) -> (tensor<1024x512xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x2048xf32>) -> tensor<1024x2048xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<1024x2048xf32>) -> tensor<1024x512xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<1024x512xf32>
    %4 = stablehlo.add %1, %3 : tensor<1024x512xf32>
    return %4 : tensor<1024x512xf32>
  }
}

module @jit_fwd_four attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<2048x98xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8,1,4]<=[4,8]T(1,0) last_tile_dim_replicate}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<98x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[4,8]T(1,0) last_tile_dim_replicate}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<256xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = call @shmap_body(%1, %3, %5) : (tensor<2048x98xf32>, tensor<98x2048xf32>, tensor<256xf32>) -> tensor<2048x256xf32>
    %7 = stablehlo.custom_call @Sharding(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x256xf32>) -> tensor<2048x256xf32>
    %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[32]}"} : (tensor<2048x256xf32>) -> tensor<8192x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %8 : tensor<8192x2048xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x98xf32>, %arg1: tensor<98x2048xf32>, %arg2: tensor<256xf32>) -> (tensor<2048x256xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x98xf32>, tensor<98x2048xf32>) -> tensor<2048x2048xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<2048x2048xf32>) -> tensor<2048x256xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<2048x256xf32>
    %4 = stablehlo.add %1, %3 : tensor<2048x256xf32>
    return %4 : tensor<2048x256xf32>
  }
}
