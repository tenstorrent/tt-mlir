// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
// UNSUPPORTED: true

module @jit_fwd_one attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<2048x392xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<392x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x1024xf32>) -> tensor<1024x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[4,2]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf32>) -> tensor<1024xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024xf32>) -> tensor<512xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %10 = call @shmap_body(%1, %3, %5, %7, %9) : (tensor<2048x392xf32>, tensor<392x2048xf32>, tensor<1024xf32>, tensor<1024x1024xf32>, tensor<512xf32>) -> tensor<2048x512xf32>
    %11 = stablehlo.custom_call @Sharding(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x512xf32>) -> tensor<2048x512xf32>
    %12 = stablehlo.custom_call @SPMDShardToFullShape(%11) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[8]}"} : (tensor<2048x512xf32>) -> tensor<8192x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %12 : tensor<8192x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x392xf32>, %arg1: tensor<392x2048xf32>, %arg2: tensor<1024xf32>, %arg3: tensor<1024x1024xf32>, %arg4: tensor<512xf32>) -> (tensor<2048x512xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x392xf32>, tensor<392x2048xf32>) -> tensor<2048x2048xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %10 = stablehlo.add %arg5, %arg6 : tensor<f32>
      stablehlo.return %10 : tensor<f32>
    }) : (tensor<2048x2048xf32>) -> tensor<2048x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<2048x1024xf32>
    %4 = stablehlo.add %1, %3 : tensor<2048x1024xf32>
    %5 = stablehlo.dot_general %4, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<2048x1024xf32>, tensor<1024x1024xf32>) -> tensor<2048x1024xf32>
    %6 = "stablehlo.reduce_scatter"(%5) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %10 = stablehlo.add %arg5, %arg6 : tensor<f32>
      stablehlo.return %10 : tensor<f32>
    }) : (tensor<2048x1024xf32>) -> tensor<2048x512xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %7 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<2048x512xf32>
    %9 = stablehlo.add %6, %8 : tensor<2048x512xf32>
    return %9 : tensor<2048x512xf32>
  }
}

module @jit_fwd_two attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x784xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1024xf32>) -> tensor<1024xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024xf32>) -> tensor<1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %10 = call @shmap_body(%1, %3, %5, %7, %9) : (tensor<1024x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<1024x1024xf32>
    %11 = stablehlo.custom_call @Sharding(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %12 = stablehlo.custom_call @SPMDShardToFullShape(%11) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<1024x1024xf32>) -> tensor<8192x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %12 : tensor<8192x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<1024x1024xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x784xf32>, tensor<784x2048xf32>) -> tensor<1024x2048xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<1024x2048xf32>
    %3 = stablehlo.add %0, %2 : tensor<1024x2048xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x2048xf32>, tensor<2048x1024xf32>) -> tensor<1024x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<1024x1024xf32>
    %7 = stablehlo.add %4, %6 : tensor<1024x1024xf32>
    return %7 : tensor<1024x1024xf32>
  }
}

module @jit_fwd_three attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<196x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<512xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x1024xf32>) -> tensor<512x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1024xf32>) -> tensor<1024xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024xf32>) -> tensor<256xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %10 = call @shmap_body(%1, %3, %5, %7, %9) : (tensor<1024x196xf32>, tensor<196x2048xf32>, tensor<512xf32>, tensor<512x1024xf32>, tensor<256xf32>) -> tensor<1024x256xf32>
    %11 = stablehlo.custom_call @Sharding(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x256xf32>) -> tensor<1024x256xf32>
    %12 = stablehlo.custom_call @SPMDShardToFullShape(%11) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<1024x256xf32>) -> tensor<8192x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %12 : tensor<8192x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x196xf32>, %arg1: tensor<196x2048xf32>, %arg2: tensor<512xf32>, %arg3: tensor<512x1024xf32>, %arg4: tensor<256xf32>) -> (tensor<1024x256xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x2048xf32>) -> tensor<1024x2048xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %10 = stablehlo.add %arg5, %arg6 : tensor<f32>
      stablehlo.return %10 : tensor<f32>
    }) : (tensor<1024x2048xf32>) -> tensor<1024x512xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<512xf32>) -> tensor<1x512xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x512xf32>) -> tensor<1024x512xf32>
    %4 = stablehlo.add %1, %3 : tensor<1024x512xf32>
    %5 = stablehlo.dot_general %4, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<512x1024xf32>) -> tensor<1024x1024xf32>
    %6 = "stablehlo.reduce_scatter"(%5) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg5: tensor<f32>, %arg6: tensor<f32>):
      %10 = stablehlo.add %arg5, %arg6 : tensor<f32>
      stablehlo.return %10 : tensor<f32>
    }) : (tensor<1024x1024xf32>) -> tensor<1024x256xf32>
    // CHECK: %[[C:.*]] = "ttir.reduce_scatter"[[C:.*]]
    %7 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<256xf32>) -> tensor<1x256xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0, 1] : (tensor<1x256xf32>) -> tensor<1024x256xf32>
    %9 = stablehlo.add %6, %8 : tensor<1024x256xf32>
    return %9 : tensor<1024x256xf32>
  }
}

module @jit_fwd_four attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<8192x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<256x784xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x2048xf32>) -> tensor<784x2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048xf32>) -> tensor<2048xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x1024xf32>) -> tensor<2048x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1024xf32>) -> tensor<1024xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024xf32>) -> tensor<1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %10 = call @shmap_body(%1, %3, %5, %7, %9) : (tensor<256x784xf32>, tensor<784x2048xf32>, tensor<2048xf32>, tensor<2048x1024xf32>, tensor<1024xf32>) -> tensor<256x1024xf32>
    %11 = stablehlo.custom_call @Sharding(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<256x1024xf32>) -> tensor<256x1024xf32>
    %12 = stablehlo.custom_call @SPMDShardToFullShape(%11) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<256x1024xf32>) -> tensor<8192x1024xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %12 : tensor<8192x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<256x784xf32>, %arg1: tensor<784x2048xf32>, %arg2: tensor<2048xf32>, %arg3: tensor<2048x1024xf32>, %arg4: tensor<1024xf32>) -> (tensor<256x1024xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x784xf32>, tensor<784x2048xf32>) -> tensor<256x2048xf32>
    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xf32>) -> tensor<1x2048xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x2048xf32>) -> tensor<256x2048xf32>
    %3 = stablehlo.add %0, %2 : tensor<256x2048xf32>
    %4 = stablehlo.dot_general %3, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<2048x1024xf32>) -> tensor<256x1024xf32>
    %5 = stablehlo.broadcast_in_dim %arg4, dims = [1] : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %6 = stablehlo.broadcast_in_dim %5, dims = [0, 1] : (tensor<1x1024xf32>) -> tensor<256x1024xf32>
    %7 = stablehlo.add %4, %6 : tensor<256x1024xf32>
    return %7 : tensor<256x1024xf32>
  }
}
