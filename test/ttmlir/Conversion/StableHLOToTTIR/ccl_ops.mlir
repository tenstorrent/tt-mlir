// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

// jax/pjrt sharding target 1x2 for n300
module @jit_matmul_basic attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x392xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<392x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = call @shmap_body(%1, %3) : (tensor<8192x392xf32>, tensor<392x16384xf32>) -> tensor<8192x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x392xf32>, %arg1: tensor<392x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x392xf32>, tensor<392x16384xf32>) -> tensor<8192x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.all_reduce"[[C:.*]]
    return %1 : tensor<8192x16384xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k
module @jit_matmul_basic2 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<4096x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<196x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = call @shmap_body(%1, %3) : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<4096x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4096x196xf32>, %arg1: tensor<196x16384xf32>) -> (tensor<4096x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.all_reduce"[[C:.*]]
    return %1 : tensor<4096x16384xf32>
  }
}

// jax/pjrt sharding target 1x8 for t3k
module @jit_matmul_basic3 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x98xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<98x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = call @shmap_body(%1, %3) : (tensor<8192x98xf32>, tensor<98x16384xf32>) -> tensor<8192x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x98xf32>, %arg1: tensor<98x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x98xf32>, tensor<98x16384xf32>) -> tensor<8192x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.all_reduce"[[C:.*]]
    return %1 : tensor<8192x16384xf32>
  }
}

// jax/pjrt sharding target 8x4 for tg
module @jit_matmul_basic4 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<196x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %4 = call @shmap_body(%1, %3) : (tensor<1024x196xf32>, tensor<196x16384xf32>) -> tensor<1024x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x16384xf32>) -> tensor<1024x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<1024x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x196xf32>, %arg1: tensor<196x16384xf32>) -> (tensor<1024x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x16384xf32>) -> tensor<1024x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<1024x16384xf32>) -> tensor<1024x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.all_reduce"[[C:.*]]
    return %1 : tensor<1024x16384xf32>
  }
}
