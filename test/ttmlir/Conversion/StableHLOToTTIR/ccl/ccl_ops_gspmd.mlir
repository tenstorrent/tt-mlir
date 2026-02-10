// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// jax/pjrt sharding target 1x2 for n300 all_reduce cluster_axis=1 rank=2
module @all_reduce_1x2_rank_2_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x392xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<392x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8192x392xf32>, tensor<392x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x392xf32>, %arg1: tensor<392x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x392xf32>, tensor<392x16384xf32>) -> tensor<8192x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    return %1 : tensor<8192x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_reduce cluster_axis=1 rank=2
module @all_reduce_2x4_rank_2_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<4096x196xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<196x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<4096x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4096x196xf32>, %arg1: tensor<196x16384xf32>) -> (tensor<4096x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    return %1 : tensor<4096x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_reduce cluster_axis=1 rank=2
module @all_reduce_1x8_rank_2_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x98xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<98x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8192x98xf32>, tensor<98x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x98xf32>, %arg1: tensor<98x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = "[('x',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x98xf32>, tensor<98x16384xf32>) -> tensor<8192x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    return %1 : tensor<8192x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_reduce cluster_axis=1 rank=2
module @all_reduce_8x4_rank_2_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x196xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<196x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1024x196xf32>, tensor<196x16384xf32>) -> tensor<1024x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x16384xf32>) -> tensor<1024x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<1024x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x196xf32>, %arg1: tensor<196x16384xf32>) -> (tensor<1024x16384xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x196xf32>, tensor<196x16384xf32>) -> tensor<1024x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<1024x16384xf32>) -> tensor<1024x16384xf32>
    return %1 : tensor<1024x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_reduce cluster_axis=1 rank=2
module @all_reduce_1x32_rank_2_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>, %arg1: tensor<800x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,32]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x25xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<800x16384xf32>) -> tensor<800x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<800x16384xf32>) -> tensor<25x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8192x25xf32>, tensor<25x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x25xf32>, %arg1: tensor<25x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = "[None, None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x25xf32>, tensor<25x16384xf32>) -> tensor<8192x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    return %1 : tensor<8192x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 all_reduce cluster_axis=0 rank=4
module @all_reduce_1x2_rank_4_cluster_0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x4096x512xf32>, tensor<1x1x512x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x4096x16384xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x4096x512xf32>) -> tensor<4096x512xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4096x512xf32>, tensor<1x1x512x16384xf32>) -> tensor<4096x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<4096x1x1x16384xf32>) -> tensor<1x1x4096x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    return %3 : tensor<1x1x4096x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 all_reduce cluster_axis=1 rank=4
module @all_reduce_1x2_rank_4_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x256x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x8192x256xf32>, tensor<1x1x256x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>, %arg1: tensor<1x1x256x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x8192x256xf32>) -> tensor<8192x256xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8192x256xf32>, tensor<1x1x256x16384xf32>) -> tensor<8192x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<8192x1x1x16384xf32>) -> tensor<1x1x8192x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    return %3 : tensor<1x1x8192x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_reduce cluster_axis=0 rank=4
module @all_reduce_1x8_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x1024x512xf32>, tensor<1x1x512x16384xf32>) -> tensor<1x1x1024x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x16384xf32>) -> tensor<1x1x1024x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x1024x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x1024x16384xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x1024x512xf32>) -> tensor<1024x512xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1024x512xf32>, tensor<1x1x512x16384xf32>) -> tensor<1024x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<1024x1x1x16384xf32>) -> tensor<1x1x1024x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7]]> : tensor<8x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x1024x16384xf32>) -> tensor<1x1x1024x16384xf32>
    return %3 : tensor<1x1x1024x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_reduce cluster_axis=1 rank=4
module @all_reduce_1x8_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x64x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x8192x64xf32>, tensor<1x1x64x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x64xf32>, %arg1: tensor<1x1x64x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x8192x64xf32>) -> tensor<8192x64xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8192x64xf32>, tensor<1x1x64x16384xf32>) -> tensor<8192x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<8192x1x1x16384xf32>) -> tensor<1x1x8192x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    return %3 : tensor<1x1x8192x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_reduce cluster_axis=0 rank=4
module @all_reduce_2x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x256x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x2048x256xf32>, tensor<1x1x256x16384xf32>) -> tensor<1x1x2048x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x16384xf32>) -> tensor<1x1x2048x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1x2048x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x256xf32>, %arg1: tensor<1x1x256x16384xf32>) -> (tensor<1x1x2048x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x2048x256xf32>) -> tensor<2048x256xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2048x256xf32>, tensor<1x1x256x16384xf32>) -> tensor<2048x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<2048x1x1x16384xf32>) -> tensor<1x1x2048x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x2048x16384xf32>) -> tensor<1x1x2048x16384xf32>
    return %3 : tensor<1x1x2048x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_reduce cluster_axis=1 rank=4
module @all_reduce_2x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x128x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x4096x128xf32>, tensor<1x1x128x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>, %arg1: tensor<1x1x128x16384xf32>) -> (tensor<1x1x4096x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x4096x128xf32>) -> tensor<4096x128xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<4096x128xf32>, tensor<1x1x128x16384xf32>) -> tensor<4096x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<4096x1x1x16384xf32>) -> tensor<1x1x4096x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    return %3 : tensor<1x1x4096x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_reduce cluster_axis=0 rank=4
module @all_reduce_1x32_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x256x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x256x512xf32>, tensor<1x1x512x16384xf32>) -> tensor<1x1x256x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x256x16384xf32>) -> tensor<1x1x256x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x256x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x256x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x256x16384xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x256x512xf32>) -> tensor<256x512xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<256x512xf32>, tensor<1x1x512x16384xf32>) -> tensor<256x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<256x1x1x16384xf32>) -> tensor<1x1x256x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]> : tensor<32x1xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x256x16384xf32>) -> tensor<1x1x256x16384xf32>
    return %3 : tensor<1x1x256x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_reduce cluster_axis=1 rank=4
module @all_reduce_1x32_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x16x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x8192x16xf32>, tensor<1x1x16x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x16xf32>, %arg1: tensor<1x1x16x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x8192x16xf32>) -> tensor<8192x16xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<8192x16xf32>, tensor<1x1x16x16384xf32>) -> tensor<8192x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<8192x1x1x16384xf32>) -> tensor<1x1x8192x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x8192x16384xf32>) -> tensor<1x1x8192x16384xf32>
    return %3 : tensor<1x1x8192x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_reduce cluster_axis=0 rank=4
module @all_reduce_8x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x64x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x2048x64xf32>, tensor<1x1x64x16384xf32>) -> tensor<1x1x2048x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x16384xf32>) -> tensor<1x1x2048x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1x2048x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x64xf32>, %arg1: tensor<1x1x64x16384xf32>) -> (tensor<1x1x2048x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x2048x64xf32>) -> tensor<2048x64xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<2048x64xf32>, tensor<1x1x64x16384xf32>) -> tensor<2048x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<2048x1x1x16384xf32>) -> tensor<1x1x2048x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x2048x16384xf32>) -> tensor<1x1x2048x16384xf32>
    return %3 : tensor<1x1x2048x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_reduce cluster_axis=1 rank=4
module @all_reduce_8x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>, %arg1: tensor<1x1x512x16384xf32>) -> (tensor<1x1x8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x512x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x512x16384xf32>) -> tensor<1x1x128x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1x1x1024x128xf32>, tensor<1x1x128x16384xf32>) -> tensor<1x1x1024x16384xf32>
    // CHECK: "ttir.all_reduce"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x16384xf32>) -> tensor<1x1x1024x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<1x1x1024x16384xf32>) -> tensor<1x1x8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %6 : tensor<1x1x8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>, %arg1: tensor<1x1x128x16384xf32>) -> (tensor<1x1x1024x16384xf32> {jax.result_info = "[None, None, ('batch',), None]"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x1x1024x128xf32>) -> tensor<1024x128xf32>
    %1 = stablehlo.dot_general %0, %arg1, contracting_dims = [1] x [2], precision = [DEFAULT, DEFAULT] : (tensor<1024x128xf32>, tensor<1x1x128x16384xf32>) -> tensor<1024x1x1x16384xf32>
    %2 = stablehlo.transpose %1, dims = [1, 2, 0, 3] : (tensor<1024x1x1x16384xf32>) -> tensor<1x1x1024x16384xf32>
    %3 = "stablehlo.all_reduce"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %4 : tensor<f32>
    }) : (tensor<1x1x1024x16384xf32>) -> tensor<1x1x1024x16384xf32>
    return %3 : tensor<1x1x1024x16384xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 all_gather cluster_axis=0 rank=2
module @all_gather_1x2_rank_2_cluster_0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x400xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<8192x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x400xf32>) -> (tensor<8192x400xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>, use_global_device_ids}> : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    return %0 : tensor<8192x400xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 all_gather cluster_axis=1 rank=2
module @all_gather_1x2_rank_2_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<16384x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<4096x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<4096x800xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<16384x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<16384x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4096x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<4096x800xf32>) -> tensor<8192x800xf32>
    return %0 : tensor<8192x800xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 all_gather cluster_axis=0 rank=4
module @all_gather_1x2_rank_4_cluster_0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>, use_global_device_ids}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %0 : tensor<1x1x8192x256xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 all_gather cluster_axis=1 rank=4
module @all_gather_1x2_rank_4_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x784xf32>) -> (tensor<1x1x16384x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x4096x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x784xf32>) -> tensor<1x1x8192x784xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x16384x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x16384x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x784xf32>) -> (tensor<1x1x8192x784xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<1x1x4096x784xf32>) -> tensor<1x1x8192x784xf32>
    return %0 : tensor<1x1x8192x784xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_gather cluster_axis=0 rank=2
module @all_gather_1x8_rank_2_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x100xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<8192x100xf32>) -> tensor<8192x100xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x100xf32>) -> tensor<8192x100xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x100xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<8192x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x100xf32>) -> (tensor<8192x100xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7]]> : tensor<8x1xi64>, use_global_device_ids}> : (tensor<8192x100xf32>) -> tensor<8192x100xf32>
    return %0 : tensor<8192x100xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_gather cluster_axis=1 rank=2
module @all_gather_1x8_rank_2_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<65536x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<1024x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1024x800xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x800xf32>) -> tensor<65536x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<65536x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<1024x800xf32>) -> tensor<8192x800xf32>
    return %0 : tensor<8192x800xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_gather cluster_axis=0 rank=4
module @all_gather_1x8_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x64xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7]]> : tensor<8x1xi64>, use_global_device_ids}> : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    return %0 : tensor<1x1x8192x64xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k all_gather cluster_axis=1 rank=4
module @all_gather_1x8_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x784xf32>) -> (tensor<1x1x65536x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x1024x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x784xf32>) -> tensor<1x1x8192x784xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x65536x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x65536x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x784xf32>) -> (tensor<1x1x8192x784xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<1x1x1024x784xf32>) -> tensor<1x1x8192x784xf32>
    return %0 : tensor<1x1x8192x784xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_gather cluster_axis=0 rank=2
module @all_gather_2x4_rank_2_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x1600xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<2048x400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2048x400xf32>) -> tensor<2048x800xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x800xf32>) -> tensor<2048x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<2048x800xf32>) -> tensor<8192x1600xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<8192x1600xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x400xf32>) -> (tensor<2048x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<2048x400xf32>) -> tensor<2048x800xf32>
    return %0 : tensor<2048x800xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_gather cluster_axis=1 rank=2
module @all_gather_2x4_rank_2_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<32768x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<2048x400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2048x400xf32>) -> tensor<8192x400xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<8192x400xf32>) -> tensor<32768x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<32768x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x400xf32>) -> (tensor<8192x400xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<2048x400xf32>) -> tensor<8192x400xf32>
    return %0 : tensor<8192x400xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_gather cluster_axis=0 rank=4
module @all_gather_2x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x16384x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x128xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x128xf32>) -> tensor<1x1x16384x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x16384x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x128xf32>
    return %0 : tensor<1x1x8192x128xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k all_gather cluster_axis=1 rank=4
module @all_gather_2x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x784xf32>) -> (tensor<1x1x32768x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x2048x392xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x392xf32>) -> tensor<1x1x8192x392xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x392xf32>) -> tensor<1x1x8192x392xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x8192x392xf32>) -> tensor<1x1x32768x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x32768x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x392xf32>) -> (tensor<1x1x8192x392xf32> {jax.result_info = "[None, None, ('model',), ('batch',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<1x1x2048x392xf32>) -> tensor<1x1x8192x392xf32>
    return %0 : tensor<1x1x8192x392xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_gather cluster_axis=0 rank=2
module @all_gather_1x32_rank_2_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,32]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x25xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<8192x25xf32>) -> tensor<8192x25xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x25xf32>) -> tensor<8192x25xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,32]<=[32]}"} : (tensor<8192x25xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<8192x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x25xf32>) -> (tensor<8192x25xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]> : tensor<32x1xi64>, use_global_device_ids}> : (tensor<8192x25xf32>) -> tensor<8192x25xf32>
    return %0 : tensor<8192x25xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_gather cluster_axis=1 rank=2
module @all_gather_1x32_rank_2_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<262144x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<256x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<256x800xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<262144x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<262144x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<256x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> : (tensor<256x800xf32>) -> tensor<8192x800xf32>
    return %0 : tensor<8192x800xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_gather cluster_axis=0 rank=4
module @all_gather_1x32_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x16xf32>) -> (tensor<1x1x8192x16xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]> : tensor<32x1xi64>, use_global_device_ids}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    return %0 : tensor<1x1x8192x16xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg all_gather cluster_axis=1 rank4
module @all_gather_1x32_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x784xf32>) -> (tensor<1x1x262144x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x256x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x256x784xf32>) -> tensor<1x1x8192x784xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x262144x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x262144x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x256x784xf32>) -> (tensor<1x1x8192x784xf32> {jax.result_info = "[None, None, ('model',), None]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> : (tensor<1x1x256x784xf32>) -> tensor<1x1x8192x784xf32>
    return %0 : tensor<1x1x8192x784xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_gather cluster_axis=0 rank=2
module @all_gather_8x4_rank_2_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x6400xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<2048x100xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2048x100xf32>) -> tensor<2048x800xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x800xf32>) -> tensor<2048x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"} : (tensor<2048x800xf32>) -> tensor<8192x6400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<8192x6400xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x100xf32>) -> (tensor<2048x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<2048x100xf32>) -> tensor<2048x800xf32>
    return %0 : tensor<2048x800xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_gather cluster_axis=1 rank=2
module @all_gather_4x8_rank_2_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<65536x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[4,8]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<1024x200xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1024x200xf32>) -> tensor<8192x200xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x200xf32>) -> tensor<8192x200xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[4,8]T(1,0)}"} : (tensor<8192x200xf32>) -> tensor<65536x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<65536x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x200xf32>) -> (tensor<8192x200xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<1024x200xf32>) -> tensor<8192x200xf32>
    return %0 : tensor<8192x200xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_gather cluster_axis=0 rank=4
module @all_gather_8x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x65536x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x8192x128xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x128xf32>) -> tensor<1x1x65536x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x65536x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<1x1x1024x128xf32>) -> tensor<1x1x8192x128xf32>
    return %0 : tensor<1x1x8192x128xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg all_gather cluster_axis=1 rank4
module @all_gather_8x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x784xf32>) -> (tensor<1x1x32768x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x784xf32>) -> tensor<1x1x2048x98xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x98xf32>) -> tensor<1x1x8192x98xf32>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x98xf32>) -> tensor<1x1x8192x98xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x98xf32>) -> tensor<1x1x32768x784xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x32768x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x98xf32>) -> (tensor<1x1x8192x98xf32> {jax.result_info = "[None, None, ('model',), ('batch',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 2 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> : (tensor<1x1x2048x98xf32>) -> tensor<1x1x8192x98xf32>
    return %0 : tensor<1x1x8192x98xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 reduce_scatter cluster_axis=0 rank4
module @reduce_scatter_1x2_rank_4_cluster_0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x4096x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x512xf32>) -> (tensor<1x1x4096x512xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    return %0 : tensor<1x1x4096x512xf32>
  }
}

// -----

// jax/pjrt sharding target 1x2 for n300 reduce_scatter cluster_axis=1 rank4
module @reduce_scatter_1x2_rank_4_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x128xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x128xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x256xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x128xf32>
    return %0 : tensor<1x1x8192x128xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k reduce_scatter cluster_axis=0 rank4
module @reduce_scatter_1x8_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x512xf32>) -> (tensor<1x1x1024x512xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7]]> : tensor<8x1xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    return %0 : tensor<1x1x1024x512xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k reduce_scatter cluster_axis=1 rank4
module @reduce_scatter_1x8_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x8xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x8xf32>) -> tensor<1x1x8192x8xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x8xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x64xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x64xf32>) -> (tensor<1x1x8192x8xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x8xf32>
    return %0 : tensor<1x1x8192x8xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k reduce_scatter cluster_axis=0 rank4
module @reduce_scatter_2x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x128xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x128xf32>) -> tensor<1x1x2048x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x2048x128xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x256xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x256xf32>) -> (tensor<1x1x2048x128xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x128xf32>
    return %0 : tensor<1x1x2048x128xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k reduce_scatter cluster_axis=1 rank4
module @reduce_scatter_2x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x32xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x32xf32>) -> tensor<1x1x4096x32xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x4096x32xf32>) -> tensor<1x1x8192x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x128xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>) -> (tensor<1x1x4096x32xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x32xf32>
    return %0 : tensor<1x1x4096x32xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg reduce_scatter cluster_axis=0 rank4
module @reduce_scatter_1x32_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x8192xf32>) -> (tensor<1x1x8192x8192xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x8192x8192xf32>) -> tensor<1x1x8192x8192xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x8192xf32>) -> tensor<1x1x256x8192xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x256x8192xf32>) -> tensor<1x1x256x8192xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x256x8192xf32>) -> tensor<1x1x256x8192xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x256x8192xf32>) -> tensor<1x1x8192x8192xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x8192xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x256x8192xf32>) -> (tensor<1x1x256x8192xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]> : tensor<32x1xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x256x8192xf32>) -> tensor<1x1x256x8192xf32>
    return %0 : tensor<1x1x256x8192xf32>
  }
}

// -----

// jax/pjrt sharding target 1x32 for tg reduce_scatter cluster_axis=1 rank4
module @reduce_scatter_1x32_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x8192xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x8192xf32>) -> tensor<1x1x8192x8192xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x8192xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x8xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x8xf32>) -> tensor<1x1x8192x8xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x8xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x256xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>) -> (tensor<1x1x8192x8xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x8xf32>
    return %0 : tensor<1x1x8192x8xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg reduce_scatter cluster_axis=0 rank4
module @reduce_scatter_8x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x8xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 0 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x8xf32>) -> tensor<1x1x2048x8xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x2048x8xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x64xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x64xf32>) -> (tensor<1x1x2048x8xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x8xf32>
    return %0 : tensor<1x1x2048x8xf32>
  }
}

// -----

// jax/pjrt sharding target 8x4 for tg reduce_scatter cluster_axis=1 rank4
module @reduce_scatter_8x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x128xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x32xf32>
    // CHECK: "ttir.reduce_scatter"
    // CHECK-SAME: cluster_axis = 1 : ui32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x32xf32>) -> tensor<1x1x1024x32xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x1024x32xf32>) -> tensor<1x1x8192x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x128xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>) -> (tensor<1x1x1024x32xf32> {jax.result_info = "[None, None, ('batch',), ('model',)]"}) {
    %0 = "stablehlo.reduce_scatter"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, scatter_dimension = 3 : i64, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %1 : tensor<f32>
    }) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x32xf32>
    return %0 : tensor<1x1x1024x32xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "x", None, "y"]
module @jit_neg_basic0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,4]<=[8]}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x512x128x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x512x128x256xf32>) -> tensor<1x512x128x256xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x512x128x256xf32>) -> tensor<1x512x128x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,4]<=[8]}"} : (tensor<1x512x128x256xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x512x128x256xf32>) -> (tensor<1x512x128x256xf32> {jax.result_info = "[None, ('x',), None, ('y',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x512x128x256xf32>
    return %0 : tensor<1x512x128x256xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "x", None, None]
module @jit_neg_basic1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x512x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x512x128x1024xf32>) -> tensor<1x512x128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x512x128x1024xf32>) -> tensor<1x512x128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x512x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x512x128x1024xf32>) -> (tensor<1x512x128x1024xf32> {jax.result_info = "[None, ('x',), None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x512x128x1024xf32>
    return %0 : tensor<1x512x128x1024xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, None, None, "y"]
module @jit_neg_basic2 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1024x128x256xf32>) -> tensor<1x1024x128x256xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x256xf32>) -> tensor<1x1024x128x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1024x128x256xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1024x128x256xf32>) -> (tensor<1x1024x128x256xf32> {jax.result_info = "[None, None, None, ('y',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x1024x128x256xf32>
    return %0 : tensor<1x1024x128x256xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "y", None, "x"]
module @jit_neg_basic3 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,2]<=[2,4]T(1,0)}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x256x128x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x256x128x512xf32>) -> tensor<1x256x128x512xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x256x128x512xf32>) -> tensor<1x256x128x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,2]<=[2,4]T(1,0)}"} : (tensor<1x256x128x512xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x256x128x512xf32>) -> (tensor<1x256x128x512xf32> {jax.result_info = "[None, ('y',), None, ('x',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x256x128x512xf32>
    return %0 : tensor<1x256x128x512xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "y", None, None]
module @jit_neg_basic4 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x256x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x256x128x1024xf32>) -> tensor<1x256x128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x256x128x1024xf32>) -> tensor<1x256x128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x256x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x256x128x1024xf32>) -> (tensor<1x256x128x1024xf32> {jax.result_info = "[None, ('y',), None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x256x128x1024xf32>
    return %0 : tensor<1x256x128x1024xf32>
  }
}

// -----

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, None, None, "x"]
module @jit_neg_basic5 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x1024x1024xf32>) -> (tensor<1x1x1024x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x1024x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x512xf32>) -> (tensor<1x1x1024x512xf32> {jax.result_info = "[None, None, None, ('x',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x1x1024x512xf32>
    return %0 : tensor<1x1x1024x512xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k - GSPMD negative, sharding [None, None, None, "y"]
module @jit_neg_basic6 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x128xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1024x128x128xf32>) -> (tensor<1x1024x128x128xf32> {jax.result_info = "[None, None, None, ('y',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x1024x128x128xf32>
    return %0 : tensor<1x1024x128x128xf32>
  }
}

// -----

// jax/pjrt sharding target 1x8 for t3k - GSPMD negative, sharding [None, "y", None, None]
module @jit_neg_basic7 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8,1,1]<=[8]}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x128x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x128x128x1024xf32>) -> tensor<1x128x128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x128x128x1024xf32>) -> tensor<1x128x128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,8,1,1]<=[8]}"} : (tensor<1x128x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 8, 1, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x128x128x1024xf32>) -> (tensor<1x128x128x1024xf32> {jax.result_info = "[None, ('y',), None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x128x128x1024xf32>
    return %0 : tensor<1x128x128x1024xf32>
  }
}

// -----

module @jit_collective_permute_1x2_rank_4_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [1, 0]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %0 : tensor<1x1x8192x256xf32>
  }
}

// -----

module @jit_collective_permute_1x2_rank_4_cluster_1_partial_target_pairs attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %0 : tensor<1x1x8192x256xf32>
  }
}

// -----

module @jit_collective_permute_1x2_rank_4_cluster_0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,1]<=[2]}"} : (tensor<1x1x4096x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x512xf32>) -> (tensor<1x1x4096x512xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<1x1x4096x512xf32>) -> tensor<1x1x4096x512xf32>
    return %0 : tensor<1x1x4096x512xf32>
  }
}

// -----

module @jit_collective_permute_1x32_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 0]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x16xf32>) -> (tensor<1x1x8192x16xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28], [28, 29], [29, 30], [30, 31], [31, 0]]> : tensor<32x2xi64>}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    return %0 : tensor<1x1x8192x16xf32>
  }
}

// -----

module @jit_collective_permute_1x32_rank_4_cluster_1_partial_target_pairs attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x16xf32>) -> (tensor<1x1x8192x16xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    return %0 : tensor<1x1x8192x16xf32>
  }
}

// -----

module @jit_collective_permute_1x32_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x256x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,32,1]<=[32]}"} : (tensor<1x1x256x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 32, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x256x512xf32>) -> (tensor<1x1x256x512xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<1x1x256x512xf32>) -> tensor<1x1x256x512xf32>
    return %0 : tensor<1x1x256x512xf32>
  }
}

// -----

module @jit_collective_permute_1x8_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x64xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]]> : tensor<8x2xi64>}> : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    return %0 : tensor<1x1x8192x64xf32>
  }
}

// -----

module @jit_collective_permute_1x8_rank_4_cluster_1_partial_target_pairs attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [2, 3], [4, 5], [6, 7]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x64xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    return %0 : tensor<1x1x8192x64xf32>
  }
}

// -----

module @jit_collective_permute_1x8_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,1]<=[8]}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x512xf32>) -> (tensor<1x1x1024x512xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<> : tensor<0x2xi64>}> : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    return %0 : tensor<1x1x1024x512xf32>
  }
}

// -----

module @jit_collective_permute_2x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>) -> (tensor<1x1x4096x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4]]> : tensor<8x2xi64>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    return %0 : tensor<1x1x4096x128xf32>
  }
}

// -----

module @jit_collective_permute_2x4_rank_4_cluster_1_partial_target_pairs attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [2, 3], [4, 5], [6, 7]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>) -> (tensor<1x1x4096x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    return %0 : tensor<1x1x4096x128xf32>
  }
}

// -----

module @jit_collective_permute_2x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x2048x256xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x256xf32>) -> (tensor<1x1x2048x256xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 4], [4, 0], [1, 5], [5, 1], [2, 6], [6, 2], [3, 7], [7, 3]]> : tensor<8x2xi64>}> : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
    return %0 : tensor<1x1x2048x256xf32>
  }
}

// -----

module @jit_collective_permute_2x4_rank_4_cluster_0_partial_target_pairs attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,2]<=[2,4]T(1,0)}"} : (tensor<1x1x2048x256xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x256xf32>) -> (tensor<1x1x2048x256xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x2048x256xf32>) -> tensor<1x1x2048x256xf32>
    return %0 : tensor<1x1x2048x256xf32>
  }
}

// -----

module @jit_collective_permute_8x4_rank_4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [8, 9], [9, 10], [10, 11], [11, 8], [12, 13], [13, 14], [14, 15], [15, 12], [16, 17], [17, 18], [18, 19], [19, 16], [20, 21], [21, 22], [22, 23], [23, 20], [24, 25], [25, 26], [26, 27], [27, 24], [28, 29], [29, 30], [30, 31], [31, 28]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>) -> (tensor<1x1x1024x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [8, 9], [9, 10], [10, 11], [11, 8], [12, 13], [13, 14], [14, 15], [15, 12], [16, 17], [17, 18], [18, 19], [19, 16], [20, 21], [21, 22], [22, 23], [23, 20], [24, 25], [25, 26], [26, 27], [27, 24], [28, 29], [29, 30], [30, 31], [31, 28]]> : tensor<32x2xi64>}> : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    return %0 : tensor<1x1x1024x128xf32>
  }
}

// -----

module @jit_collective_permute_8x4_rank_4_cluster_1_partial_target_pairs attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    // CHECK-SAME: [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>) -> (tensor<1x1x1024x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15], [16, 17], [18, 19], [20, 21], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31]]> : tensor<16x2xi64>}> : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    return %0 : tensor<1x1x1024x128xf32>
  }
}

// -----

module @jit_collective_permute_8x4_rank_4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x64xf32>) -> (tensor<1x1x2048x64xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 4], [4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 0], [1, 5], [5, 9], [9, 13], [13, 17], [17, 21], [21, 25], [25, 29], [29, 1], [2, 6], [6, 10], [10, 14], [14, 18], [18, 22], [22, 26], [26, 30], [30, 2], [3, 7], [7, 11], [11, 15], [15, 19], [19, 23], [23, 27], [27, 31], [31, 3]]> : tensor<32x2xi64>}> : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    return %0 : tensor<1x1x2048x64xf32>
  }
}

// -----

module @jit_collective_permute_8x4_rank_4_cluster_0_partial_target_pairs attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x2048x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    // CHECK: "ttir.collective_permute"
    // CHECK-SAME: source_target_pairs = dense
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,4,8]<=[8,4]T(1,0)}"} : (tensor<1x1x2048x64xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 2>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 4, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x2048x64xf32>) -> (tensor<1x1x2048x64xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_permute"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 4], [8, 12], [16, 20], [24, 28], [1, 5], [9, 13], [17, 21], [25, 29], [2, 6], [10, 14], [18, 22], [26, 30], [3, 7], [11, 15], [19, 23], [27, 31]]> : tensor<16x2xi64>}> : (tensor<1x1x2048x64xf32>) -> tensor<1x1x2048x64xf32>
    return %0 : tensor<1x1x2048x64xf32>
  }
}

// -----

module @jit_collective_broadcast_1x2_cluster_1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 1]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2]<=[2]}"} : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x256xf32>) -> (tensor<1x1x8192x256xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>}> : (tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %0 : tensor<1x1x8192x256xf32>
  }
}

// -----

module @collective_broadcast_1x8_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 1, 2, 3, 4, 5, 6, 7]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x64xf32>) -> (tensor<1x1x8192x64xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>}> : (tensor<1x1x8192x64xf32>) -> tensor<1x1x8192x64xf32>
    return %0 : tensor<1x1x8192x64xf32>
  }
}

// -----

module @collective_broadcast_2x4_cluster_0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 4], [1, 5], [2, 6], [3, 7]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>) -> (tensor<1x1x4096x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    return %0 : tensor<1x1x4096x128xf32>
  }
}

// -----

module @collective_broadcast_2x4_cluster_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 1, 2, 3], [4, 5, 6, 7]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,2,4]<=[8]}"} : (tensor<1x1x4096x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x4096x128xf32>) -> (tensor<1x1x4096x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>}> : (tensor<1x1x4096x128xf32>) -> tensor<1x1x4096x128xf32>
    return %0 : tensor<1x1x4096x128xf32>
  }
}

// -----

module @collective_broadcast_1x32_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,32]<=[32]}"} : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 32>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x8192x16xf32>) -> (tensor<1x1x8192x16xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>}> : (tensor<1x1x8192x16xf32>) -> tensor<1x1x8192x16xf32>
    return %0 : tensor<1x1x8192x16xf32>
  }
}

// -----

module @collective_broadcast_8x4_cluster_0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>) -> (tensor<1x1x1024x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>}> : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    return %0 : tensor<1x1x1024x128xf32>
  }
}

// -----

module @collective_broadcast_8x4_cluster_1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x8192x512xf32>) -> (tensor<1x1x8192x512xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x8192x512xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x8192x512xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    // CHECK: "ttir.collective_broadcast"
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,8,4]<=[32]}"} : (tensor<1x1x1024x128xf32>) -> tensor<1x1x8192x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 2, 3>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 8, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1x1x8192x512xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x128xf32>) -> (tensor<1x1x1024x128xf32> {jax.result_info = "[None, None, ('batch',), ('pipeline',)]"}) {
    %0 = "stablehlo.collective_broadcast"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>}> : (tensor<1x1x1024x128xf32>) -> tensor<1x1x1024x128xf32>
    return %0 : tensor<1x1x1024x128xf32>
  }
}

// -----

// torchax - GSPMD test with multi-user case
module @jit_jax_wrapper attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1024x1024xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}) -> (tensor<1024x1024xf32> {jax.result_info = "", mhlo.sharding = "{devices=[8,1]<=[8]}"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<1024x1024xf32>) -> tensor<128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<identity>
    %2 = call @shmap_body(%1) : (tensor<128x1024xf32>) -> tensor<128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{manual}"} : (tensor<128x1024xf32>) -> tensor<128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x1024xf32>) -> tensor<1024x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<1024x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<128x1024xf32>) -> (tensor<128x1024xf32> {jax.result_info = "[('torch_dist',), None]"}) {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<128x1024xf32>
    %1 = stablehlo.multiply %arg0, %0 : tensor<128x1024xf32>
    %2 = stablehlo.add %arg0, %1 : tensor<128x1024xf32>
    return %2 : tensor<128x1024xf32>
  }
}

// -----

module @SyncTensorsGraph.13 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<f32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<8192x4096xf32>, %arg2: tensor<1024x8192xf32>) -> tensor<1024x4096xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[2,4]0,1,2,3,4,5,6,7}"} : (tensor<1024x8192xf32>) -> tensor<1024x8192xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x8192xf32>) -> tensor<512x2048xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,2]0,4,1,5,2,6,3,7 last_tile_dim_replicate}"} : (tensor<8192x4096xf32>) -> tensor<8192x4096xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x4096xf32>) -> tensor<2048x4096xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %4 = stablehlo.dot_general %1, %3, contracting_dims = [1] x [0] : (tensor<512x2048xf32>, tensor<2048x4096xf32>) -> tensor<512x4096xf32>
    %5 = stablehlo.broadcast_in_dim %arg0, dims = [] : (tensor<f32>) -> tensor<512x4096xf32>
    %6 = stablehlo.add %4, %5 : tensor<512x4096xf32>
    %7 = "stablehlo.all_reduce"(%6) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %8 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %8 : tensor<f32>
    }) : (tensor<512x4096xf32>) -> tensor<512x4096xf32>
    %9 = stablehlo.custom_call @Sharding(%7) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<512x4096xf32>) -> tensor<512x4096xf32>
    %10 = stablehlo.custom_call @SPMDShardToFullShape(%9) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<512x4096xf32>) -> tensor<1024x4096xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %10 : tensor<1024x4096xf32>
  }
}

// -----

module @jit_negative_basic attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<256x256xf32> {mhlo.sharding = "{devices=[1,2]<=[2]}"}) -> (tensor<256x128xf32> {jax.result_info = "", mhlo.sharding = "{replicated}"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<256x256xf32>) -> tensor<256x256xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<256x256xf32>) -> tensor<256x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #ttcore.shard_type<identity>
    %2 = call @shmap_body(%1) : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {mhlo.sharding = "{manual}"} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {mhlo.sharding = "{replicated}"} : (tensor<256x128xf32>) -> tensor<256x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
    return %4 : tensor<256x128xf32>
  }
  func.func private @shmap_body(%arg0: tensor<256x128xf32>) -> (tensor<256x128xf32> {jax.result_info = "[None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<256x128xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %2 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<256x128xf32>) -> tensor<256x128xf32>
    return %1 : tensor<256x128xf32>
  }
}

// -----

module @SyncTensorsGraph.8 attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<8x1024xf32>) -> tensor<8x128xf32> {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]0,1,2,3,4,5,6,7}"} : (tensor<8x1024xf32>) -> tensor<8x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8x1024xf32>) -> tensor<8x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
      %3 = stablehlo.add %arg1, %arg2 : tensor<f32>
      stablehlo.return %3 : tensor<f32>
    }) : (tensor<8x128xf32>) -> tensor<8x128xf32>
    return %2 : tensor<8x128xf32>
  }
}


// -----

module @all_to_all_1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x128xf32>) -> (tensor<16x16xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<2x128xf32>) -> tensor<2x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2x128xf32>) -> tensor<2x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2x16xf32>) -> tensor<16x2xf32>
    // CHECK: "ttir.all_to_all"
    // CHECK-SAME: concat_dim = 0 : si32
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: split_count = 8 : si32
    // CHECK-SAME: split_dim = 1 : si32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<16x2xf32>) -> tensor<16x2xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<16x2xf32>) -> tensor<16x16xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<16x16xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2x16xf32>) -> (tensor<16x2xf32> {jax.result_info = "[None, ('x',)]"}) {
    %0 = "stablehlo.all_to_all"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, split_count = 8 : i64, split_dimension = 1 : i64}> : (tensor<2x16xf32>) -> tensor<16x2xf32>
    return %0 : tensor<16x2xf32>
  }
}

// -----

module @all_to_all_2 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x128xf32>) -> (tensor<16x32xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4x128xf32>) -> tensor<4x32xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<4x32xf32>) -> tensor<16x8xf32>
    // CHECK: "ttir.all_to_all"
    // CHECK-SAME: concat_dim = 0 : si32
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: split_count = 4 : si32
    // CHECK-SAME: split_dim = 1 : si32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<16x8xf32>) -> tensor<16x8xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<16x8xf32>) -> tensor<16x32xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<16x32xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4x32xf32>) -> (tensor<16x8xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = "stablehlo.all_to_all"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 1 : i64}> : (tensor<4x32xf32>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}

// -----

module @all_to_all_3 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<4x128xf32>) -> (tensor<16x32xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4x128xf32>) -> tensor<2x32xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2x32xf32>) -> tensor<8x8xf32>
    // CHECK: "ttir.all_to_all"
    // CHECK-SAME: concat_dim = 0 : si32
    // CHECK-SAME: replica_groups = dense
    // CHECK-SAME: split_count = 4 : si32
    // CHECK-SAME: split_dim = 1 : si32
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8x8xf32>) -> tensor<8x8xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8x8xf32>) -> tensor<16x32xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 4>
    // CHECK-SAME: shard_type = #ttcore.shard_type<devices>
    return %4 : tensor<16x32xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2x32xf32>) -> (tensor<8x8xf32> {jax.result_info = "[('x',), ('y',)]"}) {
    %0 = "stablehlo.all_to_all"(%arg0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, split_count = 4 : i64, split_dimension = 1 : i64}> : (tensor<2x32xf32>) -> tensor<8x8xf32>
    return %0 : tensor<8x8xf32>
  }
}
