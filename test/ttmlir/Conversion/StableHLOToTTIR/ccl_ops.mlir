// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

// jax/pjrt sharding target 1x2 for n300 all_reduce
module @all_reduce_1x2 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x392xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<392x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8192x392xf32>, tensor<392x16384xf32>) -> tensor<8192x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #tt.shard_type<replicate>
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

// jax/pjrt sharding target 2x4 for t3k all_reduce
module @all_reduce_2x4 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<4096x196xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<196x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<4096x196xf32>, tensor<196x16384xf32>) -> tensor<4096x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[2,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<4096x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
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

// jax/pjrt sharding target 1x8 for t3k all_reduce
module @all_reduce_1x8 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<784x16384xf32> {mhlo.layout_mode = "default"}) -> (tensor<8192x16384xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x98xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<98x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8192x98xf32>, tensor<98x16384xf32>) -> tensor<8192x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #tt.shard_type<replicate>
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

// jax/pjrt sharding target 8x4 for tg all_reduce
module @all_reduce_8x4 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>, %arg1: tensor<784x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x196xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[4,1,8]<=[8,4]T(1,0) last_tile_dim_replicate}"} : (tensor<784x16384xf32>) -> tensor<784x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x16384xf32>) -> tensor<196x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<1024x196xf32>, tensor<196x16384xf32>) -> tensor<1024x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1024x16384xf32>) -> tensor<1024x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{devices=[8,1,4]<=[32] last_tile_dim_replicate}"} : (tensor<1024x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 0, -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x196xf32>, %arg1: tensor<196x16384xf32>) -> (tensor<1024x16384xf32> {jax.result_info = "[('batch',), None]"}) {
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

// jax/pjrt sharding target 1x32 for tg all_reduce
module @all_reduce_1x32 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>, %arg1: tensor<800x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,32]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x25xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 32>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<800x16384xf32>) -> tensor<800x16384xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<800x16384xf32>) -> tensor<25x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 32, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %4 = call @shmap_body(%1, %3) : (tensor<8192x25xf32>, tensor<25x16384xf32>) -> tensor<8192x16384xf32>
    %5 = stablehlo.custom_call @Sharding(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    %6 = stablehlo.custom_call @SPMDShardToFullShape(%5) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1>
    // CHECK-SAME: shard_type = #tt.shard_type<replicate>
    return %6 : tensor<8192x16384xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x25xf32>, %arg1: tensor<25x16384xf32>) -> (tensor<8192x16384xf32> {jax.result_info = "[None, None]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x25xf32>, tensor<25x16384xf32>) -> tensor<8192x16384xf32>
    %1 = "stablehlo.all_reduce"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> ({
    ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
      %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<8192x16384xf32>) -> tensor<8192x16384xf32>
    // CHECK: %[[C:.*]] = "ttir.all_reduce"[[C:.*]]
    return %1 : tensor<8192x16384xf32>
  }
}

// jax/pjrt sharding target 1x2 for n300 all_gather cluster_axis=0
module @all_gather_1x2_cluster0 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x400xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<8192x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x400xf32>) -> (tensor<8192x400xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>, use_global_device_ids}> : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x400xf32>
  }
}

// jax/pjrt sharding target 1x2 for n300 all_gather cluster_axis=1
module @all_gather_1x2_cluster1 attributes {mhlo.num_partitions = 2 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<16384x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<4096x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<4096x800xf32>) -> tensor<8192x800xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,1]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<16384x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 2, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<16384x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4096x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<4096x800xf32>) -> tensor<8192x800xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x800xf32>
  }
}

// jax/pjrt sharding target 1x8 for t3k all_gather cluster_axis=0
module @all_gather_1x8_cluster0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x100xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<8192x100xf32>) -> tensor<8192x100xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x100xf32>) -> tensor<8192x100xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<8192x100xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<8192x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x100xf32>) -> (tensor<8192x100xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7]]> : tensor<8x1xi64>, use_global_device_ids}> : (tensor<8192x100xf32>) -> tensor<8192x100xf32>
    return %0 : tensor<8192x100xf32>
  }
}

// jax/pjrt sharding target 1x8 for t3k all_gather cluster_axis=1
module @all_gather_1x8_cluster1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<65536x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<1024x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1024x800xf32>) -> tensor<8192x800xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x800xf32>) -> tensor<65536x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<65536x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<1024x800xf32>) -> tensor<8192x800xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x800xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k all_gather cluster_axis=0
module @all_gather_2x4_cluster0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x1600xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<2048x400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2048x400xf32>) -> tensor<2048x800xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x800xf32>) -> tensor<2048x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<2048x800xf32>) -> tensor<8192x1600xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<8192x1600xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x400xf32>) -> (tensor<2048x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<2048x400xf32>) -> tensor<2048x800xf32>
    return %0 : tensor<2048x800xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k all_gather cluster_axis=1
module @all_gather_2x4_cluster1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<32768x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<2048x400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2048x400xf32>) -> tensor<8192x400xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[2,4]T(1,0)}"} : (tensor<8192x400xf32>) -> tensor<32768x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 4, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<32768x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x400xf32>) -> (tensor<8192x400xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<2048x400xf32>) -> tensor<8192x400xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x400xf32>
  }
}

// jax/pjrt sharding target 1x32 for tg all_gather cluster_axis=0
module @all_gather_1x32_cluster0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,32]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x25xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 32>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<8192x25xf32>) -> tensor<8192x25xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x25xf32>) -> tensor<8192x25xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,32]<=[32]}"} : (tensor<8192x25xf32>) -> tensor<8192x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 32>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<8192x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<8192x25xf32>) -> (tensor<8192x25xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31]]> : tensor<32x1xi64>, use_global_device_ids}> : (tensor<8192x25xf32>) -> tensor<8192x25xf32>
    return %0 : tensor<8192x25xf32>
  }
}

// jax/pjrt sharding target 1x32 for tg all_gather cluster_axis=1
module @all_gather_1x32_cluster1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<262144x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<256x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 32, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<256x800xf32>) -> tensor<8192x800xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x800xf32>) -> tensor<262144x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 32, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<262144x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<256x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> : (tensor<256x800xf32>) -> tensor<8192x800xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x800xf32>
  }
}

// jax/pjrt sharding target 8x4 for tg all_gather cluster_axis=0
module @all_gather_8x4_cluster0 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<8192x6400xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<2048x100xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 4, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<2048x100xf32>) -> tensor<2048x800xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<2048x800xf32>) -> tensor<2048x800xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[8,4]T(1,0)}"} : (tensor<2048x800xf32>) -> tensor<8192x6400xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 4, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<8192x6400xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x100xf32>) -> (tensor<2048x800xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<2048x100xf32>) -> tensor<2048x800xf32>
    return %0 : tensor<2048x800xf32>
  }
}

// jax/pjrt sharding target 8x4 for tg all_gather cluster_axis=1
module @all_gather_8x4_cluster1 attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x800xf32>) -> (tensor<65536x800xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[4,8]T(1,0)}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<1024x200xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 8, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1024x200xf32>) -> tensor<8192x200xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x200xf32>) -> tensor<8192x200xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[4,8]T(1,0)}"} : (tensor<8192x200xf32>) -> tensor<65536x800xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 0>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 8, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<65536x800xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x200xf32>) -> (tensor<8192x200xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<1024x200xf32>) -> tensor<8192x200xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x200xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "x", None, "y"]
module @jit_neg_basic0 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,4]<=[8]}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x512x128x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 3>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x512x128x256xf32>) -> tensor<1x512x128x256xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x512x128x256xf32>) -> tensor<1x512x128x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,4]<=[8]}"} : (tensor<1x512x128x256xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, 3>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x512x128x256xf32>) -> (tensor<1x512x128x256xf32> {jax.result_info = "[None, ('x',), None, ('y',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x512x128x256xf32>
    return %0 : tensor<1x512x128x256xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "x", None, None]
module @jit_neg_basic1 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x512x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x512x128x1024xf32>) -> tensor<1x512x128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x512x128x1024xf32>) -> tensor<1x512x128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2,1,1,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x512x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 1, -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 2, 1, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x512x128x1024xf32>) -> (tensor<1x512x128x1024xf32> {jax.result_info = "[None, ('x',), None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x512x128x1024xf32>
    return %0 : tensor<1x512x128x1024xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, None, None, "y"]
module @jit_neg_basic2 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x256xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1024x128x256xf32>) -> tensor<1x1024x128x256xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x256xf32>) -> tensor<1x1024x128x256xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,4,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1024x128x256xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 4>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1024x128x256xf32>) -> (tensor<1x1024x128x256xf32> {jax.result_info = "[None, None, None, ('y',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x1024x128x256xf32>
    return %0 : tensor<1x1024x128x256xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "y", None, "x"]
module @jit_neg_basic3 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,2]<=[2,4]T(1,0)}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x256x128x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x256x128x512xf32>) -> tensor<1x256x128x512xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x256x128x512xf32>) -> tensor<1x256x128x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,2]<=[2,4]T(1,0)}"} : (tensor<1x256x128x512xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x256x128x512xf32>) -> (tensor<1x256x128x512xf32> {jax.result_info = "[None, ('y',), None, ('x',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x256x128x512xf32>
    return %0 : tensor<1x256x128x512xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, "y", None, None]
module @jit_neg_basic4 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x256x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x256x128x1024xf32>) -> tensor<1x256x128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x256x128x1024xf32>) -> tensor<1x256x128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,4,1,1,2]<=[2,4]T(1,0) last_tile_dim_replicate}"} : (tensor<1x256x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 4, 1, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x256x128x1024xf32>) -> (tensor<1x256x128x1024xf32> {jax.result_info = "[None, ('y',), None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x256x128x1024xf32>
    return %0 : tensor<1x256x128x1024xf32>
  }
}

// jax/pjrt sharding target 2x4 for t3k - GSPMD negative, sharding [None, None, None, "x"]
module @jit_neg_basic5 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1x1024x1024xf32>) -> (tensor<1x1x1024x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x1024xf32>) -> tensor<1x1x1024x512xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x512xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,2,4]<=[8] last_tile_dim_replicate}"} : (tensor<1x1x1024x512xf32>) -> tensor<1x1x1024x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: 3, -1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 2>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1x1024x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1x1024x512xf32>) -> (tensor<1x1x1024x512xf32> {jax.result_info = "[None, None, None, ('x',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x1x1024x512xf32>
    return %0 : tensor<1x1x1024x512xf32>
  }
}

// jax/pjrt sharding target 1x8 for t3k - GSPMD negative, sharding [None, None, None, "y"]
module @jit_neg_basic6 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x128xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x128xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x128xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,1,1,8]<=[8]}"} : (tensor<1x1024x128x128xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 3>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 1, 1, 8>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x1024x128x128xf32>) -> (tensor<1x1024x128x128xf32> {jax.result_info = "[None, None, None, ('y',)]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x1024x128x128xf32>
    return %0 : tensor<1x1024x128x128xf32>
  }
}

// jax/pjrt sharding target 1x8 for t3k - GSPMD negative, sharding [None, "y", None, None]
module @jit_neg_basic7 attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x1024x128x1024xf32>) -> (tensor<1x1024x128x1024xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,8,1,1]<=[8]}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x1024x128x1024xf32>) -> tensor<1x128x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
    // CHECK-SAME: shard_shape = array<i64: 1, 8, 1, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    %2 = call @shmap_body(%1) : (tensor<1x128x128x1024xf32>) -> tensor<1x128x128x1024xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<1x128x128x1024xf32>) -> tensor<1x128x128x1024xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,8,1,1]<=[8]}"} : (tensor<1x128x128x1024xf32>) -> tensor<1x1024x128x1024xf32>
    // CHECK: "ttir.mesh_shard"
    // CHECK-SAME: shard_dims = array<i64: -1, 1>
    // CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
    // CHECK-SAME: shard_shape = array<i64: 1, 8, 1, 1>
    // CHECK-SAME: shard_type = #tt.shard_type<devices>
    return %4 : tensor<1x1024x128x1024xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1x128x128x1024xf32>) -> (tensor<1x128x128x1024xf32> {jax.result_info = "[None, ('y',), None, None]"}) {
    %0 = stablehlo.negate %arg0 : tensor<1x128x128x1024xf32>
    return %0 : tensor<1x128x128x1024xf32>
  }
}
