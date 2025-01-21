// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
// UNSUPPORTED: true

module @jit_fwd_one attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>) -> (tensor<65536x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x784xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = call @shmap_body(%1) : (tensor<1024x784xf32>) -> tensor<8192x784xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<65536x784xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %4 : tensor<65536x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x784xf32>) -> (tensor<8192x784xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<1024x784xf32>) -> tensor<8192x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x784xf32>
  }
}

module @jit_fwd_two attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>) -> (tensor<32768x1568xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<2048x392xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = call @shmap_body(%1) : (tensor<2048x392xf32>) -> tensor<8192x784xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,2]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<32768x1568xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %4 : tensor<32768x1568xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x392xf32>) -> (tensor<8192x784xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1], [2, 3], [4, 5], [6, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<2048x392xf32>) -> tensor<2048x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    %1 = "stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 2, 4, 6], [1, 3, 5, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<2048x784xf32>) -> tensor<8192x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %1 : tensor<8192x784xf32>
  }
}

module @jit_fwd_three attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>) -> (tensor<16384x3136xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<4096x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = call @shmap_body(%1) : (tensor<4096x196xf32>) -> tensor<8192x784xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<8192x784xf32>) -> tensor<16384x3136xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %4 : tensor<16384x3136xf32>
  }
  func.func private @shmap_body(%arg0: tensor<4096x196xf32>) -> (tensor<8192x784xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> : (tensor<4096x196xf32>) -> tensor<4096x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    %1 = "stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<4096x784xf32>) -> tensor<8192x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %1 : tensor<8192x784xf32>
  }
}

module @jit_fwd_four attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>) -> (tensor<262144x784xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<256x784xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = call @shmap_body(%1) : (tensor<256x784xf32>) -> tensor<8192x784xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[32,1]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<262144x784xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %4 : tensor<262144x784xf32>
  }
  func.func private @shmap_body(%arg0: tensor<256x784xf32>) -> (tensor<8192x784xf32> {jax.result_info = "[('batch',), None]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<1x32xi64>, use_global_device_ids}> : (tensor<256x784xf32>) -> tensor<8192x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %0 : tensor<8192x784xf32>
  }
}

module @jit_fwd_five attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>) -> (tensor<65536x3136xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<1024x196xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = call @shmap_body(%1) : (tensor<1024x196xf32>) -> tensor<8192x784xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[8,4]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<65536x3136xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %4 : tensor<65536x3136xf32>
  }
  func.func private @shmap_body(%arg0: tensor<1024x196xf32>) -> (tensor<8192x784xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23], [24, 25, 26, 27], [28, 29, 30, 31]]> : tensor<8x4xi64>, use_global_device_ids}> : (tensor<1024x196xf32>) -> tensor<1024x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    %1 = "stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 4, 8, 12, 16, 20, 24, 28], [1, 5, 9, 13, 17, 21, 25, 29], [2, 6, 10, 14, 18, 22, 26, 30], [3, 7, 11, 15, 19, 23, 27, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<1024x784xf32>) -> tensor<8192x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %1 : tensor<8192x784xf32>
  }
}

module @jit_fwd_six attributes {mhlo.num_partitions = 32 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<8192x784xf32>) -> (tensor<32768x6272xf32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<2048x98xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    %2 = call @shmap_body(%1) : (tensor<2048x98xf32>) -> tensor<8192x784xf32>
    %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x784xf32>) -> tensor<8192x784xf32>
    %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[4,8]<=[32]}"} : (tensor<8192x784xf32>) -> tensor<32768x6272xf32>
    // CHECK: %[[C:.*]] = "ttir.mesh_shard"[[C:.*]]
    return %4 : tensor<32768x6272xf32>
  }
  func.func private @shmap_body(%arg0: tensor<2048x98xf32>) -> (tensor<8192x784xf32> {jax.result_info = "[('batch',), ('model',)]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29, 30, 31]]> : tensor<4x8xi64>, use_global_device_ids}> : (tensor<2048x98xf32>) -> tensor<2048x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    %1 = "stablehlo.all_gather"(%0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 8, 16, 24], [1, 9, 17, 25], [2, 10, 18, 26], [3, 11, 19, 27], [4, 12, 20, 28], [5, 13, 21, 29], [6, 14, 22, 30], [7, 15, 23, 31]]> : tensor<8x4xi64>, use_global_device_ids}> : (tensor<2048x784xf32>) -> tensor<8192x784xf32>
    // CHECK: %[[C:.*]] = "ttir.all_gather"[[C:.*]]
    return %1 : tensor<8192x784xf32>
  }
}
