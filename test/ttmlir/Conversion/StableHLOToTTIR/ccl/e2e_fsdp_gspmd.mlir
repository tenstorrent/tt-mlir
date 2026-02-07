// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_loss_fsdp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg1: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg2: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg3: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg4: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg5: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg6: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg7: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg8: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg9: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg10: tensor<128x8xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg11: tensor<8xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg12: tensor<32x784xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg13: tensor<32x8xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x128xf32>) -> tensor<98x128xf32>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %10 = stablehlo.custom_call @Sharding(%arg5) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %12 = stablehlo.custom_call @Sharding(%arg6) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %14 = stablehlo.custom_call @Sharding(%arg7) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %15 = stablehlo.custom_call @SPMDFullToShardShape(%14) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %16 = stablehlo.custom_call @Sharding(%arg8) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %17 = stablehlo.custom_call @SPMDFullToShardShape(%16) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %18 = stablehlo.custom_call @Sharding(%arg9) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %19 = stablehlo.custom_call @SPMDFullToShardShape(%18) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %20 = stablehlo.custom_call @Sharding(%arg10) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %21 = stablehlo.custom_call @SPMDFullToShardShape(%20) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x8xf32>) -> tensor<16x8xf32>
    %22 = stablehlo.custom_call @Sharding(%arg11) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<8xf32>) -> tensor<8xf32>
    %23 = stablehlo.custom_call @SPMDFullToShardShape(%22) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8xf32>) -> tensor<1xf32>
    %24 = stablehlo.custom_call @Sharding(%arg12) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %25 = stablehlo.custom_call @SPMDFullToShardShape(%24) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x784xf32>) -> tensor<4x784xf32>
    %26 = stablehlo.custom_call @Sharding(%arg13) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<32x8xf32>) -> tensor<32x8xf32>
    %27 = stablehlo.custom_call @SPMDFullToShardShape(%26) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x8xf32>) -> tensor<4x8xf32>
    %28 = call @shmap_body(%1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27) : (tensor<98x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x8xf32>, tensor<1xf32>, tensor<4x784xf32>, tensor<4x8xf32>) -> tensor<f32>
    %29 = stablehlo.custom_call @Sharding(%28) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<f32>
    %30 = stablehlo.custom_call @SPMDShardToFullShape(%29) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<f32>) -> tensor<f32>
    return %30 : tensor<f32>
  }
  func.func private @shmap_body(%arg0: tensor<98x128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16x128xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16x128xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16x128xf32>, %arg7: tensor<16xf32>, %arg8: tensor<16x128xf32>, %arg9: tensor<16xf32>, %arg10: tensor<16x8xf32>, %arg11: tensor<1xf32>, %arg12: tensor<4x784xf32>, %arg13: tensor<4x8xf32>) -> (tensor<f32> {jax.result_info = "[]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<98x128xf32>) -> tensor<784x128xf32>
    %1 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
    %2 = stablehlo.dot_general %arg12, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x784xf32>, tensor<784x128xf32>) -> tensor<4x128xf32>
    %3 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %4 = stablehlo.broadcast_in_dim %3, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %5 = stablehlo.add %2, %4 : tensor<4x128xf32>
    %6 = call @relu(%5) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %7 = "stablehlo.all_gather"(%arg2) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
    %8 = "stablehlo.all_gather"(%arg3) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
    %9 = stablehlo.dot_general %6, %7, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %10 = stablehlo.broadcast_in_dim %8, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %11 = stablehlo.broadcast_in_dim %10, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %12 = stablehlo.add %9, %11 : tensor<4x128xf32>
    %13 = call @relu_0(%12) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %14 = "stablehlo.all_gather"(%arg4) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
    %15 = "stablehlo.all_gather"(%arg5) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
    %16 = stablehlo.dot_general %13, %14, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %17 = stablehlo.broadcast_in_dim %15, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %18 = stablehlo.broadcast_in_dim %17, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %19 = stablehlo.add %16, %18 : tensor<4x128xf32>
    %20 = call @relu_1(%19) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %21 = "stablehlo.all_gather"(%arg6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
    %22 = "stablehlo.all_gather"(%arg7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
    %23 = stablehlo.dot_general %20, %21, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %24 = stablehlo.broadcast_in_dim %22, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %25 = stablehlo.broadcast_in_dim %24, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %26 = stablehlo.add %23, %25 : tensor<4x128xf32>
    %27 = call @relu_2(%26) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %28 = "stablehlo.all_gather"(%arg8) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
    %29 = "stablehlo.all_gather"(%arg9) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
    %30 = stablehlo.dot_general %27, %28, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %31 = stablehlo.broadcast_in_dim %29, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %32 = stablehlo.broadcast_in_dim %31, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %33 = stablehlo.add %30, %32 : tensor<4x128xf32>
    %34 = call @relu_3(%33) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %35 = "stablehlo.all_gather"(%arg10) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x8xf32>) -> tensor<128x8xf32>
    %36 = "stablehlo.all_gather"(%arg11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<1xf32>) -> tensor<8xf32>
    %37 = stablehlo.dot_general %34, %35, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x8xf32>) -> tensor<4x8xf32>
    %38 = stablehlo.broadcast_in_dim %36, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %39 = stablehlo.broadcast_in_dim %38, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<4x8xf32>
    %40 = stablehlo.add %37, %39 : tensor<4x8xf32>
    %41 = stablehlo.subtract %40, %arg13 : tensor<4x8xf32>
    %42 = stablehlo.multiply %41, %41 : tensor<4x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %43 = stablehlo.reduce(%42 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %44 = stablehlo.reduce(%43 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %cst_1 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %45 = stablehlo.divide %44, %cst_1 : tensor<f32>
    %46 = "stablehlo.all_reduce"(%45) <{channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %48 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %48 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %cst_2 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %47 = stablehlo.divide %46, %cst_2 : tensor<f32>
    return %47 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_0(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_1(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_2(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
  func.func private @relu_3(%arg0: tensor<4x128xf32>) -> tensor<4x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<4x128xf32>
    return %1 : tensor<4x128xf32>
  }
}

// CHECK-LABEL @main
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<identity>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #ttcore.shard_type<replicate>
