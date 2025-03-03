// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_loss_dp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x128xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg2: tensor<128x128xf32> {mhlo.sharding = "{replicated}"}, %arg3: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg4: tensor<128x128xf32> {mhlo.sharding = "{replicated}"}, %arg5: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg6: tensor<128x128xf32> {mhlo.sharding = "{replicated}"}, %arg7: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg8: tensor<128x128xf32> {mhlo.sharding = "{replicated}"}, %arg9: tensor<128xf32> {mhlo.sharding = "{replicated}"}, %arg10: tensor<128x8xf32> {mhlo.sharding = "{replicated}"}, %arg11: tensor<8xf32> {mhlo.sharding = "{replicated}"}, %arg12: tensor<32x784xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg13: tensor<32x8xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {mhlo.sharding = "{replicated}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {mhlo.sharding = "{manual}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %2 = stablehlo.custom_call @Sharding(%arg1) {mhlo.sharding = "{replicated}"} : (tensor<128xf32>) -> tensor<128xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xf32>
    %4 = stablehlo.custom_call @Sharding(%arg2) {mhlo.sharding = "{replicated}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %6 = stablehlo.custom_call @Sharding(%arg3) {mhlo.sharding = "{replicated}"} : (tensor<128xf32>) -> tensor<128xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xf32>
    %8 = stablehlo.custom_call @Sharding(%arg4) {mhlo.sharding = "{replicated}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %10 = stablehlo.custom_call @Sharding(%arg5) {mhlo.sharding = "{replicated}"} : (tensor<128xf32>) -> tensor<128xf32>
    %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xf32>
    %12 = stablehlo.custom_call @Sharding(%arg6) {mhlo.sharding = "{replicated}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %14 = stablehlo.custom_call @Sharding(%arg7) {mhlo.sharding = "{replicated}"} : (tensor<128xf32>) -> tensor<128xf32>
    %15 = stablehlo.custom_call @SPMDFullToShardShape(%14) {mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xf32>
    %16 = stablehlo.custom_call @Sharding(%arg8) {mhlo.sharding = "{replicated}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %17 = stablehlo.custom_call @SPMDFullToShardShape(%16) {mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %18 = stablehlo.custom_call @Sharding(%arg9) {mhlo.sharding = "{replicated}"} : (tensor<128xf32>) -> tensor<128xf32>
    %19 = stablehlo.custom_call @SPMDFullToShardShape(%18) {mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<128xf32>
    %20 = stablehlo.custom_call @Sharding(%arg10) {mhlo.sharding = "{replicated}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %21 = stablehlo.custom_call @SPMDFullToShardShape(%20) {mhlo.sharding = "{manual}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %22 = stablehlo.custom_call @Sharding(%arg11) {mhlo.sharding = "{replicated}"} : (tensor<8xf32>) -> tensor<8xf32>
    %23 = stablehlo.custom_call @SPMDFullToShardShape(%22) {mhlo.sharding = "{manual}"} : (tensor<8xf32>) -> tensor<8xf32>
    %24 = stablehlo.custom_call @Sharding(%arg12) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %25 = stablehlo.custom_call @SPMDFullToShardShape(%24) {mhlo.sharding = "{manual}"} : (tensor<32x784xf32>) -> tensor<4x784xf32>
    %26 = stablehlo.custom_call @Sharding(%arg13) {mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<32x8xf32>) -> tensor<32x8xf32>
    %27 = stablehlo.custom_call @SPMDFullToShardShape(%26) {mhlo.sharding = "{manual}"} : (tensor<32x8xf32>) -> tensor<4x8xf32>
    %28 = call @shmap_body(%1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27) : (tensor<784x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<4x784xf32>, tensor<4x8xf32>) -> tensor<f32>
    %29 = stablehlo.custom_call @Sharding(%28) {mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<f32>
    %30 = stablehlo.custom_call @SPMDShardToFullShape(%29) {mhlo.sharding = "{replicated}"} : (tensor<f32>) -> tensor<f32>
    return %30 : tensor<f32>
  }
  func.func private @shmap_body(%arg0: tensor<784x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x8xf32>, %arg11: tensor<8xf32>, %arg12: tensor<4x784xf32>, %arg13: tensor<4x8xf32>) -> (tensor<f32> {jax.result_info = "[]"}) {
    %cst = stablehlo.constant dense<8.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
    %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.dot_general %arg12, %arg0, contracting_dims = [1] x [0] : (tensor<4x784xf32>, tensor<784x128xf32>) -> tensor<4x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %3 = stablehlo.add %0, %2 : tensor<4x128xf32>
    %4 = call @relu(%3) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %5 = stablehlo.dot_general %4, %arg2, contracting_dims = [1] x [0] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %6 = stablehlo.broadcast_in_dim %arg3, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %8 = stablehlo.add %5, %7 : tensor<4x128xf32>
    %9 = call @relu_0(%8) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %10 = stablehlo.dot_general %9, %arg4, contracting_dims = [1] x [0] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %11 = stablehlo.broadcast_in_dim %arg5, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %13 = stablehlo.add %10, %12 : tensor<4x128xf32>
    %14 = call @relu_1(%13) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %15 = stablehlo.dot_general %14, %arg6, contracting_dims = [1] x [0] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %16 = stablehlo.broadcast_in_dim %arg7, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %17 = stablehlo.broadcast_in_dim %16, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %18 = stablehlo.add %15, %17 : tensor<4x128xf32>
    %19 = call @relu_2(%18) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %20 = stablehlo.dot_general %19, %arg8, contracting_dims = [1] x [0] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
    %21 = stablehlo.broadcast_in_dim %arg9, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
    %22 = stablehlo.broadcast_in_dim %21, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
    %23 = stablehlo.add %20, %22 : tensor<4x128xf32>
    %24 = call @relu_3(%23) : (tensor<4x128xf32>) -> tensor<4x128xf32>
    %25 = stablehlo.dot_general %24, %arg10, contracting_dims = [1] x [0] : (tensor<4x128xf32>, tensor<128x8xf32>) -> tensor<4x8xf32>
    %26 = stablehlo.broadcast_in_dim %arg11, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
    %27 = stablehlo.broadcast_in_dim %26, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<4x8xf32>
    %28 = stablehlo.add %25, %27 : tensor<4x8xf32>
    %29 = stablehlo.subtract %28, %arg13 : tensor<4x8xf32>
    %30 = stablehlo.multiply %29, %29 : tensor<4x8xf32>
    %31 = stablehlo.reduce(%30 init: %cst_1) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
    %32 = stablehlo.reduce(%31 init: %cst_1) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
    %33 = stablehlo.divide %32, %cst_0 : tensor<f32>
    %34 = "stablehlo.all_reduce"(%33) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 0>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %36 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %36 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %35 = stablehlo.divide %34, %cst : tensor<f32>
    return %35 : tensor<f32>
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
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1>
// CHECK-SAME: shard_direction = #tt.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1>
// CHECK-SAME: shard_type = #tt.shard_type<replicate>
