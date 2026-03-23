// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
// UNSUPPORTED: true

module @jit_loss_tp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg1: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg2: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg3: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg4: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg5: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg6: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg7: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg8: tensor<128x128xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg9: tensor<128xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg10: tensor<128x8xf32> {mhlo.sharding = "{devices=[8,1]<=[8]}"}, %arg11: tensor<8xf32> {mhlo.sharding = "{devices=[8]<=[8]}"}, %arg12: tensor<32x784xf32> {mhlo.sharding = "{devices=[1,8]<=[8]}"}, %arg13: tensor<32x8xf32> {mhlo.sharding = "{devices=[1,8]<=[8]}"}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg12) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x784xf32>) -> tensor<32x98xf32>
    %2 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x128xf32>) -> tensor<98x128xf32>
    %4 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %6 = call @shmap_body(%1, %3, %5) : (tensor<32x98xf32>, tensor<98x128xf32>, tensor<16xf32>) -> tensor<32x16xf32>
    %7 = stablehlo.custom_call @Sharding(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    %8 = stablehlo.custom_call @SPMDShardToFullShape(%7) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x16xf32>) -> tensor<32x128xf32>
    %9 = call @relu(%8) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %10 = stablehlo.custom_call @Sharding(%9) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %12 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %14 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %15 = stablehlo.custom_call @SPMDFullToShardShape(%14) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %16 = call @shmap_body_0(%11, %13, %15) : (tensor<32x16xf32>, tensor<16x128xf32>, tensor<16xf32>) -> tensor<32x16xf32>
    %17 = stablehlo.custom_call @Sharding(%16) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    %18 = stablehlo.custom_call @SPMDShardToFullShape(%17) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x16xf32>) -> tensor<32x128xf32>
    %19 = call @relu(%18) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %20 = stablehlo.custom_call @Sharding(%19) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %21 = stablehlo.custom_call @SPMDFullToShardShape(%20) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %22 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %23 = stablehlo.custom_call @SPMDFullToShardShape(%22) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %24 = stablehlo.custom_call @Sharding(%arg5) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %25 = stablehlo.custom_call @SPMDFullToShardShape(%24) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %26 = call @shmap_body_1(%21, %23, %25) : (tensor<32x16xf32>, tensor<16x128xf32>, tensor<16xf32>) -> tensor<32x16xf32>
    %27 = stablehlo.custom_call @Sharding(%26) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    %28 = stablehlo.custom_call @SPMDShardToFullShape(%27) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x16xf32>) -> tensor<32x128xf32>
    %29 = call @relu(%28) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %30 = stablehlo.custom_call @Sharding(%29) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %31 = stablehlo.custom_call @SPMDFullToShardShape(%30) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %32 = stablehlo.custom_call @Sharding(%arg6) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %33 = stablehlo.custom_call @SPMDFullToShardShape(%32) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %34 = stablehlo.custom_call @Sharding(%arg7) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %35 = stablehlo.custom_call @SPMDFullToShardShape(%34) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %36 = call @shmap_body_2(%31, %33, %35) : (tensor<32x16xf32>, tensor<16x128xf32>, tensor<16xf32>) -> tensor<32x16xf32>
    %37 = stablehlo.custom_call @Sharding(%36) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    %38 = stablehlo.custom_call @SPMDShardToFullShape(%37) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x16xf32>) -> tensor<32x128xf32>
    %39 = call @relu(%38) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %40 = stablehlo.custom_call @Sharding(%39) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %41 = stablehlo.custom_call @SPMDFullToShardShape(%40) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %42 = stablehlo.custom_call @Sharding(%arg8) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %43 = stablehlo.custom_call @SPMDFullToShardShape(%42) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %44 = stablehlo.custom_call @Sharding(%arg9) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<128xf32>) -> tensor<128xf32>
    %45 = stablehlo.custom_call @SPMDFullToShardShape(%44) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %46 = call @shmap_body_3(%41, %43, %45) : (tensor<32x16xf32>, tensor<16x128xf32>, tensor<16xf32>) -> tensor<32x16xf32>
    %47 = stablehlo.custom_call @Sharding(%46) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x16xf32>) -> tensor<32x16xf32>
    %48 = stablehlo.custom_call @SPMDShardToFullShape(%47) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x16xf32>) -> tensor<32x128xf32>
    %49 = call @relu(%48) : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %50 = stablehlo.custom_call @Sharding(%49) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x128xf32>) -> tensor<32x128xf32>
    %51 = stablehlo.custom_call @SPMDFullToShardShape(%50) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %52 = stablehlo.custom_call @Sharding(%arg10) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[8]}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %53 = stablehlo.custom_call @SPMDFullToShardShape(%52) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x8xf32>) -> tensor<16x8xf32>
    %54 = stablehlo.custom_call @Sharding(%arg11) {backend_config = "", mhlo.sharding = "{devices=[8]<=[8]}"} : (tensor<8xf32>) -> tensor<8xf32>
    %55 = stablehlo.custom_call @SPMDFullToShardShape(%54) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8xf32>) -> tensor<1xf32>
    %56 = call @shmap_body_4(%51, %53, %55) : (tensor<32x16xf32>, tensor<16x8xf32>, tensor<1xf32>) -> tensor<32x1xf32>
    %57 = stablehlo.custom_call @Sharding(%56) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x1xf32>) -> tensor<32x1xf32>
    %58 = stablehlo.custom_call @SPMDShardToFullShape(%57) {backend_config = "", mhlo.sharding = "{devices=[1,8]<=[8]}"} : (tensor<32x1xf32>) -> tensor<32x8xf32>
    %59 = stablehlo.subtract %58, %arg13 : tensor<32x8xf32>
    %60 = stablehlo.multiply %59, %59 : tensor<32x8xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %61 = stablehlo.reduce(%60 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<32x8xf32>, tensor<f32>) -> tensor<32xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %62 = stablehlo.reduce(%61 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<32xf32>, tensor<f32>) -> tensor<f32>
    %cst_1 = stablehlo.constant dense<3.200000e+01> : tensor<f32>
    %63 = stablehlo.divide %62, %cst_1 : tensor<f32>
    return %63 : tensor<f32>
  }
  func.func private @shmap_body(%arg0: tensor<32x98xf32>, %arg1: tensor<98x128xf32>, %arg2: tensor<16xf32>) -> (tensor<32x16xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x98xf32>, tensor<98x128xf32>) -> tensor<32x128xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.add %1, %3 : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
  func.func private @relu(%arg0: tensor<32x128xf32>) -> tensor<32x128xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x128xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<32x128xf32>
    return %1 : tensor<32x128xf32>
  }
  func.func private @shmap_body_0(%arg0: tensor<32x16xf32>, %arg1: tensor<16x128xf32>, %arg2: tensor<16xf32>) -> (tensor<32x16xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.add %1, %3 : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
  func.func private @shmap_body_1(%arg0: tensor<32x16xf32>, %arg1: tensor<16x128xf32>, %arg2: tensor<16xf32>) -> (tensor<32x16xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.add %1, %3 : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
  func.func private @shmap_body_2(%arg0: tensor<32x16xf32>, %arg1: tensor<16x128xf32>, %arg2: tensor<16xf32>) -> (tensor<32x16xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.add %1, %3 : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
  func.func private @shmap_body_3(%arg0: tensor<32x16xf32>, %arg1: tensor<16x128xf32>, %arg2: tensor<16xf32>) -> (tensor<32x16xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x128xf32>) -> tensor<32x128xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x128xf32>) -> tensor<32x16xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<16xf32>) -> tensor<1x16xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x16xf32>) -> tensor<32x16xf32>
    %4 = stablehlo.add %1, %3 : tensor<32x16xf32>
    return %4 : tensor<32x16xf32>
  }
  func.func private @shmap_body_4(%arg0: tensor<32x16xf32>, %arg1: tensor<16x8xf32>, %arg2: tensor<1xf32>) -> (tensor<32x1xf32> {jax.result_info = "[None, ('y',)]"}) {
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x16xf32>, tensor<16x8xf32>) -> tensor<32x8xf32>
    %1 = "stablehlo.reduce_scatter"(%0) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
      %5 = stablehlo.add %arg3, %arg4 : tensor<f32>
      stablehlo.return %5 : tensor<f32>
    }) : (tensor<32x8xf32>) -> tensor<32x1xf32>
    %2 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<1xf32>) -> tensor<1x1xf32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [0, 1] : (tensor<1x1xf32>) -> tensor<32x1xf32>
    %4 = stablehlo.add %1, %3 : tensor<32x1xf32>
    return %4 : tensor<32x1xf32>
  }
}


// CHECK-LABEL @main
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
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
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
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
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
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
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
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
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
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
// CHECK-SAME: shard_dims = array<i64: -1, 1>
// CHECK-SAME: shard_direction = #ttcore.shard_direction<shard_to_full>
// CHECK-SAME: shard_shape = array<i64: 1, 8>
// CHECK-SAME: shard_type = #ttcore.shard_type<devices>
