// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline %s | FileCheck %s
// UNSUPPORTED: true

module @jit_loss_fsdp_tp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<784x128xf32>, %arg1: tensor<128xf32>, %arg2: tensor<128x128xf32>, %arg3: tensor<128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128xf32>, %arg6: tensor<128x128xf32>, %arg7: tensor<128xf32>, %arg8: tensor<128x128xf32>, %arg9: tensor<128xf32>, %arg10: tensor<128x8xf32>, %arg11: tensor<8xf32>, %arg12: tensor<32x784xf32>, %arg13: tensor<32x8xf32>) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[2,4]T(1,0)}"} : (tensor<784x128xf32>) -> tensor<784x128xf32>
    %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<784x128xf32>) -> tensor<98x128xf32>
    %2 = stablehlo.custom_call @Sharding(%arg1) {backend_config = "", mhlo.sharding = "{devices=[8]<=[2,4]T(1,0)}"} : (tensor<128xf32>) -> tensor<128xf32>
    %3 = stablehlo.custom_call @SPMDFullToShardShape(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %4 = stablehlo.custom_call @Sharding(%arg2) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[2,4]T(1,0)}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %5 = stablehlo.custom_call @SPMDFullToShardShape(%4) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %6 = stablehlo.custom_call @Sharding(%arg3) {backend_config = "", mhlo.sharding = "{devices=[8]<=[2,4]T(1,0)}"} : (tensor<128xf32>) -> tensor<128xf32>
    %7 = stablehlo.custom_call @SPMDFullToShardShape(%6) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %8 = stablehlo.custom_call @Sharding(%arg4) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[2,4]T(1,0)}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %9 = stablehlo.custom_call @SPMDFullToShardShape(%8) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %10 = stablehlo.custom_call @Sharding(%arg5) {backend_config = "", mhlo.sharding = "{devices=[8]<=[2,4]T(1,0)}"} : (tensor<128xf32>) -> tensor<128xf32>
    %11 = stablehlo.custom_call @SPMDFullToShardShape(%10) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %12 = stablehlo.custom_call @Sharding(%arg6) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[2,4]T(1,0)}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %13 = stablehlo.custom_call @SPMDFullToShardShape(%12) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %14 = stablehlo.custom_call @Sharding(%arg7) {backend_config = "", mhlo.sharding = "{devices=[8]<=[2,4]T(1,0)}"} : (tensor<128xf32>) -> tensor<128xf32>
    %15 = stablehlo.custom_call @SPMDFullToShardShape(%14) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %16 = stablehlo.custom_call @Sharding(%arg8) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[2,4]T(1,0)}"} : (tensor<128x128xf32>) -> tensor<128x128xf32>
    %17 = stablehlo.custom_call @SPMDFullToShardShape(%16) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x128xf32>) -> tensor<16x128xf32>
    %18 = stablehlo.custom_call @Sharding(%arg9) {backend_config = "", mhlo.sharding = "{devices=[8]<=[2,4]T(1,0)}"} : (tensor<128xf32>) -> tensor<128xf32>
    %19 = stablehlo.custom_call @SPMDFullToShardShape(%18) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128xf32>) -> tensor<16xf32>
    %20 = stablehlo.custom_call @Sharding(%arg10) {backend_config = "", mhlo.sharding = "{devices=[8,1]<=[2,4]T(1,0)}"} : (tensor<128x8xf32>) -> tensor<128x8xf32>
    %21 = stablehlo.custom_call @SPMDFullToShardShape(%20) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<128x8xf32>) -> tensor<16x8xf32>
    %22 = stablehlo.custom_call @Sharding(%arg11) {backend_config = "", mhlo.sharding = "{devices=[8]<=[2,4]T(1,0)}"} : (tensor<8xf32>) -> tensor<8xf32>
    %23 = stablehlo.custom_call @SPMDFullToShardShape(%22) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8xf32>) -> tensor<1xf32>
    %24 = stablehlo.custom_call @Sharding(%arg12) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<32x784xf32>) -> tensor<32x784xf32>
    %25 = stablehlo.custom_call @SPMDFullToShardShape(%24) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x784xf32>) -> tensor<16x196xf32>
    %26 = stablehlo.custom_call @Sharding(%arg13) {backend_config = "", mhlo.sharding = "{devices=[2,4]<=[8]}"} : (tensor<32x8xf32>) -> tensor<32x8xf32>
    %27 = stablehlo.custom_call @SPMDFullToShardShape(%26) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<32x8xf32>) -> tensor<16x2xf32>
    %28 = call @shmap_body(%1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27) : (tensor<98x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x128xf32>, tensor<16xf32>, tensor<16x8xf32>, tensor<1xf32>, tensor<16x196xf32>, tensor<16x2xf32>) -> tensor<f32>
    %29 = stablehlo.custom_call @Sharding(%28) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<f32>) -> tensor<f32>
    %30 = stablehlo.custom_call @SPMDShardToFullShape(%29) {backend_config = "", mhlo.sharding = "{replicated}"} : (tensor<f32>) -> tensor<f32>
    return %30 : tensor<f32>
  }
  func.func private @shmap_body(%arg0: tensor<98x128xf32>, %arg1: tensor<16xf32>, %arg2: tensor<16x128xf32>, %arg3: tensor<16xf32>, %arg4: tensor<16x128xf32>, %arg5: tensor<16xf32>, %arg6: tensor<16x128xf32>, %arg7: tensor<16xf32>, %arg8: tensor<16x128xf32>, %arg9: tensor<16xf32>, %arg10: tensor<16x8xf32>, %arg11: tensor<1xf32>, %arg12: tensor<16x196xf32>, %arg13: tensor<16x2xf32>) -> (tensor<f32> {jax.result_info = "[]"}) {
    %0 = "stablehlo.all_gather"(%arg0) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<98x128xf32>) -> tensor<196x128xf32>
    %1 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
    %2 = stablehlo.dot_general %arg12, %0, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x196xf32>, tensor<196x128xf32>) -> tensor<16x128xf32>
    %3 = "stablehlo.reduce_scatter"(%2) <{channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
    %4 = stablehlo.broadcast_in_dim %1, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
    %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %6 = stablehlo.add %3, %5 : tensor<16x32xf32>
    %7 = call @relu(%6) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %8 = "stablehlo.all_gather"(%arg2) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
    %9 = "stablehlo.all_gather"(%arg3) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
    %10 = stablehlo.dot_general %7, %8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
    %11 = "stablehlo.reduce_scatter"(%10) <{channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
    %12 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
    %13 = stablehlo.broadcast_in_dim %12, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %14 = stablehlo.add %11, %13 : tensor<16x32xf32>
    %15 = call @relu_0(%14) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %16 = "stablehlo.all_gather"(%arg4) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
    %17 = "stablehlo.all_gather"(%arg5) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
    %18 = stablehlo.dot_general %15, %16, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
    %19 = "stablehlo.reduce_scatter"(%18) <{channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
    %20 = stablehlo.broadcast_in_dim %17, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
    %21 = stablehlo.broadcast_in_dim %20, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %22 = stablehlo.add %19, %21 : tensor<16x32xf32>
    %23 = call @relu_1(%22) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %24 = "stablehlo.all_gather"(%arg6) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
    %25 = "stablehlo.all_gather"(%arg7) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
    %26 = stablehlo.dot_general %23, %24, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
    %27 = "stablehlo.reduce_scatter"(%26) <{channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
    %28 = stablehlo.broadcast_in_dim %25, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
    %29 = stablehlo.broadcast_in_dim %28, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %30 = stablehlo.add %27, %29 : tensor<16x32xf32>
    %31 = call @relu_2(%30) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %32 = "stablehlo.all_gather"(%arg8) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<32x128xf32>
    %33 = "stablehlo.all_gather"(%arg9) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 14, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<32xf32>
    %34 = stablehlo.dot_general %31, %32, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x128xf32>) -> tensor<16x128xf32>
    %35 = "stablehlo.reduce_scatter"(%34) <{channel_handle = #stablehlo.channel_handle<handle = 15, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16x128xf32>) -> tensor<16x32xf32>
    %36 = stablehlo.broadcast_in_dim %33, dims = [1] : (tensor<32xf32>) -> tensor<1x32xf32>
    %37 = stablehlo.broadcast_in_dim %36, dims = [0, 1] : (tensor<1x32xf32>) -> tensor<16x32xf32>
    %38 = stablehlo.add %35, %37 : tensor<16x32xf32>
    %39 = call @relu_3(%38) : (tensor<16x32xf32>) -> tensor<16x32xf32>
    %40 = "stablehlo.all_gather"(%arg10) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 16, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<16x8xf32>) -> tensor<32x8xf32>
    %41 = "stablehlo.all_gather"(%arg11) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 17, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> : (tensor<1xf32>) -> tensor<2xf32>
    %42 = stablehlo.dot_general %39, %40, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
    %43 = "stablehlo.reduce_scatter"(%42) <{channel_handle = #stablehlo.channel_handle<handle = 18, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, scatter_dimension = 1 : i64, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16x8xf32>) -> tensor<16x2xf32>
    %44 = stablehlo.broadcast_in_dim %41, dims = [1] : (tensor<2xf32>) -> tensor<1x2xf32>
    %45 = stablehlo.broadcast_in_dim %44, dims = [0, 1] : (tensor<1x2xf32>) -> tensor<16x2xf32>
    %46 = stablehlo.add %43, %45 : tensor<16x2xf32>
    %47 = stablehlo.subtract %46, %arg13 : tensor<16x2xf32>
    %48 = stablehlo.multiply %47, %47 : tensor<16x2xf32>
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %49 = stablehlo.reduce(%48 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<16x2xf32>, tensor<f32>) -> tensor<16xf32>
    %50 = "stablehlo.all_reduce"(%49) <{channel_handle = #stablehlo.channel_handle<handle = 19, type = 1>, replica_groups = dense<[[0, 1, 2, 3], [4, 5, 6, 7]]> : tensor<2x4xi64>, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<16xf32>) -> tensor<16xf32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %51 = stablehlo.reduce(%50 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<16xf32>, tensor<f32>) -> tensor<f32>
    %cst_1 = stablehlo.constant dense<1.600000e+01> : tensor<f32>
    %52 = stablehlo.divide %51, %cst_1 : tensor<f32>
    %53 = "stablehlo.all_reduce"(%52) <{channel_handle = #stablehlo.channel_handle<handle = 20, type = 1>, replica_groups = dense<[[0, 4], [1, 5], [2, 6], [3, 7]]> : tensor<4x2xi64>, use_global_device_ids}> ({
    ^bb0(%arg14: tensor<f32>, %arg15: tensor<f32>):
      %55 = stablehlo.add %arg14, %arg15 : tensor<f32>
      stablehlo.return %55 : tensor<f32>
    }) : (tensor<f32>) -> tensor<f32>
    %cst_2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %54 = stablehlo.divide %53, %cst_2 : tensor<f32>
    return %54 : tensor<f32>
  }
  func.func private @relu(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_0(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_1(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_2(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
  func.func private @relu_3(%arg0: tensor<16x32xf32>) -> tensor<16x32xf32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<16x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<16x32xf32>
    return %1 : tensor<16x32xf32>
  }
}

// CHECK-LABEL @main
