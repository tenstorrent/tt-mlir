// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_loss_fsdp attributes {mhlo.num_partitions = 8 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @mesh = <["x"=1, "y"=8]>
  func.func public @main(%arg0: tensor<784x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg1: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg2: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg3: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg4: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg5: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg6: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg7: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg8: tensor<128x128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg9: tensor<128xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg10: tensor<128x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg11: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, %arg12: tensor<32x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}, %arg13: tensor<32x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}, {}]>}) -> (tensor<f32> {jax.result_info = ""}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) in_shardings=[<@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}]>, <@mesh, [{"y"}, {}]>, <@mesh, [{"y"}, {}]>] out_shardings=[<@mesh, []>] manual_axes={"x", "y"} (%arg14: tensor<98x128xf32>, %arg15: tensor<16xf32>, %arg16: tensor<16x128xf32>, %arg17: tensor<16xf32>, %arg18: tensor<16x128xf32>, %arg19: tensor<16xf32>, %arg20: tensor<16x128xf32>, %arg21: tensor<16xf32>, %arg22: tensor<16x128xf32>, %arg23: tensor<16xf32>, %arg24: tensor<16x8xf32>, %arg25: tensor<1xf32>, %arg26: tensor<4x784xf32>, %arg27: tensor<4x8xf32>) {
      %1 = "stablehlo.all_gather"(%arg14) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<98x128xf32>) -> tensor<784x128xf32>
      %2 = "stablehlo.all_gather"(%arg15) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
      %3 = stablehlo.dot_general %arg26, %1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x784xf32>, tensor<784x128xf32>) -> tensor<4x128xf32>
      %4 = stablehlo.broadcast_in_dim %2, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %5 = stablehlo.broadcast_in_dim %4, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %6 = stablehlo.add %3, %5 : tensor<4x128xf32>
      %7 = func.call @relu(%6) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %8 = "stablehlo.all_gather"(%arg16) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
      %9 = "stablehlo.all_gather"(%arg17) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 4, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
      %10 = stablehlo.dot_general %7, %8, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %11 = stablehlo.broadcast_in_dim %9, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %12 = stablehlo.broadcast_in_dim %11, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %13 = stablehlo.add %10, %12 : tensor<4x128xf32>
      %14 = func.call @relu_0(%13) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %15 = "stablehlo.all_gather"(%arg18) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 5, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
      %16 = "stablehlo.all_gather"(%arg19) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 6, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
      %17 = stablehlo.dot_general %14, %15, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %18 = stablehlo.broadcast_in_dim %16, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %19 = stablehlo.broadcast_in_dim %18, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %20 = stablehlo.add %17, %19 : tensor<4x128xf32>
      %21 = func.call @relu_1(%20) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %22 = "stablehlo.all_gather"(%arg20) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 7, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
      %23 = "stablehlo.all_gather"(%arg21) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 8, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
      %24 = stablehlo.dot_general %21, %22, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %25 = stablehlo.broadcast_in_dim %23, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %26 = stablehlo.broadcast_in_dim %25, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %27 = stablehlo.add %24, %26 : tensor<4x128xf32>
      %28 = func.call @relu_2(%27) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %29 = "stablehlo.all_gather"(%arg22) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 9, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x128xf32>) -> tensor<128x128xf32>
      %30 = "stablehlo.all_gather"(%arg23) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 10, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16xf32>) -> tensor<128xf32>
      %31 = stablehlo.dot_general %28, %29, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x128xf32>) -> tensor<4x128xf32>
      %32 = stablehlo.broadcast_in_dim %30, dims = [1] : (tensor<128xf32>) -> tensor<1x128xf32>
      %33 = stablehlo.broadcast_in_dim %32, dims = [0, 1] : (tensor<1x128xf32>) -> tensor<4x128xf32>
      %34 = stablehlo.add %31, %33 : tensor<4x128xf32>
      %35 = func.call @relu_3(%34) : (tensor<4x128xf32>) -> tensor<4x128xf32>
      %36 = "stablehlo.all_gather"(%arg24) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 11, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<16x8xf32>) -> tensor<128x8xf32>
      %37 = "stablehlo.all_gather"(%arg25) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 12, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> : (tensor<1xf32>) -> tensor<8xf32>
      %38 = stablehlo.dot_general %35, %36, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<4x128xf32>, tensor<128x8xf32>) -> tensor<4x8xf32>
      %39 = stablehlo.broadcast_in_dim %37, dims = [1] : (tensor<8xf32>) -> tensor<1x8xf32>
      %40 = stablehlo.broadcast_in_dim %39, dims = [0, 1] : (tensor<1x8xf32>) -> tensor<4x8xf32>
      %41 = stablehlo.add %38, %40 : tensor<4x8xf32>
      %42 = stablehlo.subtract %41, %arg27 : tensor<4x8xf32>
      %43 = stablehlo.multiply %42, %42 : tensor<4x8xf32>
      %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %44 = stablehlo.reduce(%43 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<4x8xf32>, tensor<f32>) -> tensor<4xf32>
      %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
      %45 = stablehlo.reduce(%44 init: %cst_0) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> tensor<f32>
      %cst_1 = stablehlo.constant dense<4.000000e+00> : tensor<f32>
      %46 = stablehlo.divide %45, %cst_1 : tensor<f32>
      %47 = "stablehlo.all_reduce"(%46) <{channel_handle = #stablehlo.channel_handle<handle = 13, type = 1>, replica_groups = dense<[[0, 1, 2, 3, 4, 5, 6, 7]]> : tensor<1x8xi64>, use_global_device_ids}> ({
      ^bb0(%arg28: tensor<f32>, %arg29: tensor<f32>):
        %49 = stablehlo.add %arg28, %arg29 : tensor<f32>
        stablehlo.return %49 : tensor<f32>
      }) : (tensor<f32>) -> tensor<f32>
      %cst_2 = stablehlo.constant dense<8.000000e+00> : tensor<f32>
      %48 = stablehlo.divide %47, %cst_2 : tensor<f32>
      sdy.return %48 : tensor<f32>
    } : (tensor<784x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x128xf32>, tensor<128xf32>, tensor<128x8xf32>, tensor<8xf32>, tensor<32x784xf32>, tensor<32x8xf32>) -> tensor<f32>
    return %0 : tensor<f32>
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
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8, 1>
// CHECK-SAME: shard_type = #tt.shard_type<manual>
// CHECK: "ttir.mesh_shard"
// CHECK-SAME: shard_dims = array<i64: -1, 0>
// CHECK-SAME: shard_direction = #tt.shard_direction<full_to_shard>
// CHECK-SAME: shard_shape = array<i64: 8>
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
